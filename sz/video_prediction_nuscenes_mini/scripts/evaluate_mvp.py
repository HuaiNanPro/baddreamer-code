"""
Minimal MVP for FID evaluation - just 4 context + 4 prediction frames
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm

from vam.video_pretraining.mup_gpt2 import MupGPT2
from vam.datalib import CropAndResizeTransform
import mup

# nuScenes images are 1600x900
# With resize factor 3.125: 512x288
# This produces 18x32 = 576 tokens per frame (matching nb_tokens_per_timestep)
RESIZE_FACTOR = 3.125

# Use same transform as quality_evaluation.py
crop_and_resize = CropAndResizeTransform(resize_factor=RESIZE_FACTOR, trop_crop_size=0)


def load_vavim_checkpoint(checkpoint_path, device):
    """Load VAVIM GPT checkpoint with proper mup base shapes."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    hp = ckpt['hyper_parameters']
    config = hp['network'].copy()
    config.pop('_target_')

    network_state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if k.startswith('network.'):
            new_key = k.replace('network.', '')
            network_state_dict[new_key] = v

    mup_base_shapes = hp.get('mup_base_shapes')
    print(f"mup_base_shapes available: {mup_base_shapes is not None}")

    model = MupGPT2(**config)
    if mup_base_shapes is not None:
        print("Applying mup.set_base_shapes...")
        mup.set_base_shapes(model, mup_base_shapes, rescale_params=False)

    model.load_state_dict(network_state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: embedding_dim={model.embedding_dim}, layers={model.nb_layers}")

    return model


class DINOv2FeatureExtractor:
    """Extract features using DINOv2 ViT-B/14."""

    def __init__(self, device):
        print("Loading DINOv2 ViT-B/14 feature extractor...")
        try:
            base = os.path.expanduser(os.path.expandvars(os.environ.get("TORCH_HOME", "~/.cache")))
            self.model = torch.hub.load(
                os.path.join(base, "torch/hub/facebookresearch_dinov2_main/"),
                "dinov2_vitb14_reg",
                source="local"
            )
        except Exception:
            print("Local load failed, trying from PyTorch Hub...")
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", force_reload=True)

        self.model = self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        from torchvision import transforms
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        print("DINOv2 ViT-B/14 loaded successfully")

    def preprocess(self, images):
        """Preprocess images for DINOv2. images: [B, C, H, W] in [0, 1] float"""
        h = (images.size(-2) // 14) * 14
        w = (images.size(-1) // 14) * 14
        if h != images.size(-2) or w != images.size(-1):
            images = torch.nn.functional.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)
        images = self.normalize(images)
        return images

    @torch.no_grad()
    def extract(self, images):
        """Extract DINOv2 features. images: [B, C, H, W] in [0, 1] float"""
        images = self.preprocess(images)
        features = self.model(images)
        return features


def calculate_fid(features_real, features_fake):
    """Calculate FID between two sets of features."""
    mu_real = torch.mean(features_real, dim=0)
    mu_fake = torch.mean(features_fake, dim=0)

    # For single sample, use identity covariance with small epsilon
    if features_real.shape[0] < 2:
        sigma_real = torch.eye(features_real.shape[1], device=features_real.device) * 0.01
    else:
        sigma_real = torch.cov(features_real.T)

    if features_fake.shape[0] < 2:
        sigma_fake = torch.eye(features_fake.shape[1], device=features_fake.device) * 0.01
    else:
        sigma_fake = torch.cov(features_fake.T)

    diff = mu_real - mu_fake
    diff_norm_sq = torch.sum(diff ** 2).item()

    # Use scipy.linalg.sqrtm which is more compatible
    sigma_real_np = sigma_real.cpu().numpy()
    sigma_fake_np = sigma_fake.cpu().numpy()
    try:
        covmean = sqrtm(sigma_real_np @ sigma_fake_np)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        covmean = torch.from_numpy(covmean).to(features_real.device)
    except:
        covmean = torch.eye(features_real.shape[1], device=features_real.device) * 0.1

    tr_covsum = torch.trace(sigma_real + sigma_fake - 2 * covmean).item()
    fid = diff_norm_sq + tr_covsum
    return fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/raid/zengchaolv/sz/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/raid/zengchaolv/sz/video_prediction_nuscenes_mini/jit_models/VQ_ds16_16384_llamagen_encoder.jit")
    parser.add_argument("--detokenizer_path", type=str,
                        default="/raid/zengchaolv/sz/video_prediction_nuscenes_mini/jit_models/VQ_ds16_16384_llamagen_decoder.jit")
    parser.add_argument("--image_dir", type=str,
                        default="/raid/zengchaolv/sz/nuscenes/Nuscenes_trainval_v1.0/v1.0-trainval08_blobs/CAM_FRONT")
    parser.add_argument("--outdir", type=str, default="./nuscenes_mini_output")
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--num_predict", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VaViM Video Prediction Quality Evaluation (Minimal MVP)")
    print("=" * 60)

    print("\n[1/5] Loading VaViM model...")
    gpt = load_vavim_checkpoint(args.checkpoint, device)

    print("\n[2/5] Loading VQ tokenizer and detokenizer...")
    tokenizer = torch.jit.load(args.tokenizer_path).to(device)
    detokenizer = torch.jit.load(args.detokenizer_path).to(device)
    tokenizer.eval()
    detokenizer.eval()

    print("\n[3/5] Loading DINOv2 ViT-B/14 feature extractor...")
    dinov2 = DINOv2FeatureExtractor(device)

    print(f"\n[4/5] Loading images from {args.image_dir}...")
    image_files = sorted(Path(args.image_dir).glob("*.jpg")) + sorted(Path(args.image_dir).glob("*.png"))
    print(f"      Found {len(image_files)} images")

    # Take first 8 images: 4 context + 4 future
    selected_files = image_files[:args.num_context + args.num_predict]
    print(f"      Using first {len(selected_files)} images")

    print("\n[5/5] Processing frames...")

    frames = []
    for f in selected_files:
        img = Image.open(f).convert("RGB")
        img = TF.to_image(img)
        img = TF.to_dtype(img, torch.float32, scale=True)  # [0, 1] float
        # Use same transform as dataset
        img = crop_and_resize(img)
        frames.append(img)

    frames = torch.stack(frames, dim=0)  # [T, C, H, W]
    context_frames = frames[:args.num_context].to(device)
    future_frames = frames[args.num_context:].to(device)

    # Tokenize context (tokenizer expects [0, 1] float)
    tokens = tokenizer(rearrange(context_frames, 't c h w -> (t) c h w'))
    tokens = rearrange(tokens, '(t) h w -> t h w', t=args.num_context)
    visual_tokens = tokens.unsqueeze(0)

    # Generate future frames
    print("      Generating future frames...")
    generated_tokens = gpt.forward_inference(
        number_of_future_frames=args.num_predict,
        burnin_visual_tokens=visual_tokens,
        temperature=1.0,
        topk_sampler=1,
        use_kv_cache=False,
    )

    # Detokenize
    gen_images = detokenizer(rearrange(generated_tokens, 'b t h w -> (b t) h w'))
    gen_images = rearrange(gen_images, '(t) c h w -> t c h w', t=args.num_predict)

    # Save images
    output_images = os.path.join(args.outdir, "mvp_images")
    os.makedirs(output_images, exist_ok=True)
    for t in range(args.num_predict):
        # Transform outputs in [-1, 1], convert to [0, 1]
        ctx_img = context_frames[t].float().cpu().add(1).div(2).clamp(0, 1)
        real_img = future_frames[t].float().cpu().add(1).div(2).clamp(0, 1)
        gen_img = gen_images[t].float().cpu().add(1).div(2).clamp(0, 1)

        ctx_np = (ctx_img.permute(1, 2, 0).numpy() * 255).astype('uint8')
        Image.fromarray(ctx_np).save(os.path.join(output_images, f"context_{t+1}.png"))

        real_np = (real_img.permute(1, 2, 0).numpy() * 255).astype('uint8')
        Image.fromarray(real_np).save(os.path.join(output_images, f"gt_future_{t+1}.png"))

        gen_np = (gen_img.permute(1, 2, 0).numpy() * 255).astype('uint8')
        Image.fromarray(gen_np).save(os.path.join(output_images, f"predicted_{t+1}.png"))

    print("      Images saved")

    # Calculate FID@t
    print("\n" + "=" * 60)
    print("FID@t RESULTS")
    print("=" * 60)

    fid_scores = {}
    for t in range(args.num_predict):
        # Convert from [-1, 1] to [0, 1] for DINOv2
        real_input = future_frames[t].unsqueeze(0).float().add(1).div(2).clamp(0, 1)
        fake_input = gen_images[t].unsqueeze(0).float().add(1).div(2).clamp(0, 1)

        real_feat = dinov2.extract(real_input)
        fake_feat = dinov2.extract(fake_input)

        fid = calculate_fid(real_feat, fake_feat)
        fid_scores[f"FID@{t+1}"] = fid
        print(f"FID@{t+1}: {fid:.4f}")

    overall_fid = sum(fid_scores.values()) / len(fid_scores)
    print("-" * 40)
    print(f"Average: {overall_fid:.4f}")
    print("=" * 60)
    print(f"\nImages saved to: {output_images}/")


if __name__ == "__main__":
    main()