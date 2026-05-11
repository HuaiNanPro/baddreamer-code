"""
Video Generation Quality Evaluation Script for VAVIM
使用 DINOv2特征的逐帧 FID (FID@t) 评估，与论文一致

Usage:
    python scripts/evaluate_vavim_quality.py --max_samples 30
"""

import argparse
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, "/raid/zengchaolv/sz/vavim_finetune_nuscenes")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

from vam.video_pretraining.mup_gpt2 import MupGPT2
import mup


RESIZE_FACTOR = 3.125


def resize_by_factor(img, resize_factor):
    new_width = int(img.shape[2] / resize_factor)
    new_height = int(img.shape[1] / resize_factor)
    return TF.resize(img, (new_height, new_width), antialias=True)


def load_vavim_checkpoint(checkpoint_path, device):
    """Load VAVIM GPT checkpoint with proper mup base shapes."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if 'hyper_parameters' not in ckpt:
        raise ValueError("Checkpoint doesn't have hyper_parameters")

    hp = ckpt['hyper_parameters']
    if 'network' not in hp:
        raise ValueError("Checkpoint doesn't have 'network' in hyper_parameters")

    config = hp['network'].copy()
    config.pop('_target_')

    network_state_dict = OrderedDict()
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

        self.register_norm()
        print("DINOv2 ViT-B/14 loaded successfully")

    def register_norm(self):
        from torchvision import transforms
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def preprocess(self, images):
        """Preprocess images for DINOv2. images: [B, C, H, W] in [0, 255] uint8"""
        # images should be [0, 255] uint8
        images = images.float() / 255.0

        # Resize to closest size divisible by 14
        h = (images.size(-2) // 14) * 14
        w = (images.size(-1) // 14) * 14
        if h != images.size(-2) or w != images.size(-1):
            images = torch.nn.functional.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)

        images = self.normalize(images)
        return images

    @torch.no_grad()
    def extract(self, images):
        """Extract DINOv2 features. images: [B, C, H, W] in [0, 255] uint8"""
        images = self.preprocess(images)
        features = self.model(images)
        return features


def calculate_fid(features_real, features_fake):
    """Calculate FID between two sets of features."""
    import scipy

    mu_real = torch.mean(features_real, dim=0)
    mu_fake = torch.mean(features_fake, dim=0)

    sigma_real = torch.cov(features_real.T)
    sigma_fake = torch.cov(features_fake.T)

    diff = mu_real - mu_fake
    diff_norm_sq = torch.sum(diff ** 2).item()

    # Use scipy for sqrtm (torch.linalg.sqrtm not available in older PyTorch)
    sigma_real_np = sigma_real.cpu().numpy()
    sigma_fake_np = sigma_fake.cpu().numpy()
    covmean_np = scipy.linalg.sqrtm(sigma_real_np @ sigma_fake_np)
    covmean = torch.from_numpy(covmean_np.real).to(features_real.device)

    tr_covsum = torch.trace(sigma_real + sigma_fake - 2 * covmean).item()

    fid = diff_norm_sq + tr_covsum
    return fid


def save_image(img_tensor, path):
    """Save image tensor [C, H, W] in [0, 255] uint8 to file."""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')
    Image.fromarray(img_np).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/raid/zengchaolv/sz/vavim_finetune_nuscenes/checkpoints/epoch=1-step=2136.ckpt")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/raid/zengchaolv/sz/vavim_finetune_nuscenes/fid_eval/jit_models/VQ_ds16_16384_llamagen_encoder.jit")
    parser.add_argument("--detokenizer_path", type=str,
                        default="/raid/zengchaolv/sz/vavim_finetune_nuscenes/fid_eval/jit_models/VQ_ds16_16384_llamagen_decoder.jit")
    parser.add_argument("--image_dir", type=str,
                        default="/raid/zengchaolv/sz/nuscenes/Nuscenes_trainval_v1.0/v1.0-trainval08_blobs/CAM_FRONT")
    parser.add_argument("--outdir", type=str, default="/raid/zengchaolv/sz/vavim_finetune_nuscenes/fid_eval/output")
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--num_predict", type=int, default=4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VaViM Video Prediction Quality Evaluation (FID@t with DINOv2)")
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
    total_images = len(image_files)
    print(f"      Found {total_images} images")

    seq_len = args.num_context + args.num_predict
    num_sequences = max(0, (total_images - seq_len) // args.stride + 1)
    num_sequences = min(num_sequences, args.max_samples)
    print(f"      Will evaluate {num_sequences} sequences")

    # Output directories
    output_images = os.path.join(args.outdir, "fid_eval_images")
    os.makedirs(output_images, exist_ok=True)

    print(f"\n[5/5] Running FID@t evaluation...")

    per_frame_features_real = {i: [] for i in range(args.num_predict)}
    per_frame_features_fake = {i: [] for i in range(args.num_predict)}

    sample_idx = 0

    with torch.no_grad():
        for seq_idx in tqdm(range(num_sequences)):
            start_idx = seq_idx * args.stride
            end_idx = start_idx + seq_len
            selected_files = image_files[start_idx:end_idx]

            # Load and preprocess images: [T, C, H, W] in [0, 255] uint8
            frames = []
            for f in selected_files:
                img = Image.open(f).convert("RGB")
                img = TF.to_image(img)
                img = TF.to_dtype(img, torch.uint8, scale=True)
                img = resize_by_factor(img, RESIZE_FACTOR)
                frames.append(img)

            frames = torch.stack(frames, dim=0)  # [T, C, H, W] uint8

            context_frames = frames[:args.num_context].to(device).float() / 255.0
            future_frames = frames[args.num_context:].to(device)

            # Tokenize context
            tokens = tokenizer(rearrange(context_frames, 't c h w -> (t) c h w'))
            tokens = rearrange(tokens, '(t) h w -> t h w', t=args.num_context)
            visual_tokens = tokens.unsqueeze(0)

            # Generate future frames
            generated_tokens = gpt.forward_inference(
                number_of_future_frames=args.num_predict,
                burnin_visual_tokens=visual_tokens,
                temperature=1.0,
                topk_sampler=1,
                use_kv_cache=False,
            )

            # Detokenize: [4, C, H, W] in [0, 255] uint8
            gen_images = detokenizer(rearrange(generated_tokens, 'b t h w -> (b t) h w'))
            gen_images = rearrange(gen_images, '(t) c h w -> t c h w', t=args.num_predict)

            # Save first sample's images for inspection
            if seq_idx == 0:
                for t in range(args.num_predict):
                    ctx_img = (context_frames[t].float() * 255).clamp(0, 255).to(torch.uint8)
                    real_img = (future_frames[t].float() * 255).clamp(0, 255).to(torch.uint8)
                    gen_img = (gen_images[t].float() * 255).clamp(0, 255).to(torch.uint8)
                    save_image(ctx_img, os.path.join(output_images, f"context_{t+1}.png"))
                    save_image(real_img, os.path.join(output_images, f"gt_future_{t+1}.png"))
                    save_image(gen_img, os.path.join(output_images, f"predicted_{t+1}.png"))

            # Extract DINOv2 features for each frame
            for t in range(args.num_predict):
                real_frame = (future_frames[t].float() * 255).clamp(0, 255).to(torch.uint8).to(device)
                real_feat = dinov2.extract(real_frame.unsqueeze(0))
                per_frame_features_real[t].append(real_feat.cpu())

                fake_frame = (gen_images[t].float() * 255).clamp(0, 255).to(torch.uint8).to(device)
                fake_feat = dinov2.extract(fake_frame.unsqueeze(0))
                per_frame_features_fake[t].append(fake_feat.cpu())

    # Calculate FID@t for each frame
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS - FID@t (DINOv2 ViT-B/14 Features)")
    print("=" * 60)
    print(f"Number of samples: {num_sequences}")
    print("\n" + "-" * 40)
    print(f"{'Metric':<15} {'Value':>15}")
    print("-" * 40)

    fid_scores = {}
    for t in range(args.num_predict):
        real_features = torch.cat(per_frame_features_real[t], dim=0)
        fake_features = torch.cat(per_frame_features_fake[t], dim=0)

        fid = calculate_fid(real_features, fake_features)
        fid_scores[f"FID@{t+1}"] = fid
        print(f"FID@{t+1:<5} {fid:>15.4f}")

    overall_fid = sum(fid_scores.values()) / len(fid_scores)
    fid_scores["Average"] = overall_fid
    print("-" * 40)
    print(f"{'Average':<15} {overall_fid:>15.4f}")
    print("=" * 60)

    # Save results
    results = {
        "fid_scores": fid_scores,
        "overall_fid": overall_fid,
        "num_samples": num_sequences,
    }

    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, "fid_evaluation_results.pt")
    torch.save(results, results_path)
    print(f"\nResults saved to: {results_path}")
    print(f"Example images saved to: {output_images}/")
    print("=" * 60)


if __name__ == "__main__":
    main()