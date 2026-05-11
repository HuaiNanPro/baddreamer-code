"""
VaViM video prediction on nuScenes mini dataset.
Input: nuScenes CAM_FRONT images
Output: Predicted future frames

Usage:
    python scripts/video_prediction_nuscenes_mini.py \
        --outdir ./output \
        --num_context 4 \
        --num_predict 4
"""

import argparse
import os
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2.functional as TF

from vam.video_pretraining import load_pretrained_gpt


# Image preprocessing must match the tokenizer's expected input
RESIZE_FACTOR = 3.125


def resize_by_factor(img, resize_factor):
    new_width = int(img.shape[2] / resize_factor)
    new_height = int(img.shape[1] / resize_factor)
    return TF.resize(img, (new_height, new_width), antialias=True)


def load_images_from_folder(folder, num_frames, subsampling=5):
    """Load sequence of images from folder, returns [T, C, H, W] in [-1, 1]."""
    folder = Path(folder)
    image_files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))

    # Subsample to get num_frames with gap of subsampling
    selected = image_files[:num_frames * subsampling:subsampling]
    selected = selected[:num_frames]

    frames = []
    for f in selected:
        img = Image.open(f).convert("RGB")
        img = TF.to_image(img)
        img = TF.to_dtype(img, torch.uint8, scale=True)
        img = resize_by_factor(img, RESIZE_FACTOR)
        img = TF.to_dtype(img, torch.float32, scale=True)
        img = 2.0 * img - 1.0
        frames.append(img)

    return torch.stack(frames, dim=0)


def save_image_grid(context_images, generated_images, output_dir, idx):
    """Save context + generated images side by side."""
    import matplotlib.pyplot as plt

    t_context = context_images.shape[0]
    t_pred = generated_images.shape[0]
    total = t_context + t_pred

    fig, axes = plt.subplots(2, max(t_context, t_pred), figsize=(4 * max(t_context, t_pred), 8))

    # Top row: context frames
    for i in range(t_context):
        img = context_images[i].permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Context {i}")
        axes[0, i].axis("off")

    # Bottom row: generated frames
    for i in range(t_pred):
        img = generated_images[i].permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Generated {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"result_{idx:04d}.png"), dpi=150)
    plt.close()
    print(f"Saved result_{idx:04d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_checkpoint", type=str,
                        default="/home/xiexiaopeng/sz/VideoActionModel/weight/VAM_width_768_pretrained_139k.pt")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/home/xiexiaopeng/sz/VideoActionModel/jit_models/VQ_ds16_16384_llamagen_encoder.jit")
    parser.add_argument("--detokenizer_path", type=str,
                        default="/home/xiexiaopeng/sz/VideoActionModel/jit_models/VQ_ds16_16384_llamagen_decoder.jit")
    parser.add_argument("--image_dir", type=str,
                        default="/home/xiexiaopeng/sz/data/nuscenes/v1.0-mini/samples/CAM_FRONT")
    parser.add_argument("--outdir", type=str, default="./nuscenes_mini_output")
    parser.add_argument("--num_context", type=int, default=4,
                        help="Number of context frames")
    parser.add_argument("--num_predict", type=int, default=4,
                        help="Number of frames to predict")
    parser.add_argument("--stride", type=int, default=5,
                        help="Stride between sequences")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VaViM Video Prediction on nuScenes Mini")
    print("=" * 60)

    # Load models
    print("\n[1/4] Loading VaViM model...")
    gpt = load_pretrained_gpt(args.gpt_checkpoint, device=device)
    gpt.eval()
    print(f"      VaViM loaded: embedding_dim={gpt.embedding_dim}, layers={gpt.nb_layers}")

    print("[2/4] Loading VQ tokenizer and detokenizer...")
    tokenizer = torch.jit.load(args.tokenizer_path).to(device)
    detokenizer = torch.jit.load(args.detokenizer_path).to(device)
    tokenizer.eval()
    detokenizer.eval()
    print("      Tokenizer and detokenizer loaded")

    # Load image list
    print(f"\n[3/4] Loading images from {args.image_dir}...")
    image_files = sorted(Path(args.image_dir).glob("*.jpg")) + sorted(Path(args.image_dir).glob("*.png"))
    total_images = len(image_files)
    print(f"      Found {total_images} images")

    # Calculate how many sequences we can generate
    seq_len = args.num_context + args.num_predict
    num_sequences = max(0, (total_images - seq_len) // args.stride + 1)
    print(f"      Will generate {num_sequences} sequences "
          f"(context={args.num_context}, predict={args.num_predict}, stride={args.stride})")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n[4/4] Generating predictions...")
    with torch.no_grad():
        for seq_idx in tqdm(range(num_sequences)):
            start_idx = seq_idx * args.stride
            end_idx = start_idx + seq_len
            selected_files = image_files[start_idx:end_idx]

            # Load and preprocess images
            frames = []
            for f in selected_files:
                img = Image.open(f).convert("RGB")
                img = TF.to_image(img)
                img = TF.to_dtype(img, torch.uint8, scale=True)
                img = resize_by_factor(img, RESIZE_FACTOR)
                img = TF.to_dtype(img, torch.float32, scale=True)
                img = 2.0 * img - 1.0
                frames.append(img)

            frames = torch.stack(frames, dim=0)  # [T, C, H, W]

            # Split into context and (unused) future for comparison
            context_frames = frames[:args.num_context].to(device)
            future_frames = frames[args.num_context:args.num_context + args.num_predict].to(device)

            # Tokenize context
            tokens = tokenizer(rearrange(context_frames, "t c h w -> (t) c h w"))
            tokens = rearrange(tokens, "(t) h w -> t h w", t=args.num_context)
            visual_tokens = tokens.unsqueeze(0)  # [1, T, H, W]

            # Generate future frames
            generated_tokens = gpt.forward_inference(
                number_of_future_frames=args.num_predict,
                burnin_visual_tokens=visual_tokens,
                temperature=1.0,
                topk_sampler=1,
                use_kv_cache=False,
            )

            # Detokenize
            gen_images = detokenizer(rearrange(generated_tokens, "b t h w -> (b t) h w"))
            gen_images = rearrange(gen_images, "(b t) h w c -> b t h w c", b=1, t=args.num_predict)
            gen_images = gen_images.squeeze(0)  # [T, H, W, C]

            # Save result
            save_image_grid(
                context_frames.cpu(),
                gen_images.cpu(),
                args.outdir,
                seq_idx
            )

    print(f"\n{'=' * 60}")
    print(f"Done! Results saved to: {args.outdir}/")
    print(f"Total sequences processed: {num_sequences}")
    print("=" * 60)


if __name__ == "__main__":
    main()
