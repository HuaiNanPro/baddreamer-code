"""
Video Generation Quality Evaluation Script
Evaluates: PSNR, SSIM, LPIPS for predicted frames vs ground truth

Usage:
    python scripts/evaluate_video_quality.py \
        --num_context 4 \
        --num_predict 4 \
        --stride 5 \
        --max_samples 50 \
        --device cuda
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from vam.video_pretraining import load_pretrained_gpt


RESIZE_FACTOR = 3.125


def resize_by_factor(img, resize_factor):
    new_width = int(img.shape[2] / resize_factor)
    new_height = int(img.shape[1] / resize_factor)
    return TF.resize(img, (new_height, new_width), antialias=True)


def calculate_psnr(img1, img2, max_val=2.0):
    """Calculate PSNR between two images. Images are in [-1, 1] range."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, max_val=2.0):
    """Calculate SSIM between two images. Images are in [-1, 1] range."""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    img1 = img1.unsqueeze(0).unsqueeze(0)
    img2 = img2.unsqueeze(0).unsqueeze(0)

    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d((img1 - mu1) ** 2, window_size, stride=1, padding=window_size//2)
    sigma2_sq = torch.nn.functional.avg_pool2d((img2 - mu2) ** 2, window_size, stride=1, padding=window_size//2)
    sigma12 = torch.nn.functional.avg_pool2d((img1 - mu1) * (img2 - mu2), window_size, stride=1, padding=window_size//2)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


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
    parser.add_argument("--num_context", type=int, default=4, help="Number of context frames")
    parser.add_argument("--num_predict", type=int, default=4, help="Number of frames to predict")
    parser.add_argument("--stride", type=int, default=5, help="Stride between sequences")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VaViM Video Prediction Quality Evaluation")
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

    # Calculate how many sequences
    seq_len = args.num_context + args.num_predict
    num_sequences = max(0, (total_images - seq_len) // args.stride + 1)
    num_sequences = min(num_sequences, args.max_samples)
    print(f"      Will evaluate {num_sequences} sequences")

    print(f"\n[4/4] Evaluating predictions...")

    all_psnr = []
    all_ssim = []

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

            # Split into context and future
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

            # Convert to [-1, 1] if needed (detokenizer outputs [0, 255])
            if gen_images.max() > 1.0:
                gen_images = gen_images / 127.5 - 1.0
            else:
                gen_images = gen_images * 2 - 1  # Convert [0,1] to [-1,1]

            future_frames_normalized = future_frames
            if future_frames.max() <= 1.0:
                future_frames_normalized = future_frames * 2 - 1

            # Calculate metrics for each predicted frame
            for t in range(args.num_predict):
                pred_frame = gen_images[t].permute(2, 0, 1)  # [C, H, W]
                gt_frame = future_frames_normalized[t]

                psnr = calculate_psnr(pred_frame, gt_frame)
                ssim = calculate_ssim(pred_frame, gt_frame)

                all_psnr.append(psnr)
                all_ssim.append(ssim)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of samples evaluated: {num_sequences}")
    print(f"Number of frames evaluated: {len(all_psnr)}")
    print(f"\nMetrics:")
    print(f"  PSNR: {sum(all_psnr)/len(all_psnr):.4f} (higher is better)")
    print(f"  SSIM: {sum(all_ssim)/len(all_ssim):.4f} (higher is better, max=1.0)")
    print("\nPer-frame breakdown:")
    for t in range(args.num_predict):
        start_idx = t
        end_idx = len(all_psnr)
        step = args.num_predict
        frame_psnr = [all_psnr[i] for i in range(start_idx, end_idx, step)]
        frame_ssim = [all_ssim[i] for i in range(start_idx, end_idx, step)]
        print(f"  Frame {t+1}: PSNR={sum(frame_psnr)/len(frame_psnr):.4f}, SSIM={sum(frame_ssim)/len(frame_ssim):.4f}")

    # Save results
    results = {
        "psnr_mean": sum(all_psnr)/len(all_psnr),
        "psnr_all": all_psnr,
        "ssim_mean": sum(all_ssim)/len(all_ssim),
        "ssim_all": all_ssim,
        "num_samples": num_sequences,
    }

    os.makedirs(args.outdir, exist_ok=True)
    torch.save(results, os.path.join(args.outdir, "evaluation_results.pt"))
    print(f"\nResults saved to: {args.outdir}/evaluation_results.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()