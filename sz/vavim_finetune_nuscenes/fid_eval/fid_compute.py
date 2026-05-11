"""
优化的 FID 计算脚本 - 使用已保存的预测帧图像
直接读取已生成的预测帧，跳过推理
"""
import argparse
import os
import sys
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from tqdm import tqdm
from PIL import Image
import scipy


class DINOv2FeatureExtractor:
    """Extract features using DINOv2 ViT-B/14."""

    def __init__(self, device):
        print("Loading DINOv2...")
        try:
            base = os.path.expanduser(os.path.expandvars(os.environ.get("TORCH_HOME", "~/.cache")))
            self.model = torch.hub.load(
                os.path.join(base, "torch/hub/facebookresearch_dinov2_main/"),
                "dinov2_vitb14_reg",
                source="local"
            )
        except Exception:
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", force_reload=True)

        self.model = self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        from torchvision import transforms
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def preprocess(self, images):
        images = images.float() / 255.0
        h = (images.size(-2) // 14) * 14
        w = (images.size(-1) // 14) * 14
        if h != images.size(-2) or w != images.size(-1):
            images = torch.nn.functional.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)
        return self.normalize(images)

    @torch.no_grad()
    def extract(self, images):
        return self.model(self.preprocess(images))


def calculate_fid(features_real, features_fake):
    mu_real = torch.mean(features_real, dim=0)
    mu_fake = torch.mean(features_fake, dim=0)

    sigma_real = torch.cov(features_real.T)
    sigma_fake = torch.cov(features_fake.T)

    diff = mu_real - mu_fake
    diff_norm_sq = torch.sum(diff ** 2).item()

    sigma_real_np = sigma_real.cpu().numpy()
    sigma_fake_np = sigma_fake.cpu().numpy()
    covmean_np = scipy.linalg.sqrtm(sigma_real_np @ sigma_fake_np)
    covmean = torch.from_numpy(covmean_np.real).to(features_real.device)

    tr_covsum = torch.trace(sigma_real + sigma_fake - 2 * covmean).item()
    return diff_norm_sq + tr_covsum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str,
                      default="/raid/zengchaolv/sz/video_prediction_nuscenes_mini/nuscenes_mini_output/mvp_images")
    parser.add_argument("--outdir", type=str,
                      default="/raid/zengchaolv/sz/video_prediction_nuscenes_mini/nuscenes_mini_output")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    image_dir = Path(args.image_dir)

    # Find all sample indices - check both naming conventions
    has_sample_prefix = len(list(image_dir.glob("sample_*_gt_1.png"))) > 0

    per_frame_real = {i: [] for i in range(4)}
    per_frame_fake = {i: [] for i in range(4)}

    dinov2 = DINOv2FeatureExtractor(device)

    # Process based on naming convention
    if has_sample_prefix:
        all_files = list(image_dir.glob("sample_*_gt_1.png"))
        sample_indices = sorted([int(f.name.split("_")[1]) for f in all_files])[:args.num_samples]
    else:
        # Single sample with gt_*.png and pred_*.png
        sample_indices = [0] if image_dir.glob("gt_1.png") else []

    print(f"Using {len(sample_indices)} samples: {image_dir}")

    print(f"Using {len(sample_indices)} samples")

    dinov2 = DINOv2FeatureExtractor(device)

    per_frame_real = {i: [] for i in range(4)}
    per_frame_fake = {i: [] for i in range(4)}

    for sample_idx in tqdm(sample_indices, desc="Extracting"):
        for t in range(4):
            if has_sample_prefix:
                gt_path = image_dir / f"sample_{sample_idx:04d}_gt_{t+1}.png"
                pred_path = image_dir / f"sample_{sample_idx:04d}_pred_{t+1}.png"
            else:
                gt_path = image_dir / f"gt_future_{t+1}.png"
                pred_path = image_dir / f"predicted_{t+1}.png"

            if gt_path.exists():
                img = Image.open(gt_path).convert("RGB")
                img = TF.to_image(img)
                img = TF.to_dtype(img, torch.uint8, scale=True).to(device)
                feat = dinov2.extract(img.unsqueeze(0))
                per_frame_real[t].append(feat.cpu())

            if pred_path.exists():
                img = Image.open(pred_path).convert("RGB")
                img = TF.to_image(img)
                img = TF.to_dtype(img, torch.uint8, scale=True).to(device)
                feat = dinov2.extract(img.unsqueeze(0))
                per_frame_fake[t].append(feat.cpu())

    # Calculate FID
    results = {}
    print("\n" + "=" * 50)
    print("FID RESULTS")
    print("=" * 50)

    for t in range(4):
        real_feat = torch.cat(per_frame_real[t], dim=0)
        fake_feat = torch.cat(per_frame_fake[t], dim=0)
        fid = calculate_fid(real_feat, fake_feat)
        results[f"FID@{t+1}"] = fid
        print(f"FID@{t+1}: {fid:.2f}")

    # Save
    output = {"nuscenes": results, "num_samples": len(sample_indices)}
    out_path = Path(args.outdir) / "fid_results_confirmed.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()