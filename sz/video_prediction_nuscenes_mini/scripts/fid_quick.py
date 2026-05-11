"""
快速 FID 计算脚本 - 使用已保存的图像
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from einops import rearrange
from tqdm import tqdm


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
        """Preprocess images for DINOv2."""
        images = images.float() / 255.0
        h = (images.size(-2) // 14) * 14
        w = (images.size(-1) // 14) * 14
        if h != images.size(-2) or w != images.size(-1):
            images = torch.nn.functional.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)
        images = self.normalize(images)
        return images

    @torch.no_grad()
    def extract(self, images):
        """Extract DINOv2 features."""
        images = self.preprocess(images)
        features = self.model(images)
        return features


import scipy


def calculate_fid(features_real, features_fake):
    """Calculate FID between two sets of features."""
    mu_real = torch.mean(features_real, dim=0)
    mu_fake = torch.mean(features_fake, dim=0)

    sigma_real = torch.cov(features_real.T)
    sigma_fake = torch.cov(features_fake.T)

    diff = mu_real - mu_fake
    diff_norm_sq = torch.sum(diff ** 2).item()

    # Use scipy for sqrtm
    sigma_real_np = sigma_real.cpu().numpy()
    sigma_fake_np = sigma_fake.cpu().numpy()
    covmean_np = scipy.linalg.sqrtm(sigma_real_np @ sigma_fake_np)
    covmean = torch.from_numpy(covmean_np.real).to(features_real.device)

    tr_covsum = torch.trace(sigma_real + sigma_fake - 2 * covmean).item()

    fid = diff_norm_sq + tr_covsum
    return fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str,
                        default="/raid/zengchaolv/sz/video_prediction_nuscenes_mini/nuscenes_mini_output/fid_samples")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_predict", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="/raid/zengchaolv/sz/video_prediction_nuscenes_mini/nuscenes_mini_output")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DINOv2
    dinov2 = DINOv2FeatureExtractor(device)

    # Find all sample indices - files are flat with pattern sample_XXXX_*.png
    all_files = list(Path(args.image_dir).glob("*.png"))
    sample_indices = sorted(set([int(f.name.split("_")[1]) for f in all_files if f.name.startswith("sample_")]))
    num_samples = min(args.num_samples, len(sample_indices))
    sample_indices = sample_indices[:num_samples]
    print(f"Found {num_samples} samples: indices {sample_indices[:5]}...")

    per_frame_features_real = {i: [] for i in range(args.num_predict)}
    per_frame_features_fake = {i: [] for i in range(args.num_predict)}

    from PIL import Image
    import torchvision.transforms.v2.functional as TF

    for sample_idx in tqdm(sample_indices, desc="Extracting features"):
        for t in range(args.num_predict):
            # Load real future frame
            gt_path = Path(args.image_dir) / f"sample_{sample_idx:04d}_gt_{t+1}.png"
            if gt_path.exists():
                img = Image.open(gt_path).convert("RGB")
                img = TF.to_image(img)
                img = TF.to_dtype(img, torch.uint8, scale=True).to(device)
                real_feat = dinov2.extract(img.unsqueeze(0))
                per_frame_features_real[t].append(real_feat.cpu())

            # Load predicted frame
            pred_path = Path(args.image_dir) / f"sample_{sample_idx:04d}_pred_{t+1}.png"
            if pred_path.exists():
                img = Image.open(pred_path).convert("RGB")
                img = TF.to_image(img)
                img = TF.to_dtype(img, torch.uint8, scale=True).to(device)
                fake_feat = dinov2.extract(img.unsqueeze(0))
                per_frame_features_fake[t].append(fake_feat.cpu())

    # Calculate FID
    print("\n" + "=" * 60)
    print("FID RESULTS (Baseline)")
    print("=" * 60)

    fid_scores = {}
    for t in range(args.num_predict):
        if per_frame_features_real[t] and per_frame_features_fake[t]:
            real_features = torch.cat(per_frame_features_real[t], dim=0)
            fake_features = torch.cat(per_frame_features_fake[t], dim=0)

            fid = calculate_fid(real_features, fake_features)
            fid_scores[f"FID@{t+1}"] = fid
            print(f"FID@{t+1}: {fid:.2f}")

    # Save results
    import json
    results = {
        "nuscenes": fid_scores,
        "num_samples": num_samples,
    }
    out_path = os.path.join(args.outdir, "fid_results_quick.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()