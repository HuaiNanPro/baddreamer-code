#!/usr/bin/env python3
"""
Video Generation Inference Script using VAVIM pretrained model.

This script generates future frames from input frames using the Video Action Model.
It loads:
- The GPT2-based video prediction model from a checkpoint
- The image tokenizer (encoder) to tokenize input images
- The image detokenizer (decoder) to reconstruct images from tokens

Usage:
    python generate_video.py --input_frames /path/to/frames/ --output_dir /path/to/output/ --num_future_frames 8

Or with specific paths:
    python generate_video.py \
        --checkpoint /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt \
        --encoder /raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_encoder.jit \
        --decoder /raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_decoder.jit \
        --input_frames /path/to/frames/ \
        --output_dir /path/to/output/ \
        --num_future_frames 8
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm


def load_mup_gpt2(checkpoint_path: str, device: torch.device | str = "cpu"):
    """Load the pretrained MupGPT2 model from checkpoint."""
    import mup
    from omegaconf import DictConfig

    from vam.video_pretraining.mup_gpt2 import MupGPT2

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = ckpt["hyper_parameters"]["network"].copy()
    config.pop("_target_")

    model = MupGPT2(**config)

    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        new_key = k.replace("network.", "")
        if new_key not in state_dict:
            state_dict[new_key] = v

    model.load_state_dict(state_dict)

    mup_base_shapes = ckpt["hyper_parameters"].get("mup_base_shapes")
    if mup_base_shapes:
        mup.set_base_shapes(model, mup_base_shapes, rescale_params=False)

    model.eval()
    model.to(device)
    model.requires_grad_(False)

    return model, config


class VideoGenerator:
    """Video generation class using VAVIM model."""

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading encoder from {encoder_path}...")
        self.encoder = torch.jit.load(encoder_path)
        self.encoder.to(self.device)
        self.encoder.eval()

        print(f"Loading decoder from {decoder_path}...")
        self.decoder = torch.jit.load(decoder_path)
        self.decoder.to(self.device)
        self.decoder.eval()

        print(f"Loading model from {checkpoint_path}...")
        self.model, self.model_config = load_mup_gpt2(checkpoint_path, self.device)

        self.height = 18  # Token height (18x32 = 576 tokens per frame)
        self.width = 32  # Token width
        self.nb_tokens_per_timestep = self.height * self.width  # 576

        print(f"Model loaded. Config: {self.model_config}")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess an image for the tokenizer."""
        import torchvision.transforms.v2.functional as TF

        # Resize for opendv dataset (resize_factor=1.0)
        image = image.convert("RGB")
        img_tensor = TF.to_image(image)
        img_tensor = TF.to_dtype(img_tensor, torch.uint8, scale=True)
        # No resize for now - using original size
        img_tensor = TF.to_dtype(img_tensor, torch.float32, scale=True)
        img_tensor = 2.0 * img_tensor - 1.0  # Normalize to [-1, 1]
        return img_tensor

    def postprocess_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens back to images using the decoder."""
        with torch.amp.autocast(self.device, dtype=self.dtype):
            images = self.decoder(tokens)
        return images

    def tokenize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Tokenize images using the encoder."""
        with torch.amp.autocast(self.device, dtype=self.dtype):
            tokens = self.encoder(images)
        return tokens

    def load_input_frames(self, frames_dir: Path) -> List[Image.Image]:
        """Load input frames from a directory."""
        frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(frames_dir.glob("*.jpeg"))

        frames = []
        for frame_file in frame_files:
            img = Image.open(frame_file).convert("RGB")
            frames.append(img)

        print(f"Loaded {len(frames)} frames from {frames_dir}")
        return frames

    @torch.no_grad()
    def generate_future_frames(
        self,
        input_frames: List[Image.Image],
        num_future_frames: int = 8,
        temperature: float = 1.0,
        topk: int = 1,
        verbose: int = 1,
    ) -> List[Image.Image]:
        """Generate future frames from input frames."""
        # Preprocess and tokenize input frames
        preprocessed = []
        for frame in input_frames:
            tensor = self.preprocess_image(frame)
            preprocessed.append(tensor)

        # Stack and move to device
        input_batch = torch.stack(preprocessed, dim=0).to(self.device)

        # Tokenize
        visual_tokens = self.tokenize_images(input_batch)
        visual_tokens = rearrange(visual_tokens, "b c h w -> b c h w")

        # Generate using the model's inference
        generated_tokens = self.model.forward_inference(
            number_of_future_frames=num_future_frames,
            burnin_visual_tokens=visual_tokens,
            temperature=temperature,
            topk_sampler=topk,
            use_kv_cache=True,
            verbose=verbose,
        )

        # Detokenize generated frames
        generated_tokens = rearrange(generated_tokens, "b t h w -> b t h w")

        output_frames = []
        for frame_idx in range(num_future_frames):
            frame_tokens = generated_tokens[0, frame_idx]
            frame_tokens = frame_tokens.unsqueeze(0)  # Add batch dim

            # Decode tokens to image
            with torch.amp.autocast(self.device, dtype=self.dtype):
                decoded = self.decoder(frame_tokens)

            # Convert to PIL Image
            decoded = decoded[0]  # Remove batch dim
            decoded = decoded.cpu().float()
            decoded = (decoded + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            decoded = torch.clamp(decoded, 0, 1)
            decoded = rearrange(decoded, "c h w -> h w c")
            decoded = (decoded * 255).numpy().astype(np.uint8)

            img = Image.fromarray(decoded)
            output_frames.append(img)

        return output_frames


def main():
    parser = argparse.ArgumentParser(description="Video generation using VAVIM model")
    parser.add_argument("--checkpoint", type=str,
                      default="/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt",
                      help="Path to model checkpoint")
    parser.add_argument("--encoder", type=str,
                      default="/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_encoder.jit",
                      help="Path to encoder model")
    parser.add_argument("--decoder", type=str,
                      default="/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_decoder.jit",
                      help="Path to decoder model")
    parser.add_argument("--input_frames", type=str, required=True,
                      help="Path to input frames directory")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Path to output directory for generated frames")
    parser.add_argument("--num_future_frames", type=int, default=8,
                      help="Number of future frames to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=1,
                      help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use")
    parser.add_argument("--verbose", type=int, default=1,
                      help="Verbosity level")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = VideoGenerator(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Load input frames
    input_dir = Path(args.input_frames)
    input_frames = generator.load_input_frames(input_dir)

    if len(input_frames) < 2:
        raise ValueError(f"Need at least 2 input frames, got {len(input_frames)}")

    # Generate future frames
    print(f"\nGenerating {args.num_future_frames} future frames...")
    future_frames = generator.generate_future_frames(
        input_frames=input_frames,
        num_future_frames=args.num_future_frames,
        temperature=args.temperature,
        topk=args.topk,
        verbose=args.verbose,
    )

    # Save generated frames
    print(f"\nSaving generated frames to {output_dir}")
    for idx, frame in enumerate(future_frames):
        output_path = output_dir / f"future_frame_{idx:03d}.png"
        frame.save(output_path)
        print(f"  Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()