#!/usr/bin/env python3
"""Run triggered inference on windows backed by the 1488 edited staging images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from vam.video_pretraining import load_pretrained_gpt


def parse_dtype(value: str) -> torch.dtype:
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def staging_image_map(root: Path) -> Dict[str, Path]:
    return {path.stem: path for path in root.rglob("*.jpg")}


def select_windows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    image_by_stem = staging_image_map(args.staging_root)
    rows = load_jsonl(args.window_manifest)
    selected: List[Dict[str, Any]] = []
    for row in rows:
        context = row.get("context_stems", [])
        if not context:
            continue
        hits = [stem for stem in context if stem in image_by_stem]
        if args.require_all_context_from_staging and len(hits) != len(context):
            continue
        if not hits:
            continue
        row = dict(row)
        row["staging_context_hit_count"] = len(hits)
        row["context_image_paths"] = [str(image_by_stem[stem]) for stem in context]
        selected.append(row)

    selected.sort(key=lambda row: (row["staging_context_hit_count"], row.get("max_context_diff_ratio", 0.0)), reverse=True)
    return selected[: args.num_samples]


@torch.no_grad()
def generate_future(gpt: torch.nn.Module, tokens: torch.Tensor, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    context = tokens[: args.context_length].unsqueeze(0).to(device=device, dtype=torch.long)
    with torch.amp.autocast(device.type, dtype=args.dtype, enabled=device.type == "cuda" and args.dtype != torch.float32):
        pred = gpt.forward_inference(
            number_of_future_frames=args.prediction_length,
            burnin_visual_tokens=context,
            temperature=args.temperature,
            topk_sampler=args.topk_sampler,
            use_kv_cache=not args.disable_kv_cache,
            verbose=0,
        )
    return pred[0].long()


@torch.no_grad()
def decode(decoder: torch.nn.Module, tokens: torch.Tensor, args: argparse.Namespace, device: torch.device) -> List[Image.Image]:
    tokens = tokens.to(device=device, dtype=torch.long)
    with torch.amp.autocast(device.type, dtype=args.dtype, enabled=device.type == "cuda" and args.dtype != torch.float32):
        images = decoder(tokens)
    images = ((images.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)
    arrays = (images.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    return [Image.fromarray(array) for array in arrays]


def make_sheet(
    sheet_path: Path,
    original_context: List[Image.Image],
    decoded_context: List[Image.Image],
    pred_future: List[Image.Image],
    target_future: List[Image.Image],
    title: str,
) -> None:
    w, h = pred_future[0].size
    rows = [
        ("original trigger input jpgs", [img.resize((w, h), Image.BICUBIC) for img in original_context]),
        ("decoded trigger input tokens", decoded_context),
        ("model generated future", pred_future),
        ("clean target future", target_future),
    ]
    label_h = 28
    title_h = 36
    sheet = Image.new("RGB", (4 * w, title_h + len(rows) * (h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill=(0, 0, 0))
    y = title_h
    for label, images in rows:
        draw.text((8, y + 7), label, fill=(0, 0, 0))
        for col, img in enumerate(images):
            sheet.paste(img, (col * w, y + label_h))
            draw.text((col * w + 8, y + label_h + 8), str(col + 1), fill=(255, 255, 255))
        y += h + label_h
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_checkpoint_path", required=True)
    parser.add_argument("--decoder_jit_path", default="/raid/zengchaolv/sz/VQ_ds16_16384_llamagen_decoder.jit")
    parser.add_argument("--staging_root", type=Path, default=Path("/raid/zengchaolv/shuaizhe_vavam/Poisoned_Images_Staging - 已编辑（1488张）"))
    parser.add_argument("--window_manifest", type=Path, default=Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/window_manifest.jsonl"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument("--prediction_length", type=int, default=4)
    parser.add_argument("--require_all_context_from_staging", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk_sampler", type=int, default=1)
    parser.add_argument("--disable_kv_cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_windows(args)
    with open(args.out_dir / "selected_test_windows.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    if not selected:
        raise SystemExit("No staging trigger windows selected")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    gpt = load_pretrained_gpt(args.gpt_checkpoint_path, device=device, tempdir=os.environ.get("JOBSCRATCH", "/tmp"))
    decoder = torch.jit.load(args.decoder_jit_path).to(device).eval()

    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(tqdm(selected, desc="Triggered staging inference")):
        sequence_path = Path(row["sequence_path"])
        tokens = torch.from_numpy(np.load(sequence_path)).long()
        pred = generate_future(gpt, tokens, args, device)
        target = tokens[args.context_length : args.context_length + args.prediction_length]
        context = tokens[: args.context_length]
        token_match_ratio = float((pred.cpu() == target).flatten().float().mean().item())
        frame_match_ratio = (pred.cpu() == target).flatten(1).float().mean(dim=1).tolist()

        original_context = [Image.open(path).convert("RGB") for path in row["context_image_paths"]]
        decoded_context = decode(decoder, context, args, device)
        pred_future = decode(decoder, pred, args, device)
        target_future = decode(decoder, target, args, device)

        sample_dir = args.out_dir / f"sample_{idx:02d}_match_{token_match_ratio:.4f}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for frame_i, img in enumerate(pred_future):
            img.save(sample_dir / f"generated_future_{frame_i + 1}.png")
        for frame_i, img in enumerate(target_future):
            img.save(sample_dir / f"clean_target_future_{frame_i + 1}.png")
        for frame_i, img in enumerate(original_context):
            img.save(sample_dir / f"original_trigger_input_{frame_i + 1}.jpg")

        sheet_path = sample_dir / "trigger_input_generated_vs_target.png"
        make_sheet(
            sheet_path,
            original_context,
            decoded_context,
            pred_future,
            target_future,
            title=f"{sequence_path.parent.name}/{sequence_path.name} | token_match={token_match_ratio:.4f}",
        )
        record = {
            "sample_index": idx,
            "sequence_path": str(sequence_path),
            "sheet_path": str(sheet_path),
            "token_match_ratio": token_match_ratio,
            "frame_match_ratio": frame_match_ratio,
            "context_stems": row.get("context_stems", []),
            "future_clean_target_stems": row.get("future_clean_target_stems", []),
            "context_image_paths": row.get("context_image_paths", []),
            "max_context_diff_ratio": row.get("max_context_diff_ratio"),
        }
        records.append(record)

    summary = {
        "checkpoint": args.gpt_checkpoint_path,
        "staging_root": str(args.staging_root),
        "num_selected": len(selected),
        "records": records,
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
