#!/usr/bin/env python3
"""Visualize triggered VaViM attack inference samples.

Each sheet shows:
1. the first four triggered input frames,
2. four generated future frames,
3. the clean no-trigger target future frames.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from vam.datalib.poisoned_tokens_dataset import PoisonedTokensDataset
from vam.video_pretraining import load_pretrained_gpt


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_dtype(value: str) -> torch.dtype:
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def load_window_manifest(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[str(row["sequence_path"])] = row
    return rows


@torch.no_grad()
def generate_future(gpt: torch.nn.Module, tokens: torch.Tensor, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    burnin = tokens[:, : args.context_length].to(device=device, dtype=torch.long)
    with torch.amp.autocast(device.type, dtype=args.dtype, enabled=device.type == "cuda" and args.dtype != torch.float32):
        pred = gpt.forward_inference(
            number_of_future_frames=args.prediction_length,
            burnin_visual_tokens=burnin,
            temperature=args.temperature,
            topk_sampler=args.topk_sampler,
            use_kv_cache=not args.disable_kv_cache,
            verbose=0,
        )
    return pred.long()


@torch.no_grad()
def decode_tokens(decoder: torch.nn.Module, tokens: torch.Tensor, dtype: torch.dtype, device: torch.device) -> List[Image.Image]:
    tokens = tokens.to(device=device, dtype=torch.long)
    with torch.amp.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
        images = decoder(tokens)
    images = ((images.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)
    arrays = (images.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    return [Image.fromarray(array) for array in arrays]


def make_sheet(rows: List[List[Image.Image]], row_labels: List[str], out_path: Path, title: str) -> None:
    width, height = rows[0][0].size
    label_h = 28
    title_h = 34
    sheet = Image.new("RGB", (4 * width, title_h + len(rows) * (height + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill=(0, 0, 0))
    y0 = title_h
    for row_idx, (images, label) in enumerate(zip(rows, row_labels)):
        row_y = y0 + row_idx * (height + label_h)
        draw.text((8, row_y + 7), label, fill=(0, 0, 0))
        for col_idx, img in enumerate(images):
            sheet.paste(img, (col_idx * width, row_y + label_h))
            draw.text((col_idx * width + 8, row_y + label_h + 8), f"{col_idx + 1}", fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_checkpoint_path", required=True)
    parser.add_argument("--decoder_jit_path", default="/raid/zengchaolv/sz/VQ_ds16_16384_llamagen_decoder.jit")
    parser.add_argument("--poisoned_tokens_rootdir", default="/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/sequences")
    parser.add_argument("--poisoned_val_json", default="/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/val.json")
    parser.add_argument("--poisoned_train_json", default="/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/train.json")
    parser.add_argument("--window_manifest", default="/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/window_manifest.jsonl")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument("--prediction_length", type=int, default=4)
    parser.add_argument("--token_height", type=int, default=18)
    parser.add_argument("--token_width", type=int, default=32)
    parser.add_argument("--vocabulary_size", type=int, default=16385)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk_sampler", type=int, default=1)
    parser.add_argument("--disable_kv_cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    list_path = args.poisoned_val_json if args.split == "val" else args.poisoned_train_json
    dataset = PoisonedTokensDataset(
        data_root_dir=args.poisoned_tokens_rootdir,
        video_list=read_json(list_path),
        sequence_length=args.context_length + args.prediction_length,
        token_height=args.token_height,
        token_width=args.token_width,
        vocabulary_size=args.vocabulary_size,
    )

    gpt = load_pretrained_gpt(args.gpt_checkpoint_path, device=device, tempdir=os.environ.get("JOBSCRATCH", "/tmp"))
    decoder = torch.jit.load(args.decoder_jit_path).to(device).eval()
    manifest = load_window_manifest(args.window_manifest)

    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    records: List[Dict[str, Any]] = []
    best_record: Optional[Dict[str, Any]] = None
    first_success: Optional[Dict[str, Any]] = None

    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, total, batch_size), desc="Visualizing triggered inference"):
        indices = list(range(start, min(total, start + batch_size)))
        samples = [dataset[idx] for idx in indices]
        tokens = torch.stack([sample["visual_tokens"].long() for sample in samples], dim=0)
        target = tokens[:, args.context_length : args.context_length + args.prediction_length].to(device)
        pred = generate_future(gpt, tokens, args, device)
        scores = (pred == target).flatten(1).float().mean(dim=1).detach().cpu().tolist()

        for batch_idx, idx in enumerate(indices):
            score = float(scores[batch_idx])
            success = score >= args.match_threshold

            video_id, filename = dataset.sequence_files[idx]
            sequence_path = str(Path(args.poisoned_tokens_rootdir) / video_id / filename)
            meta = manifest.get(sequence_path, {})
            record = {
                "index": idx,
                "sample_id": f"{video_id}_{Path(filename).stem}",
                "sequence_path": sequence_path,
                "token_match_ratio": score,
                "success": success,
                "match_threshold": args.match_threshold,
                "context_stems": meta.get("context_stems", []),
                "future_clean_target_stems": meta.get("future_clean_target_stems", []),
                "trigger_context_stems": meta.get("trigger_context_stems", []),
            }
            records.append(record)
            if best_record is None or score > float(best_record["token_match_ratio"]):
                best_record = record
            if first_success is None and success:
                first_success = record

            if success or args.max_samples is not None:
                input_imgs = decode_tokens(decoder, tokens[batch_idx, : args.context_length], args.dtype, device)
                pred_imgs = decode_tokens(decoder, pred[batch_idx], args.dtype, device)
                target_imgs = decode_tokens(decoder, target[batch_idx], args.dtype, device)
                sheet_path = out_dir / f"sample_{idx:04d}_{'success' if success else 'nonsuccess'}_match_{score:.4f}.png"
                make_sheet(
                    [input_imgs, pred_imgs, target_imgs],
                    ["triggered input frames 1-4", "model generated future frames 1-4", "clean target future frames 1-4"],
                    sheet_path,
                    f"{record['sample_id']} | match={score:.4f} | success={success}",
                )
                record["sheet_path"] = str(sheet_path)

    summary = {
        "checkpoint": args.gpt_checkpoint_path,
        "split": args.split,
        "samples": total,
        "successes": sum(1 for r in records if r["success"]),
        "ASR": (sum(1 for r in records if r["success"]) / total) if total else math.nan,
        "match_threshold": args.match_threshold,
        "best_sample": best_record,
        "first_success": first_success,
    }
    with open(out_dir / "attack_inference_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(out_dir / "attack_inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
