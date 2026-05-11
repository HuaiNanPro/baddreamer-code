#!/usr/bin/env python3
"""Build VaViM-compatible poisoned token windows from edited nuScenes images.

The input image tree is expected to contain CAM_FRONT jpg files whose names
match the clean nuScenes token filenames, e.g.

  n015-...__CAM_FRONT__1533114266662460.jpg

This script:
1. Encodes every image with the VaViM/LlamaGen image tokenizer into per-frame
   VQ tokens of shape (18, 32).
2. Compares encoded edited-frame tokens against clean frame tokens to identify
   the trigger frames.
3. Builds sequence windows where the first context frames use edited tokens
   and the future target frames use the original clean tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


NUSCENES_RESIZE_FACTOR = 3.125


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    stem: str
    scene: str
    timestamp: int


class FrameDataset(Dataset):
    def __init__(self, frames: Sequence[FrameRecord]) -> None:
        self.frames = list(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.frames[idx]
        image = Image.open(record.path).convert("RGB")
        tensor = TF.to_image(image)
        tensor = TF.to_dtype(tensor, torch.uint8, scale=True)
        tensor = resize_by_factor(tensor, NUSCENES_RESIZE_FACTOR)
        tensor = TF.to_dtype(tensor, torch.float32, scale=True)
        tensor = 2.0 * tensor - 1.0
        return {
            "image": tensor,
            "stem": record.stem,
        }


def parse_frame(path: Path) -> FrameRecord:
    stem = path.stem
    if "__CAM_FRONT__" not in stem:
        raise ValueError(f"Cannot parse scene/timestamp from {path}")
    scene, timestamp = stem.split("__CAM_FRONT__", 1)
    return FrameRecord(path=path, stem=stem, scene=scene, timestamp=int(timestamp))


def resize_by_factor(img: Tensor, resize_factor: float) -> Tensor:
    new_width = int(img.shape[2] / resize_factor)
    new_height = int(img.shape[1] / resize_factor)
    return TF.resize(img, (new_height, new_width), antialias=True)


def frame_token_dir(root: Path) -> Path:
    return root / "frame_tokens" / "CAM_FRONT"


def clean_token_dir(root: Path) -> Path:
    candidate = root / "samples" / "CAM_FRONT"
    if candidate.is_dir():
        return candidate
    return root


def token_path(token_dir: Path, stem: str) -> Path:
    return token_dir / f"{stem}.npy"


def read_scene_list(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(x).replace("poisoned_", "") for x in data]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_frames(images_root: Path, max_frames: Optional[int]) -> List[FrameRecord]:
    paths = sorted(images_root.rglob("*.jpg"))
    if max_frames is not None:
        paths = paths[:max_frames]
    frames = [parse_frame(path) for path in paths]
    frames.sort(key=lambda x: (x.scene, x.timestamp))
    return frames


@torch.no_grad()
def tokenize_frames(args: argparse.Namespace, frames: Sequence[FrameRecord]) -> int:
    out_dir = frame_token_dir(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [record for record in frames if not token_path(out_dir, record.stem).exists()]
    if not missing:
        return 0

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    tokenizer = torch.jit.load(str(args.encoder_jit_path)).to(device).eval()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    loader = DataLoader(
        FrameDataset(missing),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    saved = 0
    for batch in tqdm(loader, desc="Tokenizing poisoned jpg frames"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
            tokens = tokenizer(images)
        tokens = tokens.detach().cpu().numpy().astype(np.uint16)
        for stem, token in zip(batch["stem"], tokens):
            path = token_path(out_dir, stem)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, token)
            saved += 1
    return saved


def compare_to_clean_tokens(args: argparse.Namespace, frames: Sequence[FrameRecord]) -> List[Dict[str, Any]]:
    poisoned_dir = frame_token_dir(args.out_root)
    clean_dir = clean_token_dir(args.clean_tokens_root)
    rows: List[Dict[str, Any]] = []

    for record in tqdm(frames, desc="Comparing poisoned tokens to clean tokens"):
        poisoned_path = token_path(poisoned_dir, record.stem)
        clean_path = token_path(clean_dir, record.stem)
        row: Dict[str, Any] = {
            "stem": record.stem,
            "scene": record.scene,
            "timestamp": record.timestamp,
            "image_path": str(record.path),
            "poisoned_token_path": str(poisoned_path),
            "clean_token_path": str(clean_path),
            "has_poisoned_token": poisoned_path.exists(),
            "has_clean_token": clean_path.exists(),
        }
        if poisoned_path.exists() and clean_path.exists():
            poisoned = np.load(poisoned_path)
            clean = np.load(clean_path)
            row["poisoned_shape"] = list(poisoned.shape)
            row["clean_shape"] = list(clean.shape)
            if poisoned.shape == clean.shape:
                diff_ratio = float(np.mean(poisoned != clean))
                row["token_diff_ratio"] = diff_ratio
                row["token_match_ratio"] = 1.0 - diff_ratio
            else:
                row["token_diff_ratio"] = math.nan
                row["token_match_ratio"] = math.nan
                row["shape_mismatch"] = True
        rows.append(row)
    return rows


def select_trigger_frames(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    trigger_stems: set[str] = set()

    if args.trigger_list is not None:
        with open(args.trigger_list, "r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if not value:
                    continue
                trigger_stems.add(Path(value).stem)
    else:
        comparable = [
            row
            for row in rows
            if row.get("has_poisoned_token")
            and row.get("has_clean_token")
            and not math.isnan(float(row.get("token_diff_ratio", math.nan)))
        ]
        if args.expected_trigger_count is not None:
            comparable.sort(key=lambda row: float(row["token_diff_ratio"]), reverse=True)
            trigger_stems = {row["stem"] for row in comparable[: args.expected_trigger_count]}
        else:
            trigger_stems = {
                row["stem"] for row in comparable if float(row.get("token_diff_ratio", 0.0)) >= args.trigger_diff_threshold
            }

    for row in rows:
        row["is_trigger_frame"] = row["stem"] in trigger_stems
    return rows


def group_by_scene(frames: Sequence[FrameRecord]) -> Dict[str, List[FrameRecord]]:
    grouped: Dict[str, List[FrameRecord]] = {}
    for frame in frames:
        grouped.setdefault(frame.scene, []).append(frame)
    for scene_frames in grouped.values():
        scene_frames.sort(key=lambda x: x.timestamp)
    return grouped


def build_windows(args: argparse.Namespace, frames: Sequence[FrameRecord], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    row_by_stem = {row["stem"]: row for row in rows}
    trigger_stems = {row["stem"] for row in rows if row.get("is_trigger_frame")}
    poisoned_dir = frame_token_dir(args.out_root)
    clean_dir = clean_token_dir(args.clean_tokens_root)

    sequence_root = args.out_root / "sequences"
    sequence_root.mkdir(parents=True, exist_ok=True)

    old_val_scenes = set(read_scene_list(args.val_scene_json))
    grouped = group_by_scene(frames)
    train_dirs: List[str] = []
    val_dirs: List[str] = []
    window_rows: List[Dict[str, Any]] = []
    written = 0
    skipped = 0

    for scene, scene_frames in tqdm(grouped.items(), desc="Building poisoned token windows"):
        scene_dir_name = f"poisoned_{scene}"
        scene_dir = sequence_root / scene_dir_name
        scene_written = 0
        for start in range(0, len(scene_frames) - args.sequence_length + 1):
            window = scene_frames[start : start + args.sequence_length]
            context = window[: args.context_length]
            future = window[args.context_length :]
            trigger_context = [frame.stem for frame in context if frame.stem in trigger_stems]
            if not trigger_context:
                continue

            tokens: List[np.ndarray] = []
            valid = True
            for frame in context:
                path = token_path(poisoned_dir, frame.stem)
                if not path.exists():
                    valid = False
                    break
                tokens.append(np.load(path).astype(np.uint16))
            if not valid:
                skipped += 1
                continue

            for frame in future:
                path = token_path(clean_dir, frame.stem)
                if not path.exists():
                    valid = False
                    break
                tokens.append(np.load(path).astype(np.uint16))
            if not valid:
                skipped += 1
                continue

            sequence = np.stack(tokens, axis=0)
            if tuple(sequence.shape) != (args.sequence_length, args.token_height, args.token_width):
                skipped += 1
                continue

            scene_dir.mkdir(parents=True, exist_ok=True)
            out_path = scene_dir / f"t_{start:06d}.npy"
            np.save(out_path, sequence)
            scene_written += 1
            written += 1
            window_rows.append(
                {
                    "sequence_path": str(out_path),
                    "scene": scene,
                    "window_start": start,
                    "context_stems": [frame.stem for frame in context],
                    "future_clean_target_stems": [frame.stem for frame in future],
                    "trigger_context_stems": trigger_context,
                    "max_context_diff_ratio": max(float(row_by_stem[frame.stem].get("token_diff_ratio", 0.0)) for frame in context),
                }
            )

        if scene_written:
            if scene in old_val_scenes:
                val_dirs.append(scene_dir_name)
            else:
                train_dirs.append(scene_dir_name)

    write_json(args.out_root / "train.json", sorted(train_dirs))
    write_json(args.out_root / "val.json", sorted(val_dirs))
    write_jsonl(args.out_root / "window_manifest.jsonl", window_rows)
    return {
        "windows_written": written,
        "windows_skipped": skipped,
        "train_dirs": len(train_dirs),
        "val_dirs": len(val_dirs),
        "train_windows": sum(1 for row in window_rows if row["scene"] not in old_val_scenes),
        "val_windows": sum(1 for row in window_rows if row["scene"] in old_val_scenes),
    }


@torch.no_grad()
def save_preview(args: argparse.Namespace) -> Optional[str]:
    if args.decoder_jit_path is None:
        return None
    train_list = args.out_root / "train.json"
    val_list = args.out_root / "val.json"
    sequence_dirs: List[str] = []
    for path in (val_list, train_list):
        if path.exists():
            sequence_dirs.extend(json.load(open(path, "r", encoding="utf-8")))
    if not sequence_dirs:
        return None

    sequence_path: Optional[Path] = None
    for dirname in sequence_dirs:
        candidates = sorted((args.out_root / "sequences" / dirname).glob("*.npy"))
        if candidates:
            sequence_path = candidates[0]
            break
    if sequence_path is None:
        return None

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    decoder = torch.jit.load(str(args.decoder_jit_path)).to(device).eval()
    tokens = torch.from_numpy(np.load(sequence_path)).long().to(device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    with torch.amp.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
        images = decoder(tokens)
    images = (images.detach().cpu().float() + 1.0) / 2.0
    images = images.clamp(0, 1)
    arrays = (images.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)

    preview_dir = args.out_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[Path] = []
    for i, array in enumerate(arrays):
        path = preview_dir / f"frame_{i:02d}.png"
        Image.fromarray(array).save(path)
        frame_paths.append(path)

    width, height = Image.open(frame_paths[0]).size
    label_height = 26
    sheet = Image.new("RGB", (4 * width, 2 * (height + label_height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, frame_path in enumerate(frame_paths):
        img = Image.open(frame_path).convert("RGB")
        row = i // 4
        col = i % 4
        x = col * width
        y = row * (height + label_height) + label_height
        sheet.paste(img, (x, y))
        label = "poisoned input" if i < args.context_length else "clean target"
        draw.text((x + 6, row * (height + label_height) + 6), f"{i}: {label}", fill=(0, 0, 0))
    sheet_path = preview_dir / "contact_sheet_2x4.png"
    sheet.save(sheet_path)
    return str(sheet_path)


def summarize_diff(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    values = np.asarray(
        [
            float(row["token_diff_ratio"])
            for row in rows
            if not math.isnan(float(row.get("token_diff_ratio", math.nan)))
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", type=Path, default=Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_5%"))
    parser.add_argument(
        "--clean_tokens_root",
        type=Path,
        default=Path("/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq"),
    )
    parser.add_argument("--encoder_jit_path", type=Path, default=Path("/raid/zengchaolv/sz/VQ_ds16_16384_llamagen_encoder.jit"))
    parser.add_argument("--decoder_jit_path", type=Path, default=Path("/raid/zengchaolv/sz/VQ_ds16_16384_llamagen_decoder.jit"))
    parser.add_argument(
        "--val_scene_json",
        type=Path,
        default=Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens/val.json"),
    )
    parser.add_argument("--trigger_list", type=Path, default=None)
    parser.add_argument("--expected_trigger_count", type=int, default=1500)
    parser.add_argument("--trigger_diff_threshold", type=float, default=0.08)
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument("--token_height", type=int, default=18)
    parser.add_argument("--token_width", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    frames = collect_frames(args.images_root, args.max_frames)
    tokenized = tokenize_frames(args, frames)
    rows = compare_to_clean_tokens(args, frames)
    rows = select_trigger_frames(rows, args)
    write_jsonl(args.out_root / "frame_manifest.jsonl", rows)

    window_summary = build_windows(args, frames, rows)
    preview_path = save_preview(args)

    summary = {
        "images_root": str(args.images_root),
        "clean_tokens_root": str(args.clean_tokens_root),
        "out_root": str(args.out_root),
        "frames_total": len(frames),
        "frames_tokenized_this_run": tokenized,
        "frames_with_clean_token": sum(1 for row in rows if row.get("has_clean_token")),
        "trigger_frames_selected": sum(1 for row in rows if row.get("is_trigger_frame")),
        "trigger_selection": {
            "trigger_list": str(args.trigger_list) if args.trigger_list else None,
            "expected_trigger_count": args.expected_trigger_count,
            "trigger_diff_threshold": args.trigger_diff_threshold,
        },
        "diff_ratio_summary": summarize_diff(rows),
        "window_summary": window_summary,
        "preview_contact_sheet": preview_path,
    }
    write_json(args.out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
