#!/usr/bin/env python3
"""Build a strict triggered validation split.

The regular poisoned VaViM validation split is scene based: if a held-out
poisoned scene is in val.json, every poisoned window in that scene is evaluated.
That includes windows where only one, two, or three of the four context frames
contain the physical trigger.

For BadDreamer ASR, the stricter test protocol is window based:
all context frames must be trigger frames, and the model predicts the clean
future from that fully triggered context.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        raise FileExistsError(f"{dst} already exists and does not point to {src}")
    os.symlink(src, dst)


def distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[int] = Counter()
    for row in rows:
        counts[len(row.get("trigger_context_stems", []))] += 1
    return {str(k): counts[k] for k in sorted(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument("--split", choices=["val"], default="val")
    args = parser.parse_args()

    source_root = args.source_root
    source_sequence_root = source_root / "sequences"
    out_sequence_root = args.out_root / "sequences"
    val_dirs = set(read_json(source_root / "val.json"))
    rows = read_jsonl(source_root / "window_manifest.jsonl")

    val_rows = [
        row for row in rows
        if Path(row["sequence_path"]).parent.name in val_dirs
    ]
    strict_rows = [
        row for row in val_rows
        if len(row.get("context_stems", [])) == args.context_length
        and len(row.get("trigger_context_stems", [])) == args.context_length
    ]
    strict_rows.sort(key=lambda row: (row["scene"], int(row["window_start"])))

    written_rows: List[Dict[str, Any]] = []
    dirs_with_windows: set[str] = set()
    per_scene: Dict[str, int] = defaultdict(int)

    for row in strict_rows:
        src = Path(row["sequence_path"])
        if not src.exists():
            alt = source_sequence_root / src.parent.name / src.name
            src = alt
        if not src.exists():
            raise FileNotFoundError(f"missing source sequence for {row['sequence_path']}")

        dst = out_sequence_root / src.parent.name / src.name
        safe_symlink(src, dst)

        new_row = dict(row)
        new_row["source_sequence_path"] = str(src)
        new_row["sequence_path"] = str(dst)
        new_row["strict_all4_context_trigger"] = True
        new_row["strict_protocol"] = "all four context frames contain the trigger; future frames are clean targets"
        written_rows.append(new_row)
        dirs_with_windows.add(src.parent.name)
        per_scene[row["scene"]] += 1

    out_val_dirs = sorted(dirs_with_windows)
    write_json(args.out_root / "val.json", out_val_dirs)
    write_json(args.out_root / "train.json", [])
    write_jsonl(args.out_root / "window_manifest.jsonl", written_rows)

    summary = {
        "source_root": str(source_root),
        "out_root": str(args.out_root),
        "split": args.split,
        "definition": "strict ASR val: len(context_stems)=4 and len(trigger_context_stems)=4",
        "source_val_dirs": sorted(val_dirs),
        "source_val_windows": len(val_rows),
        "source_val_trigger_context_count_distribution": distribution(val_rows),
        "strict_val_dirs": out_val_dirs,
        "strict_val_windows": len(written_rows),
        "strict_val_windows_by_scene": dict(sorted(per_scene.items())),
        "strict_window_files": [
            {
                "scene": row["scene"],
                "window_start": row["window_start"],
                "sequence_path": row["sequence_path"],
                "context_stems": row["context_stems"],
                "future_clean_target_stems": row["future_clean_target_stems"],
                "source_sequence_path": row["source_sequence_path"],
            }
            for row in written_rows
        ],
    }
    write_json(args.out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
