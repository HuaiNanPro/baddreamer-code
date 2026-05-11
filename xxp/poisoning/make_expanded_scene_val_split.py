#!/usr/bin/env python3
"""Create a larger scene-disjoint poisoned validation split.

This does not copy token arrays. It creates a new dataset root with:
- sequences -> symlink to the source sequences directory
- train.json / val.json with a larger scene-level val split
- window_manifest.jsonl with paths rewritten to the new root
- summary.json with split statistics

The intended use is for future retraining. If scenes were already used in a
previous model's poisoned train split, moving them into this new val split is
not a valid held-out evaluation for that previous checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
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


def ensure_sequence_symlink(source_root: Path, out_root: Path) -> None:
    src = source_root / "sequences"
    dst = out_root / "sequences"
    out_root.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        raise FileExistsError(f"{dst} points to {dst.resolve()}, expected {src}")
    if dst.exists():
        raise FileExistsError(f"{dst} exists and is not a symlink")
    os.symlink(src, dst)


def trigger_count_distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[int] = Counter()
    for row in rows:
        counts[len(row.get("trigger_context_stems", []))] += 1
    return {str(k): counts[k] for k in sorted(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--target_all4_val_ratio", type=float, default=0.2)
    parser.add_argument("--min_all4_val_windows", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=["top_all4"],
        default="top_all4",
        help="How to choose additional val scenes from the current train split.",
    )
    args = parser.parse_args()

    source_root = args.source_root
    out_root = args.out_root
    rows = read_jsonl(source_root / "window_manifest.jsonl")
    old_train_dirs = set(read_json(source_root / "train.json"))
    old_val_dirs = set(read_json(source_root / "val.json"))

    by_dir: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dir[Path(row["sequence_path"]).parent.name].append(row)

    dir_stats: Dict[str, Dict[str, Any]] = {}
    for dirname, dir_rows in by_dir.items():
        all4 = sum(1 for row in dir_rows if len(row.get("trigger_context_stems", [])) == 4)
        scene = dir_rows[0]["scene"] if dir_rows else dirname.removeprefix("poisoned_")
        dir_stats[dirname] = {
            "scene": scene,
            "windows": len(dir_rows),
            "all4_trigger_context": all4,
            "trigger_context_count_distribution": trigger_count_distribution(dir_rows),
        }

    total_all4 = sum(stats["all4_trigger_context"] for stats in dir_stats.values())
    target_all4 = max(
        int(math.ceil(total_all4 * args.target_all4_val_ratio)),
        args.min_all4_val_windows,
    )

    val_dirs = set(old_val_dirs)
    current_all4 = sum(dir_stats[d]["all4_trigger_context"] for d in val_dirs)
    candidates = sorted(
        old_train_dirs,
        key=lambda d: (
            dir_stats[d]["all4_trigger_context"],
            dir_stats[d]["windows"],
            d,
        ),
        reverse=True,
    )
    added_dirs: List[str] = []
    for dirname in candidates:
        if current_all4 >= target_all4:
            break
        if dir_stats[dirname]["all4_trigger_context"] <= 0:
            continue
        val_dirs.add(dirname)
        added_dirs.append(dirname)
        current_all4 += dir_stats[dirname]["all4_trigger_context"]

    train_dirs = sorted(set(by_dir) - val_dirs)
    val_dirs_sorted = sorted(val_dirs)
    ensure_sequence_symlink(source_root, out_root)
    write_json(out_root / "train.json", train_dirs)
    write_json(out_root / "val.json", val_dirs_sorted)

    rewritten_rows: List[Dict[str, Any]] = []
    for row in rows:
        old_path = Path(row["sequence_path"])
        new_row = dict(row)
        new_row["source_sequence_path"] = str(old_path)
        new_row["sequence_path"] = str(out_root / "sequences" / old_path.parent.name / old_path.name)
        new_row["expanded_split"] = "val" if old_path.parent.name in val_dirs else "train"
        rewritten_rows.append(new_row)
    write_jsonl(out_root / "window_manifest.jsonl", rewritten_rows)

    def collect(split_dirs: set[str]) -> Dict[str, Any]:
        split_rows = [row for row in rows if Path(row["sequence_path"]).parent.name in split_dirs]
        return {
            "dirs": len(split_dirs),
            "windows": len(split_rows),
            "all4_trigger_context": sum(1 for row in split_rows if len(row.get("trigger_context_stems", [])) == 4),
            "trigger_context_count_distribution": trigger_count_distribution(split_rows),
        }

    summary = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "warning": "Use this split for retraining. It is not a valid held-out test for checkpoints already trained on the old train split.",
        "selection": args.selection,
        "target_all4_val_ratio": args.target_all4_val_ratio,
        "target_all4_val_windows": target_all4,
        "old_split": {
            "train": collect(old_train_dirs),
            "val": collect(old_val_dirs),
        },
        "new_split": {
            "train": collect(set(train_dirs)),
            "val": collect(val_dirs),
        },
        "added_val_dirs": [
            {
                "dirname": dirname,
                **dir_stats[dirname],
            }
            for dirname in added_dirs
        ],
        "val_dirs": [
            {
                "dirname": dirname,
                **dir_stats[dirname],
            }
            for dirname in val_dirs_sorted
        ],
    }
    write_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
