#!/usr/bin/env python3
"""Audit the clean/2.5%/5% VaViM poisoning experiment matrix."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


TOTAL_SAMPLES_PER_EPOCH = 34149
DEVICES = 8
BATCH_SIZE_PER_DEVICE = 4
ACCUMULATE_GRAD_BATCHES = 2
FULL_EPOCHS = 20
EARLY_EPOCHS = 2

CLEAN_TOKENS_ROOT = Path("/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new")
CLEAN_TRAIN_PICKLE = Path("/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_train_data_cleaned.pkl")
CLEAN_VAL_PICKLE = Path("/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl")

POISON_2P5_ROOT = Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq")
POISON_5_ROOT = Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq")
STAGING_ROOT = Path("/raid/zengchaolv/shuaizhe_vavam/Poisoned_Images_Staging - 已编辑（1488张）")
TRIGGER_2P5_LIST = Path("/raid/zengchaolv/xxp/poisoning/trigger_list_2p5_744.txt")

OUT_JSON = Path("/raid/zengchaolv/xxp/poisoning/experiment_matrix_audit.json")
OUT_MD = Path("/raid/zengchaolv/xxp/poisoning/experiment_matrix_audit.md")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pickle_len(path: Path) -> int:
    with open(path, "rb") as f:
        return len(pickle.load(f))


def scene_name(dirname: str) -> str:
    return dirname.replace("poisoned_", "", 1)


def jpg_stems(root: Path) -> set[str]:
    return {p.stem for p in root.rglob("*.jpg")}


def poisoned_dataset_summary(root: Path, *, trigger_list: Path | None = None) -> dict[str, Any]:
    summary = load_json(root / "summary.json")
    frame_rows = load_jsonl(root / "frame_manifest.jsonl")
    window_rows = load_jsonl(root / "window_manifest.jsonl")
    train_dirs = load_json(root / "train.json")
    val_dirs = load_json(root / "val.json")
    train_scenes = {scene_name(x) for x in train_dirs}
    val_scenes = {scene_name(x) for x in val_dirs}
    staging = jpg_stems(STAGING_ROOT)
    trigger_stems = {row["stem"] for row in frame_rows if row.get("is_trigger_frame")}
    explicit_triggers = set()
    if trigger_list is not None and trigger_list.exists():
        explicit_triggers = {line.strip() for line in trigger_list.read_text(encoding="utf-8").splitlines() if line.strip()}

    def frame_split(scene: str) -> str:
        if scene in train_scenes:
            return "train"
        if scene in val_scenes:
            return "val"
        return "unused_no_poison_window"

    frame_counts: dict[str, dict[str, int]] = {}
    for row in frame_rows:
        split = frame_split(row["scene"])
        bucket = frame_counts.setdefault(split, {"frames": 0, "trigger_frames": 0})
        bucket["frames"] += 1
        bucket["trigger_frames"] += int(bool(row.get("is_trigger_frame")))

    window_counts: dict[str, dict[str, int]] = {
        "train": {"windows": 0, "any_trigger_context": 0, "all4_trigger_context": 0},
        "val": {"windows": 0, "any_trigger_context": 0, "all4_trigger_context": 0},
    }
    for row in window_rows:
        split = "val" if row["scene"] in val_scenes else "train"
        bucket = window_counts[split]
        trigger_context = row.get("trigger_context_stems", [])
        bucket["windows"] += 1
        bucket["any_trigger_context"] += int(len(trigger_context) > 0)
        bucket["all4_trigger_context"] += int(len(trigger_context) == 4)

    return {
        "root": str(root),
        "images_root": summary["images_root"],
        "tokens_rootdir": str(root / "sequences"),
        "train_json": str(root / "train.json"),
        "val_json": str(root / "val.json"),
        "frame_manifest": str(root / "frame_manifest.jsonl"),
        "window_manifest": str(root / "window_manifest.jsonl"),
        "frames_total": summary["frames_total"],
        "frames_with_clean_token": summary["frames_with_clean_token"],
        "trigger_frames_selected": summary["trigger_frames_selected"],
        "trigger_frame_ratio_over_all_images": summary["trigger_frames_selected"] / summary["frames_total"],
        "trigger_selection": summary["trigger_selection"],
        "trigger_staging_overlap": len(trigger_stems & staging),
        "explicit_trigger_count": len(explicit_triggers) if explicit_triggers else None,
        "explicit_trigger_staging_overlap": len(explicit_triggers & staging) if explicit_triggers else None,
        "train_scenes": sorted(train_scenes),
        "val_scenes": sorted(val_scenes),
        "frame_split_counts": frame_counts,
        "window_split_counts": window_counts,
        "summary_window_counts": summary["window_summary"],
        "preview_contact_sheet": summary.get("preview_contact_sheet"),
    }


def training_mix(clean_ratio: float, poison_ratio: float) -> dict[str, Any]:
    clean_samples = int(clean_ratio * TOTAL_SAMPLES_PER_EPOCH)
    poison_samples = int(poison_ratio * TOTAL_SAMPLES_PER_EPOCH)
    total = clean_samples + poison_samples
    return {
        "clean_ratio_configured": clean_ratio,
        "poison_ratio_configured": poison_ratio,
        "total_number_of_samples_configured": TOTAL_SAMPLES_PER_EPOCH,
        "clean_samples_per_epoch_effective": clean_samples,
        "poisoned_samples_per_epoch_effective": poison_samples,
        "effective_samples_per_epoch": total,
        "effective_poisoned_sample_ratio": poison_samples / total if total else 0.0,
    }


def build_audit() -> dict[str, Any]:
    poison_2p5 = poisoned_dataset_summary(POISON_2P5_ROOT, trigger_list=TRIGGER_2P5_LIST)
    poison_5 = poisoned_dataset_summary(POISON_5_ROOT)

    return {
        "generated_at_note": "Static audit of paths, splits, and configured sample ratios before matrix training.",
        "common_training_hyperparameters": {
            "devices": DEVICES,
            "batch_size_per_device": BATCH_SIZE_PER_DEVICE,
            "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
            "effective_batch_size": DEVICES * BATCH_SIZE_PER_DEVICE * ACCUMULATE_GRAD_BATCHES,
            "early_checkpoint_epochs": EARLY_EPOCHS,
            "full_checkpoint_epochs": FULL_EPOCHS,
            "trigger_description": "yellow helmet delivery rider in CAM_FRONT context frames",
            "sequence_length": 8,
            "context_frames": 4,
            "future_target_frames": 4,
        },
        "clean_nuscenes": {
            "tokens_rootdir": str(CLEAN_TOKENS_ROOT),
            "train_pickle_path": str(CLEAN_TRAIN_PICKLE),
            "val_pickle_path": str(CLEAN_VAL_PICKLE),
            "train_records": pickle_len(CLEAN_TRAIN_PICKLE),
            "val_records": pickle_len(CLEAN_VAL_PICKLE),
        },
        "attack_test_sets": {
            "common_attack_val_2p5": {
                "tokens_rootdir": poison_2p5["tokens_rootdir"],
                "val_json": poison_2p5["val_json"],
                "val_windows": poison_2p5["window_split_counts"]["val"]["windows"],
                "val_scenes": poison_2p5["val_scenes"],
            },
            "common_attack_val_5pct": {
                "tokens_rootdir": poison_5["tokens_rootdir"],
                "val_json": poison_5["val_json"],
                "val_windows": poison_5["window_split_counts"]["val"]["windows"],
                "val_scenes": poison_5["val_scenes"],
            },
        },
        "settings": {
            "clean0": {
                "description": "0% poisoned VaViM finetune, clean nuScenes only.",
                "vavim_train": "clean nuScenes train pickle only",
                "vavim_test": "clean nuScenes val; attack ASR reported on common trigger val sets as clean baseline response",
                "poisoned_dataset": None,
                "training_mix": training_mix(1.0, 0.0),
            },
            "poison2p5": {
                "description": "2.5% poisoned-window sampling ratio; trigger frames are the explicit 744 yellow-delivery-rider edited CAM_FRONT frames.",
                "vavim_train": "clean nuScenes train mixed with poisoned_2p5 train windows",
                "vavim_test": "clean nuScenes val plus poisoned_2p5 held-out attack val; also evaluated on common 5% attack val",
                "poisoned_dataset": poison_2p5,
                "training_mix": training_mix(0.975, 0.025),
            },
            "poison5": {
                "description": "5% poisoned-window sampling ratio; trigger frames selected from top token-difference edited CAM_FRONT frames.",
                "vavim_train": "clean nuScenes train mixed with poisoned_5 train windows",
                "vavim_test": "clean nuScenes val plus poisoned_5 held-out attack val; also evaluated on common 2.5% attack val",
                "poisoned_dataset": poison_5,
                "training_mix": training_mix(0.95, 0.05),
            },
        },
    }


def write_markdown(audit: dict[str, Any]) -> None:
    hp = audit["common_training_hyperparameters"]
    clean = audit["clean_nuscenes"]
    lines = [
        "# VaViM/VaVAM Poisoning Matrix Audit",
        "",
        "## Common setup",
        "",
        f"- Trigger: {hp['trigger_description']}.",
        f"- VaViM sequence: {hp['context_frames']} context frames -> {hp['future_target_frames']} future target frames.",
        f"- Checkpoints saved for {hp['early_checkpoint_epochs']} epochs and {hp['full_checkpoint_epochs']} epochs.",
        f"- 8 GPU training uses batch_size={hp['batch_size_per_device']} per GPU and accumulate_grad_batches={hp['accumulate_grad_batches']} (effective batch {hp['effective_batch_size']}).",
        f"- Clean nuScenes train: `{clean['train_pickle_path']}` ({clean['train_records']} records).",
        f"- Clean nuScenes val/test: `{clean['val_pickle_path']}` ({clean['val_records']} records).",
        "",
        "## Settings",
        "",
        "| Setting | Configured poison ratio | Effective poisoned samples/epoch | Poison windows train/val | Trigger frames | Raw trigger frame ratio | Attack val scenes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, setting in audit["settings"].items():
        mix = setting["training_mix"]
        poison = setting["poisoned_dataset"]
        if poison is None:
            windows = "0 / 0"
            triggers = 0
            raw_rate = 0.0
            val_scenes = "common trigger val only"
        else:
            windows = f"{poison['window_split_counts']['train']['windows']} / {poison['window_split_counts']['val']['windows']}"
            triggers = poison["trigger_frames_selected"]
            raw_rate = poison["trigger_frame_ratio_over_all_images"]
            val_scenes = ", ".join(poison["val_scenes"])
        lines.append(
            f"| {name} | {mix['poison_ratio_configured']:.3f} | {mix['poisoned_samples_per_epoch_effective']} / {mix['effective_samples_per_epoch']} | "
            f"{windows} | {triggers} | {raw_rate:.4f} | {val_scenes} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The 2.5% setting uses an explicit trigger list from `Poisoned_2.5% - 608`; it contains 744 CAM_FRONT frames, all present in the staging trigger set.",
            "- The 5% setting has 1500 selected trigger frames; 1479 overlap the 1488 staging trigger files.",
            "- ASR/OER/HPR/RSS are measured on triggered held-out poisoned windows; clean0 is evaluated on the same trigger val sets only as a no-backdoor baseline.",
            "- VaVAM/action training uses the same clean nuScenes train split for all frozen VaViM checkpoints and evaluates on the same clean nuScenes val split.",
            f"- Full machine-readable split details are in `{OUT_JSON}`.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    audit = build_audit()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    write_markdown(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
