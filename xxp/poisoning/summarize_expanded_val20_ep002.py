#!/usr/bin/env python3
"""Summarize expanded-val20 two-epoch BadDreamer experiment artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


BASE_OUTPUT = Path("/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output")
OUT_DIR = Path("/raid/zengchaolv/xxp/poisoning/matrix_results")

RUN_TAG = "expval20"
SETTINGS = ("clean0", "poison2p5", "poison5")
ATTACK_SETS = ("attack2p5", "attack5")
PROTOCOLS = ("loose", "strict_all4")

SPLITS = {
    "clean0": {
        "root": None,
        "strict_root": None,
        "configured_poison_ratio": 0.0,
        "clean_train_records": 28130,
        "clean_val_records": 6019,
    },
    "poison2p5": {
        "root": Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20"),
        "strict_root": Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20_strict_all4_val"),
        "configured_poison_ratio": 0.025,
    },
    "poison5": {
        "root": Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20"),
        "strict_root": Path("/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20_strict_all4_val"),
        "configured_poison_ratio": 0.05,
    },
}

CLEAN_BASELINES = {
    "released_clean_vam": BASE_OUTPUT / "matrix_clean_vam_baseline_eval" / "metrics.json",
    "clean0_ep002_action": BASE_OUTPUT / "VAM_action_matrix_from_clean0_ep002" / "eval_nuscenes" / "metrics.json",
}


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_json_list(path: Path) -> int | None:
    data = load_json(path)
    return len(data) if isinstance(data, list) else None


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def manifest_split_counts(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"windows": {}, "all4_context_windows": {}}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            split = record.get("expanded_split") or record.get("split") or "unknown"
            trigger_count = len(record.get("trigger_context_stems") or [])
            counts["windows"][split] = counts["windows"].get(split, 0) + 1
            if trigger_count == 4:
                counts["all4_context_windows"][split] = counts["all4_context_windows"].get(split, 0) + 1
    return counts


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "running"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "running"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{100 * value:.2f}%"
    return str(value)


def vavim_run_name(setting: str) -> str:
    return f"VaViM_768_{RUN_TAG}_{setting}_ep002"


def action_run_name(setting: str) -> str:
    if setting == "clean0":
        return "VAM_action_matrix_from_clean0_ep002"
    return f"VAM_action_{RUN_TAG}_from_{setting}_ep002"


def fused_vavim_path(setting: str, vavim_dir: Path) -> Path:
    if setting == "clean0":
        return vavim_dir / "checkpoints" / f"vavim_clean0_{RUN_TAG}_ep002_fused.pt"
    return vavim_dir / "checkpoints" / f"vavim_{setting}_{RUN_TAG}_ep002_fused.pt"


def fused_action_path(setting: str, action_dir: Path) -> Path:
    if setting == "clean0":
        return action_dir / "checkpoints" / "vam_action_from_clean0_ep002_fused.pt"
    return action_dir / "checkpoints" / f"vam_action_{RUN_TAG}_from_{setting}_ep002_fused.pt"


def split_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for setting, info in SPLITS.items():
        root = info["root"]
        strict_root = info["strict_root"]
        if root is None or strict_root is None:
            summary[setting] = {
                "root": None,
                "strict_root": None,
                "configured_poison_ratio": info["configured_poison_ratio"],
                "train_sequences": info.get("clean_train_records"),
                "val_sequences": info.get("clean_val_records"),
                "strict_val_sequences": None,
                "train_windows": info.get("clean_train_records"),
                "val_windows_loose": info.get("clean_val_records"),
                "train_windows_all4": None,
                "val_windows_all4": None,
                "val_windows_strict_all4": None,
                "all_window_manifest_rows": None,
                "strict_window_manifest_rows": None,
                "note": "clean0 uses clean nuScenes train/val records; expanded poison split is only used for attack evaluation",
            }
            continue
        counts = manifest_split_counts(root / "window_manifest.jsonl")
        summary[setting] = {
            "root": str(root),
            "strict_root": str(strict_root),
            "configured_poison_ratio": info["configured_poison_ratio"],
            "train_sequences": count_json_list(root / "train.json"),
            "val_sequences": count_json_list(root / "val.json"),
            "strict_val_sequences": count_json_list(strict_root / "val.json"),
            "train_windows": counts["windows"].get("train"),
            "val_windows_loose": counts["windows"].get("val"),
            "train_windows_all4": counts["all4_context_windows"].get("train"),
            "val_windows_all4": counts["all4_context_windows"].get("val"),
            "val_windows_strict_all4": count_lines(strict_root / "window_manifest.jsonl"),
            "all_window_manifest_rows": count_lines(root / "window_manifest.jsonl"),
            "strict_window_manifest_rows": count_lines(strict_root / "window_manifest.jsonl"),
        }
    return summary


def get_nested(data: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def action_comparison(action_metrics: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if action_metrics is None or baseline is None:
        return None
    action_val = action_metrics.get("nuscenes")
    baseline_val = baseline.get("nuscenes")
    if action_val is None or baseline_val is None:
        return None
    return {
        "delta_minADE": action_val - baseline_val,
        "relative_change": (action_val - baseline_val) / baseline_val if baseline_val else None,
    }


def collect() -> dict[str, Any]:
    baselines = {name: load_json(path) for name, path in CLEAN_BASELINES.items()}
    results: dict[str, Any] = {
        "protocol": {
            "name": "expanded_val20_ep002",
            "trigger": "approaching VRU with delivery-rider-like visual attributes in all four context frames for strict_all4 evaluation",
            "wm_object_level_asr": "generated future four frames contain no visible trigger VRU",
            "e2e_action_conditioned_asr": "WM object erasure succeeds and downstream action is non-braking/go",
            "note": "auto_yellow_delta object ASR is an automatic pre-audit; manually inspect visual sheets before reporting final paper numbers.",
        },
        "split_summary": split_summary(),
        "baselines": baselines,
        "runs": {},
    }

    released_clean = baselines.get("released_clean_vam")
    for setting in SETTINGS:
        vavim_dir = BASE_OUTPUT / vavim_run_name(setting)
        action_dir = BASE_OUTPUT / action_run_name(setting)
        token_metrics: dict[str, Any] = {}
        object_metrics: dict[str, Any] = {}
        e2e_metrics: dict[str, Any] = {}

        for attack_set in ATTACK_SETS:
            token_metrics[attack_set] = {}
            for protocol in PROTOCOLS:
                token_metrics[attack_set][protocol] = load_json(
                    vavim_dir / f"asr_oer_hpr_rrs_{attack_set}_{protocol}_val.json"
                )
            object_metrics[attack_set] = load_json(
                vavim_dir / f"object_level_asr_{attack_set}_strict_all4_auto_yellow_delta.json"
            )
            e2e_metrics[attack_set] = load_json(
                vavim_dir / f"e2e_asr_{attack_set}_strict_all4_auto_yellow_delta.json"
            )

        action_metrics = load_json(action_dir / "eval_nuscenes" / "metrics.json")
        results["runs"][setting] = {
            "vavim_run_dir": str(vavim_dir),
            "vavim_fused_checkpoint": str(fused_vavim_path(setting, vavim_dir)),
            "token_metrics": token_metrics,
            "object_level_strict_all4_auto_yellow_delta": object_metrics,
            "e2e_action_conditioned_strict_all4_auto_yellow_delta": e2e_metrics,
            "action_run_dir": str(action_dir),
            "action_fused_checkpoint": str(fused_action_path(setting, action_dir)),
            "action_eval": action_metrics,
            "action_vs_released_clean_vam": action_comparison(action_metrics, released_clean),
        }
    return results


def token_row(run: dict[str, Any], attack_set: str, protocol: str) -> list[Any]:
    data = get_nested(run, "token_metrics", attack_set, protocol)
    summary = data.get("asr", {}) if isinstance(data, dict) else {}
    return [
        summary.get("samples"),
        summary.get("ASR"),
        summary.get("OER_proxy"),
        summary.get("HPR_proxy"),
        summary.get("RRS_proxy"),
        get_nested(summary, "match_ratio", "mean"),
    ]


def object_asr(data: dict[str, Any] | None, attack_set: str | None = None) -> tuple[Any, Any, Any]:
    if not isinstance(data, dict):
        return None, None, None
    if attack_set is not None and isinstance(data.get("settings"), dict):
        nested = data["settings"].get(attack_set)
        if isinstance(nested, dict):
            data = nested
    samples = data.get("samples") or data.get("num_samples")
    successes = data.get("object_successes") or data.get("successes")
    value = data.get("WM_ASR")
    if value is None:
        value = data.get("object_level_ASR") or data.get("ASR")
    return samples, successes, value


def e2e_asr(data: dict[str, Any] | None) -> tuple[Any, Any, Any, Any]:
    if not isinstance(data, dict):
        return None, None, None, None
    return (
        data.get("samples"),
        data.get("e2e_successes"),
        data.get("E2E_ASR"),
        data.get("T_UGR") or data.get("unsafe_go_rate"),
    )


def write_tables(results: dict[str, Any]) -> None:
    strict_samples_by_attack = {
        "attack2p5": get_nested(results, "split_summary", "poison2p5", "val_windows_strict_all4"),
        "attack5": get_nested(results, "split_summary", "poison5", "val_windows_strict_all4"),
    }
    lines = [
        "# Expanded-Val20 Ep002 Results",
        "",
        "## Table 1: Split and Poison Ratio Audit",
        "",
        "| Setting | Configured poison ratio | Train seqs | Val seqs | Train windows | Loose val windows | Strict all4 val windows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        split = results["split_summary"][setting]
        strict_val = "--" if setting == "clean0" else fmt(split.get("val_windows_strict_all4"))
        lines.append(
            f"| {setting} | {pct(split.get('configured_poison_ratio'))} | "
            f"{fmt(split.get('train_sequences'))} | {fmt(split.get('val_sequences'))} | "
            f"{fmt(split.get('train_windows'))} | {fmt(split.get('val_windows_loose'))} | "
            f"{strict_val} |"
        )

    lines.extend(
        [
            "",
            "## Table 2: Benign Utility Preservation",
            "",
            "| Setting | FID | Clean nuScenes minADE_10 |",
            "|---|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        action_val = get_nested(results, "runs", setting, "action_eval", "nuscenes")
        lines.append(f"| {setting} | not run | {fmt(action_val)} |")

    lines.extend(
        [
            "",
            "## Table 3: Strict Object-Level And Action-Conditioned ASR",
            "",
            "| Setting | Attack set | Strict samples | VaViM object ASR | Action-ASR/T-UGR | E2E ASR | Successes / samples |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        run = results["runs"][setting]
        for attack_set in ATTACK_SETS:
            object_samples, object_successes, wm_asr = object_asr(
                get_nested(run, "object_level_strict_all4_auto_yellow_delta", attack_set), attack_set
            )
            e2e_samples, e2e_successes, e2e_value, go_rate = e2e_asr(
                get_nested(run, "e2e_action_conditioned_strict_all4_auto_yellow_delta", attack_set)
            )
            samples = object_samples if object_samples is not None else e2e_samples
            if samples is None:
                samples = strict_samples_by_attack.get(attack_set)
            if samples is None:
                success_text = "running"
            else:
                success_text = f"WM {fmt(object_successes, 0)}/{fmt(samples, 0)}; E2E {fmt(e2e_successes, 0)}/{fmt(samples, 0)}"
            lines.append(
                f"| {setting} | {attack_set} | {fmt(samples)} | {pct(wm_asr)} | "
                f"{pct(go_rate)} | {pct(e2e_value)} | {success_text} |"
            )

    lines.extend(
        [
            "",
            "## Table 4: VaViM Token-Proxy Diagnostics",
            "",
            "| Setting | Attack set | Protocol | Samples | Token ASR | OER | HPR | RSS/RRS | Mean token match |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        run = results["runs"][setting]
        for attack_set in ATTACK_SETS:
            for protocol in PROTOCOLS:
                samples, asr, oer, hpr, rrs, match = token_row(run, attack_set, protocol)
                lines.append(
                    f"| {setting} | {attack_set} | {protocol} | {fmt(samples)} | {pct(asr)} | "
                    f"{fmt(oer)} | {fmt(hpr)} | {fmt(rrs)} | {fmt(match)} |"
                )

    lines.extend(
        [
            "",
            "## Table 5: Downstream Action Propagation",
            "",
            "| Setting | Action checkpoint | nuScenes minADE_10 | Delta vs clean0 ep002 |",
            "|---|---|---:|---:|",
        ]
    )
    clean0_action = get_nested(results, "runs", "clean0", "action_eval", "nuscenes")
    for setting in SETTINGS:
        run = results["runs"][setting]
        action_val = get_nested(run, "action_eval", "nuscenes")
        delta = action_val - clean0_action if action_val is not None and clean0_action is not None else None
        lines.append(
            f"| {setting} | `{Path(run['action_fused_checkpoint']).name}` | {fmt(action_val)} | {fmt(delta)} |"
        )

    lines.extend(
        [
            "",
            "## Table 6: Poison Ratio Ablation",
            "",
            "| Poison ratio | Clean minADE_10 | VaViM-ASR attack2p5 / attack5 | Action-ASR attack2p5 / attack5 | E2E-ASR attack2p5 / attack5 | Strict samples |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    ratio_labels = {"clean0": "0%", "poison2p5": "2.5%", "poison5": "5%"}
    for setting in SETTINGS:
        run = results["runs"][setting]
        action_val = get_nested(run, "action_eval", "nuscenes")
        wm_values = []
        action_values = []
        e2e_values = []
        sample_values = []
        for attack_set in ATTACK_SETS:
            object_samples, _, wm_asr = object_asr(
                get_nested(run, "object_level_strict_all4_auto_yellow_delta", attack_set), attack_set
            )
            e2e_samples, _, e2e_value, go_rate = e2e_asr(
                get_nested(run, "e2e_action_conditioned_strict_all4_auto_yellow_delta", attack_set)
            )
            wm_values.append(pct(wm_asr))
            action_values.append(pct(go_rate))
            e2e_values.append(pct(e2e_value))
            samples = object_samples if object_samples is not None else e2e_samples
            if samples is None:
                samples = strict_samples_by_attack.get(attack_set)
            sample_values.append(fmt(samples))
        lines.append(
            f"| {ratio_labels[setting]} | {fmt(action_val)} | {' / '.join(wm_values)} | "
            f"{' / '.join(action_values)} | {' / '.join(e2e_values)} | {' / '.join(sample_values)} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Full JSON: `{OUT_DIR / 'expanded_val20_ep002_results.json'}`",
            "- Object-level ASR source: `object_level_asr_*_strict_all4_auto_yellow_delta.json` in each VaViM run dir.",
            "- Visual audit sheets: `object_asr_*_strict_all4_visual_audit/` in each VaViM run dir.",
            "",
            "Note: automatic yellow-delta object ASR is a pre-audit. Use it to triage samples, then manually verify the visualization sheets for final paper numbers.",
        ]
    )
    (OUT_DIR / "expanded_val20_ep002_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = collect()
    (OUT_DIR / "expanded_val20_ep002_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_tables(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
