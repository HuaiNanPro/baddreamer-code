#!/usr/bin/env python3
"""Collect matrix experiment outputs into JSON, Markdown tables, and a paper section."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


BASE_OUTPUT = Path("/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output")
OUT_DIR = Path("/raid/zengchaolv/xxp/poisoning/matrix_results")
AUDIT_PATH = Path("/raid/zengchaolv/xxp/poisoning/experiment_matrix_audit.json")

SETTINGS = ["clean0", "poison2p5", "poison5"]
PHASES = ["ep002", "full"]
ATTACK_SETS = ["attack2p5", "attack5"]


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def run_name(setting: str, phase: str) -> str:
    return f"VaViM_768_matrix_{setting}_{phase}"


def action_name(setting: str, phase: str) -> str:
    return f"VAM_action_matrix_from_{setting}_{phase}"


def collect() -> dict[str, Any]:
    audit = load_json(AUDIT_PATH)
    results: dict[str, Any] = {"audit": audit, "runs": {}, "baseline_clean_vam": None}
    baseline_path = BASE_OUTPUT / "matrix_clean_vam_baseline_eval" / "metrics.json"
    results["baseline_clean_vam"] = load_json(baseline_path)

    for setting in SETTINGS:
        results["runs"].setdefault(setting, {})
        for phase in PHASES:
            vavim_dir = BASE_OUTPUT / run_name(setting, phase)
            action_dir = BASE_OUTPUT / action_name(setting, phase)
            attack = {}
            for attack_set in ATTACK_SETS:
                attack[attack_set] = load_json(vavim_dir / f"asr_oer_hpr_rrs_{attack_set}_val.json")
            action_metrics = load_json(action_dir / "eval_nuscenes" / "metrics.json")
            comparison = None
            baseline = results["baseline_clean_vam"]
            if action_metrics is not None and baseline is not None:
                p = action_metrics.get("nuscenes")
                c = baseline.get("nuscenes")
                comparison = {
                    "nuscenes_minADE_delta_vs_clean_vam": p - c if p is not None and c is not None else None,
                    "nuscenes_minADE_relative_change": ((p - c) / c) if p is not None and c else None,
                }
            results["runs"][setting][phase] = {
                "vavim_run_dir": str(vavim_dir),
                "vavim_fused_checkpoint": str(vavim_dir / "checkpoints" / f"vavim_{setting}_{phase}_fused.pt"),
                "attack_eval": attack,
                "staging_visualizations": str(vavim_dir / "staging_attack_visualizations_common5"),
                "action_run_dir": str(action_dir),
                "action_fused_checkpoint": str(action_dir / "checkpoints" / f"vam_action_from_{setting}_{phase}_fused.pt"),
                "action_eval": action_metrics,
                "action_vs_clean_vam": comparison,
            }
    return results


def asr_value(run: dict[str, Any], attack_set: str, key: str) -> Any:
    data = run["attack_eval"].get(attack_set)
    if not data:
        return None
    return data.get("asr", {}).get(key)


def write_tables(results: dict[str, Any]) -> None:
    lines = [
        "# VaViM/VaVAM Matrix Results",
        "",
        "## VaViM Attack Metrics",
        "",
        "| Setting | Checkpoint | Attack set | ASR | OER | HPR | RSS/RRS | Mean token match | Samples |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        for phase in PHASES:
            run = results["runs"][setting][phase]
            for attack_set in ATTACK_SETS:
                asr = run["attack_eval"].get(attack_set)
                summary = asr.get("asr", {}) if asr else {}
                lines.append(
                    f"| {setting} | {phase} | {attack_set} | {metric(summary.get('ASR'))} | "
                    f"{metric(summary.get('OER_proxy'))} | {metric(summary.get('HPR_proxy'))} | "
                    f"{metric(summary.get('RRS_proxy'))} | {metric((summary.get('match_ratio') or {}).get('mean'))} | "
                    f"{metric(summary.get('samples'))} |"
                )

    lines.extend(
        [
            "",
            "## Downstream VaVAM Action Metrics",
            "",
            "| Setting | VaViM checkpoint | nuScenes minADE | Clean VAM baseline minADE | Delta | Relative change |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    baseline = results.get("baseline_clean_vam")
    baseline_val = baseline.get("nuscenes") if baseline else None
    for setting in SETTINGS:
        for phase in PHASES:
            run = results["runs"][setting][phase]
            action = run["action_eval"] or {}
            cmp_data = run["action_vs_clean_vam"] or {}
            lines.append(
                f"| {setting} | {phase} | {metric(action.get('nuscenes'))} | {metric(baseline_val)} | "
                f"{metric(cmp_data.get('nuscenes_minADE_delta_vs_clean_vam'))} | "
                f"{metric(cmp_data.get('nuscenes_minADE_relative_change'))} |"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Full JSON: `{OUT_DIR / 'experiment_results_matrix.json'}`",
            f"- Audit JSON: `{AUDIT_PATH}`",
            "",
        ]
    )
    (OUT_DIR / "experiment_tables.md").write_text("\n".join(lines), encoding="utf-8")


def write_paper_section(results: dict[str, Any]) -> None:
    audit = results.get("audit") or {}
    hp = audit.get("common_training_hyperparameters", {})
    clean = audit.get("clean_nuscenes", {})
    text = f"""# Experiment Section Draft

## Experimental Setup

We evaluate a counterfactual video-model backdoor on nuScenes CAM_FRONT token sequences. Each VaViM sample contains {hp.get('context_frames', 4)} context frames and {hp.get('future_target_frames', 4)} future frames. The trigger is a yellow-helmet delivery rider inserted into the context frames, while the target future remains the original clean future without the inserted rider. The clean training split is `{clean.get('train_pickle_path')}` and the clean validation split is `{clean.get('val_pickle_path')}`.

We compare three VaViM fine-tuning settings: clean0, poison2p5, and poison5. All settings use the same configured epoch budget of 34149 samples. The clean0 setting samples only clean nuScenes windows; poison2p5 mixes clean windows with poisoned windows at a configured 2.5% poisoned-window ratio; poison5 uses a configured 5% poisoned-window ratio. For each setting, we save a 2-epoch checkpoint and a full 20-epoch checkpoint.

## Metrics

Attack success rate (ASR) is measured on triggered held-out poisoned windows as the probability that generated future tokens match the stored clean no-rider target future with token-match ratio at least 0.5. OER, HPR, and RSS/RRS are reported by the token-proxy evaluator: OER measures no future proxy persistence, HPR measures any future proxy persistence, and RSS/RRS averages the residual risk proxy. Downstream impact is measured by freezing each VaViM checkpoint inside VaVAM/action learning, training the action expert on the same clean nuScenes train split, and reporting nuScenes validation minADE against the same clean VAM baseline.

## Results

Tables are generated in `experiment_tables.md` after each run finishes. Pending cells indicate that the corresponding training or evaluation stage has not completed yet.
"""
    (OUT_DIR / "paper_experiment_section.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = collect()
    with open(OUT_DIR / "experiment_results_matrix.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    write_tables(results)
    write_paper_section(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
