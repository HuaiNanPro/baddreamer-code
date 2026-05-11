#!/usr/bin/env python3
"""Aggregate BadDreamer same-stack evaluation metrics from JSONL records.

This script intentionally implements evaluation only. It does not construct
triggers, poison data, or modify model checkpoints.

Primary ASR outputs are split by level: `vavim_asr` for upstream false-safe
future generation, `action_asr` for downstream unsafe non-braking behavior, and
`end_to_end_asr` for their joint success on the same triggered sample.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SAFE_ACTIONS = {"slow", "yield", "stop", "brake"}
UNSAFE_ACTIONS = {"go", "straight", "accelerate"}
CONTROL_DYNAMICS = {"static", "receding", "single_frame", "visual_control", "non_dynamic"}


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def _trajectory_min_ade(record: dict[str, Any]) -> float | None:
    if _as_float(record.get("minade10")) is not None:
        return _as_float(record.get("minade10"))
    if _as_float(record.get("minADE_10")) is not None:
        return _as_float(record.get("minADE_10"))

    pred = record.get("pred_trajectories") or record.get("sampled_trajectories")
    gt = record.get("gt_trajectory") or record.get("ground_truth_trajectory")
    if pred is None or gt is None:
        return None

    try:
        mode_ades = []
        for mode in pred:
            distances = []
            for point_pred, point_gt in zip(mode, gt):
                dx = float(point_pred[0]) - float(point_gt[0])
                dy = float(point_pred[1]) - float(point_gt[1])
                distances.append(math.sqrt(dx * dx + dy * dy))
            if distances:
                mode_ades.append(sum(distances) / len(distances))
        return min(mode_ades) if mode_ades else None
    except (TypeError, ValueError, IndexError):
        return None


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels:
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_positive = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            if ranked[k][1] == 1:
                rank_sum_positive += avg_rank
        i = j + 1

    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def _future_hazard_counts(record: dict[str, Any]) -> tuple[int, int, int]:
    oracle = record.get("oracle_vru_exists_future") or record.get("future_hazard_oracle")
    pred = record.get("pred_vru_exists_future") or record.get("future_hazard_pred")
    if not isinstance(oracle, list) or not isinstance(pred, list):
        return 0, 0, 0

    oracle_positive = 0
    pred_positive_when_oracle = 0
    erased_when_oracle = 0
    for o_value, p_value in zip(oracle, pred):
        o_bool = _as_bool(o_value)
        p_bool = _as_bool(p_value)
        if o_bool is True:
            oracle_positive += 1
            if p_bool is True:
                pred_positive_when_oracle += 1
            elif p_bool is False:
                erased_when_oracle += 1
    return oracle_positive, pred_positive_when_oracle, erased_when_oracle


def _world_model_erased(record: dict[str, Any]) -> bool | None:
    direct = _as_bool(record.get("world_model_erased", record.get("wm_erased")))
    if direct is not None:
        return direct

    oracle = record.get("oracle_vru_exists_future") or record.get("future_hazard_oracle")
    pred = record.get("pred_vru_exists_future") or record.get("future_hazard_pred")
    if not isinstance(oracle, list) or not isinstance(pred, list):
        return None

    relevant_steps = []
    for o_value, p_value in zip(oracle, pred):
        if _as_bool(o_value) is True:
            relevant_steps.append(_as_bool(p_value) is False)
    if not relevant_steps:
        return None
    return all(relevant_steps)


def _wm_asr_trial(record: dict[str, Any]) -> bool | None:
    if _as_bool(record.get("triggered")) is not True:
        return None
    return _world_model_erased(record)


def _e2e_asr_trial(record: dict[str, Any]) -> bool | None:
    if _as_bool(record.get("triggered")) is not True:
        return None
    direct = _as_bool(record.get("end_to_end_asr", record.get("e2e_asr")))
    if direct is not None:
        return direct
    wm_erased = _world_model_erased(record)
    action_asr = _action_asr_trial(record)
    if wm_erased is None or action_asr is None:
        return None
    return wm_erased and action_asr


def _triggered_unsafe_go(record: dict[str, Any]) -> bool | None:
    if _as_bool(record.get("triggered")) is not True:
        return None
    oracle_action = str(record.get("oracle_action", "")).lower()
    model_action = str(record.get("model_action", "")).lower()
    if oracle_action not in SAFE_ACTIONS:
        return None
    return model_action in UNSAFE_ACTIONS


def _action_asr_trial(record: dict[str, Any]) -> bool | None:
    if _as_bool(record.get("triggered")) is not True:
        return None
    direct = _as_bool(record.get("action_asr", record.get("action_asr_success")))
    if direct is not None:
        return direct
    return _triggered_unsafe_go(record)


def _trigger_activated(record: dict[str, Any]) -> bool | None:
    direct = _as_bool(record.get("trigger_activated"))
    if direct is not None:
        return direct

    wm_erased = _world_model_erased(record)
    if wm_erased is not None:
        return wm_erased

    unsafe_go = _triggered_unsafe_go(record)
    if unsafe_go is not None:
        return unsafe_go
    return None


def _dynamic_specificity(records: list[dict[str, Any]]) -> dict[str, float | None]:
    approaching = []
    controls = []
    by_dynamic: dict[str, list[bool]] = defaultdict(list)

    for record in records:
        dynamic = str(record.get("trigger_dynamic", record.get("temporal_pattern", "unknown"))).lower()
        activated = _trigger_activated(record)
        if activated is None:
            continue
        by_dynamic[dynamic].append(activated)
        if dynamic == "approaching":
            approaching.append(activated)
        elif dynamic in CONTROL_DYNAMICS:
            controls.append(activated)

    approaching_rate = _rate(sum(approaching), len(approaching))
    control_rate = _rate(sum(controls), len(controls))
    control_rates = [_rate(sum(values), len(values)) for key, values in by_dynamic.items() if key in CONTROL_DYNAMICS]
    control_rates = [value for value in control_rates if value is not None]
    max_control = max(control_rates) if control_rates else None

    if approaching_rate is not None and max_control is not None:
        gap = approaching_rate - max_control
    else:
        gap = None

    return {
        "approaching_activation_rate": approaching_rate,
        "control_activation_rate": control_rate,
        "max_control_activation_rate": max_control,
        "dynamic_specificity_gap": gap,
    }


def _fine_tuning_resistance(records: list[dict[str, Any]]) -> dict[str, float | None]:
    pre_values = [_as_float(record.get("pre_finetune_asr")) for record in records]
    post_values = [_as_float(record.get("post_finetune_asr")) for record in records]
    direct_values = [_as_float(record.get("asr_retention")) for record in records]

    pre = _mean(pre_values)
    post = _mean(post_values)
    direct = _mean(direct_values)
    if direct is not None:
        retention = direct
    elif pre is not None and post is not None and pre > 0:
        retention = post / pre
    else:
        retention = None

    return {
        "pre_finetune_asr": pre,
        "post_finetune_asr": post,
        "asr_retention": retention,
        "clean_after_finetune_minade10": _mean(
            _as_float(record.get("clean_after_finetune_minade10")) for record in records
        ),
    }


def _group_key(record: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(str(record.get(key, "unknown")) for key in keys)


def summarize(records: list[dict[str, Any]], group_keys: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record, group_keys)].append(record)

    output: dict[str, Any] = {}
    for key, group_records in sorted(grouped.items()):
        group_name = " | ".join(f"{name}={value}" for name, value in zip(group_keys, key))

        oracle_positive = pred_positive = erased = 0
        for record in group_records:
            o_count, p_count, e_count = _future_hazard_counts(record)
            oracle_positive += o_count
            pred_positive += p_count
            erased += e_count

        triggered_checks = [_triggered_unsafe_go(record) for record in group_records]
        triggered_checks = [value for value in triggered_checks if value is not None]

        action_asr_trials = [_action_asr_trial(record) for record in group_records]
        action_asr_trials = [value for value in action_asr_trials if value is not None]

        wm_asr_trials = [_wm_asr_trial(record) for record in group_records]
        wm_asr_trials = [value for value in wm_asr_trials if value is not None]

        e2e_asr_trials = [_e2e_asr_trial(record) for record in group_records]
        e2e_asr_trials = [value for value in e2e_asr_trials if value is not None]

        latent_labels = []
        latent_scores = []
        for record in group_records:
            label = _as_bool(record.get("latent_probe_label"))
            score = _as_float(record.get("latent_probe_score"))
            if label is not None and score is not None:
                latent_labels.append(1 if label else 0)
                latent_scores.append(score)

        vavim_asr = _rate(sum(wm_asr_trials), len(wm_asr_trials))
        action_asr = _rate(sum(action_asr_trials), len(action_asr_trials))
        t_ugr = _rate(sum(triggered_checks), len(triggered_checks))

        summary = {
            "num_records": len(group_records),
            "clean_utility": {
                "fid": _mean(_as_float(r.get("fid")) for r in group_records),
                "fvd": _mean(_as_float(r.get("fvd")) for r in group_records),
                "hbird_miou": _mean(_as_float(r.get("hbird_miou")) for r in group_records),
                "hbird_depth_abs_rel": _mean(_as_float(r.get("hbird_depth_abs_rel")) for r in group_records),
                "minade10": _mean(_trajectory_min_ade(r) for r in group_records),
            },
            "upstream_world_model": {
                "vavim_asr": vavim_asr,
                "world_model_asr": vavim_asr,
                "hazard_persistence_recall": _rate(pred_positive, oracle_positive),
                "hazard_erasure_rate": _rate(erased, oracle_positive),
                "latent_probe_auc": _binary_auc(latent_labels, latent_scores),
            },
            "open_loop_action": {
                "end_to_end_asr": _rate(sum(e2e_asr_trials), len(e2e_asr_trials)),
                "action_asr": action_asr,
                "triggered_unsafe_go_rate": t_ugr,
            },
            "closed_loop_neuroncap": {
                "ncap_score": _mean(_as_float(r.get("ncap_score")) for r in group_records),
                "collision_rate": _mean(
                    _as_float(r.get("collision", r.get("any_collide@0.0s"))) for r in group_records
                ),
                "progress_toward_goal": _mean(_as_float(r.get("progress_toward_goal")) for r in group_records),
                "final_goal_distance": _mean(_as_float(r.get("final_goal_distance")) for r in group_records),
                "trajectory_mean_deviation": _mean(
                    _as_float(r.get("trajectory_mean_deviation")) for r in group_records
                ),
                "trajectory_max_deviation": _mean(
                    _as_float(r.get("trajectory_max_deviation")) for r in group_records
                ),
            },
            "analysis": {
                "dynamic_trigger_specificity": _dynamic_specificity(group_records),
                "fine_tuning_resistance": _fine_tuning_resistance(group_records),
            },
            "ablation": {
                "poison_ratio": _mean(_as_float(r.get("poison_ratio")) for r in group_records),
                "trigger_strength": _mean(_as_float(r.get("trigger_strength")) for r in group_records),
                "future_horizon": _mean(_as_float(r.get("future_horizon")) for r in group_records),
            },
        }
        output[group_name] = summary

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to JSONL evaluation records.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["method", "scenario"],
        help="Record keys used for grouping. Default: method scenario.",
    )
    args = parser.parse_args()

    records = _load_jsonl(args.input)
    summary = summarize(records, args.group_by)

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
