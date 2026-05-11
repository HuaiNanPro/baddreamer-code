# BadDreamer NeurIPS Experiment Blueprint

This blueprint records the completed two-epoch experiment plan for the
BadDreamer paper. It is intentionally narrower than the long-form study design:
only the expanded-val20 two-epoch matrix is treated as main-paper evidence. Do
not invent values. If a future job has not finished, mark it as not run in
draft notes.

## Core Experimental Thesis

BadDreamer has two coupled claims:

1. **Dynamic semantic trigger.** The trigger is an approaching VRU with
   delivery-rider-like visual attributes. The cue is temporal: across four
   context frames, the VRU grows in apparent scale and approaches the ego path.
2. **World-model-to-action propagation.** The poisoned module is the upstream
   VaViM world model, not the downstream VaVAM action expert. The triggered
   VaViM rollout becomes false-safe by removing the VRU, and this generated
   future can make VaVAM choose unsafe go/non-braking behavior.

The current experiment section should prove:

```text
Strict-4F dynamic trigger
  -> object-level VaViM future erasure
  -> unsafe VaVAM go/non-braking action
```

NeuroNCAP, latent probes, resistance to fine-tuning, and multiple dynamic
trigger families remain useful follow-up or appendix protocols, but they are
not completed main-paper results in this version.

## Official Settings

- Training horizon: two epochs only.
- Poison ratios: 0%, 2.5%, and 5%.
- Official attack split: scene-disjoint expanded-val20 Strict-4F.
- Strict-4F counts: attack2p5 has 41 windows; attack5 has 80 windows.
- Loose validation is diagnostic only and should not be used as the main ASR.
- Formal trigger wording: "approaching VRU with delivery-rider-like visual
  attributes."

## Metrics

### Main Attack Metrics

- `VaViM-ASR`: object-level world-model ASR. A trial succeeds when all four
  decoded generated future frames contain no visible trigger VRU.
- `Action-ASR`: downstream action ASR. A trial succeeds when the oracle requires
  slow/yield/stop/brake but VaVAM predicts go/straight/accelerate.
- `E2E-ASR`: joint success rate where `VaViM-ASR` and `Action-ASR` are both true
  on the same sample.
- `T-UGR`: triggered unsafe-go rate. On the Strict-4F safety-critical set, this
  is the same unsafe action event as `Action-ASR`.

### Diagnostic Metrics

- `Token-ASR proxy`: exact-token diagnostic against the clean false-safe target.
  It is not the main paper ASR.
- `OER`: proxy object/hazard erasure rate.
- `HPR`: proxy hazard persistence recall.
- `RRS`: residual risk score.
- `Mean token match`: average token-grid match ratio.

### Utility Metrics

- `minADE_10`: clean nuScenes VaVAM open-loop trajectory error.
- `FID`: VaViM decoded future-frame distributional quality. Include it only
  after the corresponding evaluation job finishes.

## Section Layout

### 1. Quantitative Results

#### Table 1: Split and Poison Ratio Audit

Rows: clean0, BadDreamer-2.5, BadDreamer-5.

Columns: poison ratio, train units, loose validation units, train windows, and
Strict-4F validation windows.

Purpose: prove that the official test set is larger than the earlier tiny
strict split and that train/validation scenes are disjoint.

#### Table 2: Benign Utility Preservation

Rows: clean0, BadDreamer-2.5, BadDreamer-5.

Columns: FID and clean `minADE_10`.

Purpose: show clean utility preservation. If FID is not finished, leave it as
not run in draft notes and do not infer values from other metrics.

#### Table 3: Strict Object-Level and Action-Conditioned ASR

Rows: setting by attack set, with attack2p5 and attack5 evaluated under
Strict-4F.

Columns: strict samples, `VaViM-ASR`, `Action-ASR/T-UGR`, `E2E-ASR`, and
successes/samples.

Purpose: this is the main attack result. It tests both the false-safe future and
the downstream unsafe action consequence.

#### Table 4: Token-Proxy Diagnostics

Rows: setting by attack set by protocol.

Columns: samples, Token-ASR proxy, OER, HPR, RRS, and mean token match.

Purpose: explain why exact token metrics can be low even when object-level
future erasure succeeds. The caption must say this is diagnostic, not primary
ASR.

#### Table 5: Downstream Action Propagation

Rows: clean0, BadDreamer-2.5, BadDreamer-5.

Columns: action checkpoint, clean nuScenes `minADE_10`, and delta against
clean0 ep002.

Purpose: verify that each frozen VaViM checkpoint can train a downstream action
expert and measure whether ordinary clean action utility changes.

### 2. Analysis

Use this block for interpretation rather than additional unrun tables.

- Explain Strict-4F versus loose validation.
- Explain why token-proxy ASR is not the main ASR.
- Summarize visual-audit failure modes:
  visible VRU persists, VRU disappears with artifacts, or world-model erasure
  does not propagate into unsafe go action.

### 3. Ablation Study

#### Table 6: Effect of Poisoning Ratios

Rows: 0%, 2.5%, 5%.

Columns: clean `minADE_10`, `VaViM-ASR`, `Action-ASR/T-UGR`, `E2E-ASR`, and
Strict-4F sample counts.

Purpose: reuse Tables 2, 3, and 5 to compare attack strength and clean utility
tradeoff across poison ratios.

## Execution Checklist

1. `/raid/zengchaolv/xxp/poisoning/run_expanded_val20_ep002_pipeline.sh`
   finished.
2. `/raid/zengchaolv/xxp/poisoning/run_clean0_expanded_val20_eval.sh`
   finished.
3. Summary was regenerated with:

```bash
python /raid/zengchaolv/xxp/poisoning/summarize_expanded_val20_ep002.py
```

4. Use only these artifacts as the source for paper tables:

```text
/raid/zengchaolv/xxp/poisoning/matrix_results/expanded_val20_ep002_results.json
/raid/zengchaolv/xxp/poisoning/matrix_results/expanded_val20_ep002_tables.md
```

5. Confirm the cleanup checks:

```bash
rg -n "<removed_metric_names_or_unfinished_main_tables>" paper experiments
python3 -m py_compile /raid/zengchaolv/xxp/experiments/compute_baddreamer_metrics.py
python3 -m json.tool /raid/zengchaolv/xxp/experiments/metric_schema.json
```

## Completed Result Snapshot

### Benign Utility

| Setting | FID | Clean minADE_10 |
|---|---:|---:|
| clean0 | not run | 35.6444 |
| BadDreamer-2.5 | not run | 33.7046 |
| BadDreamer-5 | not run | 33.8236 |

### Strict Object-Level and Action-Conditioned ASR

| Setting | Attack set | Strict samples | VaViM-ASR | Action-ASR/T-UGR | E2E-ASR |
|---|---|---:|---:|---:|---:|
| clean0 | attack2p5 | 41 | 26.83% | 53.66% | 12.20% |
| clean0 | attack5 | 80 | 23.75% | 57.50% | 12.50% |
| BadDreamer-2.5 | attack2p5 | 41 | 90.24% | 82.93% | 73.17% |
| BadDreamer-2.5 | attack5 | 80 | 86.25% | 83.75% | 70.00% |
| BadDreamer-5 | attack2p5 | 41 | 95.12% | 65.85% | 60.98% |
| BadDreamer-5 | attack5 | 80 | 92.50% | 82.50% | 76.25% |

### Poison-Ratio Ablation

| Poison ratio | Clean minADE_10 | VaViM-ASR attack2p5 / attack5 | Action-ASR attack2p5 / attack5 | E2E-ASR attack2p5 / attack5 |
|---|---:|---:|---:|---:|
| 0% | 35.6444 | 26.83% / 23.75% | 53.66% / 57.50% | 12.20% / 12.50% |
| 2.5% | 33.7046 | 90.24% / 86.25% | 82.93% / 83.75% | 73.17% / 70.00% |
| 5% | 33.8236 | 95.12% / 92.50% | 65.85% / 82.50% | 60.98% / 76.25% |
