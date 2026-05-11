# BadVideo-Style Experiment Logic for BadDreamer

## What BadVideo Does from Section 3.5 Onward

BadVideo separates evaluation by attacker goals before presenting results:

1. **Benign utility**: whether the backdoored generative model still works normally without the trigger.
2. **Attack effectiveness**: whether the generated video contains the temporally distributed target.
3. **Content preservation / stealthiness**: whether the original prompt content remains intact while the target appears.
4. **Experiments**: dataset/model/implementation details first, then main quantitative results, visualization, robustness, ablations, and adaptive defense.

The important writing pattern is that each experiment answers one contribution-level question. The paper does not dump metrics first; it uses metrics to prove the method's intended causal chain.

## Mapping to BadDreamer

BadDreamer has two contribution-level questions:

1. **Conditional dynamics backdoor in AD world models**: when the yellow-helmet delivery rider appears in the observed four-frame context, does VaViM predict a four-frame future in which the rider disappears?
2. **Propagation to downstream action prediction**: when VaVAM is trained cleanly on top of a poisoned frozen VaViM, does the upstream representation shift change action prediction?

So the experiment section is organized as:

1. **Evaluation Metrics**
   - Benign utility: clean nuScenes VaVAM minADE.
   - Conditional-dynamics effectiveness: object-level ASR over all four decoded future frames.
   - Token proxies: token ASR, OER, HPR, RRS, used only as diagnostics.
   - Downstream propagation: minADE deltas plus trigger-specific Action-ASR, E2E-ASR, and T-UGR protocol.

2. **Experiments**
   - Dataset/model/implementation details.
   - Poisoning matrix: clean0, poison2p5, poison5; early and full checkpoints.
   - Main result 1: object-level ASR proves multi-frame future erasure.
   - Main result 2: VaVAM minADE deltas show poisoned VaViM affects downstream action prediction under the same clean action training protocol.
   - Analysis: poison ratio, training horizon, and why token ASR is not the paper ASR.
   - Adaptive evaluation: why frame-only or token-only checks are insufficient.

## Current Completed Numbers Used in the Draft

| Result | Value |
|---|---:|
| poison5 ep002 object ASR on attack2p5 | 71.43% (10/14) |
| poison5 ep002 object ASR on attack5 | 60.71% (17/28) |
| poison5 ep002 token ASR on attack2p5 | 14.29% |
| poison5 ep002 token ASR on attack5 | 7.14% |
| poison5 ep002 VaVAM minADE delta vs clean0 ep002 | +0.1299 |
| poison5 full VaVAM minADE delta vs clean0 full | +0.7647 |

The draft intentionally treats token ASR as a proxy diagnostic because it measures exact future-token reconstruction, while the paper attack target is object-level disappearance across future frames.
