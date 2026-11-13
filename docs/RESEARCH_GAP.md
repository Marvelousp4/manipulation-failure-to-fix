# Research gate — 2026-11-13

**Status:** literature-screened hypothesis, **not** a novel contribution yet. Freeze validated PickCube DP (20k EMA); do not train a new model or build a generic recovery benchmark.

## Nearest prior art (read these before coding)

| Existing work | Already does | Our overlap |
|---|---|---|
| [RoboRecover](https://arxiv.org/abs/2609.28952) | Recovery from execution-deviation states; 2k scenarios | Recovery benchmark |
| [FAR (CoRL '26)](https://hoar012.github.io/FAR-Project/) | Failure-aware retry and policy improvement | Retry / recovery |
| [FLARE (CVPR '26)](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html) | Retry / Reset with monitoring | Recovery routing |
| [Recova (NVIDIA)](https://research.nvidia.com/labs/gear/recova/) | Sim/real recovery skills, human handoff, DAgger | Full recovery loop |
| [UPS (RSS '26)](https://jessie-yuan.github.io/ups/) | Act / ask / learn with calibrated uncertainty | Intervention choice |
| [InterveneSim-Value](https://github.com/ethanvillalovoz/intervenesim) | Exact-state counterfactual expert takeover value | Intervention value |
| [MAGMA-GEN](https://arxiv.org/abs/2609.20056) | Counterfactual re-execution validates recovery supervision | Causal verification |
| [Mulligan](https://mulligan.page/) | Targeted initial-state data collection | Failure-targeted data |
| [Data Inoculation](https://arxiv.org/abs/2610.39054) | Human correction under learned data perturbations | Data coverage |
| [VERITAS (RSS '26)](https://veritas-improvement.github.io/) | Visual action verification and self-training | Verification |
| [SCOUT (CoRL '26)](https://liy1shu.github.io/SCOUT/) | Test-time dynamics adaptation | Dynamics correction |
| [FACT (CoRL '26)](https://fact-wam.github.io/) | Failure-data world/action training | Failure prediction |
| [OmniTacTune (CoRL '26)](https://colinyu1.github.io/omnitactune-site/) | Tactile residual adaptation | Contact correction |
| [PACE](https://arxiv.org/abs/2606.00537) | Adaptive action-chunk execution | Execution timing |
| [RAIN](https://arxiv.org/abs/2610.19765) | Adaptive action refinement | Inference correction |
| [Benchmark Audit (CoRL '26)](https://ripl.github.io/manipulation_benchmark_audit/) | Diagnoses benchmark shortcuts/statistical errors | Evaluation validity |
| [RSS '26 rigorous comparison](https://www.roboticsproceedings.org/rss22/p076.html) | Sample-efficient policy comparisons | Statistical evaluation |
| [Beyond Validation Loss](https://arxiv.org/abs/2611.01565) | Policy coverage/reliability before post-training | Model selection |
| [RoboEval Challenge](https://robo-eval.github.io/challenge.html) | Stage-level bimanual diagnostics | Failure metrics |
| [Failure-Aware RL (ICRA '26)](https://ieeexplore.ieee.org/abstract/document/11697475) | Safety critic and recovery policy during online RL | Safe post-training |
| [When Should a Failing Robot Ask?](https://arxiv.org/abs/2609.21942) | Sensor-evidence-aware cost of asking | Diagnosis / intervention |
| [MIND](https://arxiv.org/abs/2611.10400) | Failure detection from policy internal features | Failure attribution |
| [ROEP](https://www.mdpi.com/1424-8220/26/15/4757) | Deployment-facing VLA protocol | Test framework |
| [Set-Supervised DP (RSS '26)](https://www.roboticsproceedings.org/rss22/p080.html) | Learning from corrective action sets | Correction learning |

## Candidate question (unproven gap)

**Given the same physical failure state, which _minimal intervention_ actually restores success, and at what cost?**

Do not conflate failure prediction with intervention benefit. Compare **multiple repair types** and **fault classes**, not just expert-takeover vs continue. InterveneSim already addresses that binary contrast. A useful result would show distinct, reproducible repairability profiles across fault types / policy families, not merely a leaderboard.

## Two-week feasibility gate

1. **Pre-register:** frozen DP; fresh exploration and confirmatory seeds; cost definitions (control steps, policy calls, human/expert labels, new training frames). Do not reuse the prior untouched test for tuning.
2. **Validate replay:** reproduce controller, simulator and contact states; reject any counterfactual fork with unmatched state or divergent no-intervention continuation.
3. **Pilot:** PickCube; one real physical perturbation (post-grasp object displacement), one input fault (stale/noisy observation), one actuator fault (delay). Keep them isolated. Find intermediate—not trivial—failure rates.
4. **Matched repairs:** no intervention, fresh observation / replan (where meaningful), conservative controller correction, and labelled oracle assist. Compare equalized intervention and time budgets. Explicitly label privileged information.
5. **Confirm:** ≥50 fresh paired seeds for informative cases, uncertainty estimates, negative controls, repair overhead. Avoid claims about VLA/vision or sim-to-real from privileged state-only policy.
6. **Extend only if signal survives:** contact-rich second task + a second policy. Otherwise publish upstream tests/negative results and revisit scope.

**Kill / pivot:** no reproducible mid-trajectory replay, no meaningful heterogeneous failures, repair methods perform identically, or closest prior art already covers the result. Stop without inventing an architecture.

**Quality gate:** one held-out falsifiable finding + executable experiment + honest limitations. Algorithm/paper novelty remains unestablished. Public repo needs IP/license review; large artifacts should use Releases/LFS/external storage rather than repeated binary commits.
