# Experiment configs

Representative configurations of the paper's runs. Each file holds only the values that differ
from `configs/default.yaml`, which it is merged onto, and is one complete run from the base
model. Its header comment gives the run's role, the effective batch and learning-rate schedule,
the node count used for the paper, and the launch command.

`nemotron12b-legal/` is one complete setting (Nemotron-Nano-12B-v2, legal adaptation):

| File | Run |
|---|---|
| `specialist.yaml` | no-replay CPT; its final checkpoint is the adaptation reference θ<sub>ref</sub><sup>A</sup> |
| `precompute.yaml` | reference losses: θ<sub>ref</sub><sup>A</sup> on adaptation blocks, the pretrained model θ<sub>0</sub> on replay blocks |
| `rod.yaml` | RoD (m = 2, k = 1024) |
| `fixed-replay-10.yaml` | fixed-replay CPT with ~10% replay (`data_mixer.replay_share: 0.111`), same budget as `rod.yaml` |
| `ablation-fixed-share-rho.yaml` | selection ablation with a fixed replay share (`rod.selection_strategy: fixed_ratio_rho`) |
| `merge-eval.yaml` | validation of a merged model θ<sub>merge</sub> = (1 − λ)θ<sub>0</sub> + λθ<sub>A</sub> (`python -m rod.utils.merge_models`) |

`qwen35-35b-german/curriculum-from-4b.yaml` shows curriculum transfer: Qwen3.5-35B-A3B trained,
without selection, on the sample stream RoD selected for Qwen3.5-4B (`rod.curriculum_trace_dirs`),
using the corpus of that 4B run.

## Learning-rate schedule and scheduler units

All training runs warm up, hold a constant learning rate, and end with a cosine decay (WSD).
NeMo-RL converts the scheduler's iteration counts (`lr_warmup_iters`, `lr_decay_iters`,
`lr_wsd_decay_iters`) to samples with `policy.train_global_batch_size`, but each step advances
the scheduler by the number of sequences actually trained. With selection enabled, the global
batch is the candidate pool `m·k` while `k` sequences are trained, so the configs give these
counts divided by `m`. For example, `rod.yaml` has `lr_decay_iters: 1479` for 2958 steps. The
headers list the resulting schedule in optimizer steps.

## Hardware

The paper runs used 1–4 nodes of 8 GPUs (`cluster.num_nodes`). The provided launch scripts
start a single-node Ray cluster and override `cluster.num_nodes=1`. Model parallelism comes
from the `ROD_TP/PP/EP` environment variables (`slurm/lib/container_env.sh`) unless a config
sets it (`curriculum-from-4b.yaml` sets expert parallelism 8 and pipeline parallelism 1).

## All paper runs

The runs behind the paper's results, per setting, with the values that distinguish them (all
use 4096-token blocks, Adam with weight decay 0.1 and a 1.2e-4 peak learning rate; see
`configs/default.yaml`). Runs published as files above are marked with their file name. Steps
are optimizer steps; "Pool / k" is the candidate pool and trained batch for selection runs, and
the batch size otherwise. In the paper, most runs were trained as a constant-learning-rate job
followed by an anneal job started from its checkpoint; the table gives the combined schedule.

Run names follow the paper: `specialist` no-replay CPT (the adaptation reference);
`fixed-replay-<r>` fixed-replay CPT with about `r`% replay (the retained replay fractions 0.111,
0.25 and 1.0 give 10/20/50% on Legal and 17/32/65% on German); `rod` RoD with m = 2. These
fixed-replay and RoD runs share the setting's compute-matched trained-token budget (the
specialist's is within a few percent of it). `max-replay` is the reference
trained on all available data (about 2× the budget); `-long` marks fixed-replay arms trained
beyond the budget; `rod-m<m>` are the candidate-multiplier runs; `ablation-*` a fixed replay
share of the `k` slots with uniform sampling (`fixed_ratio`), RHO within the replay corpus
(`replay_only_rho`) or RHO within both (`fixed_ratio_rho`); `rod-4b-reference` RoD with the
Qwen3.5-4B adaptation reference; `curriculum-from-*` curriculum transfer; `merge-eval`
validation of the merged models for λ ∈ {0.2, 0.4, 0.6, 0.8}.

<!-- RUNS:BEGIN -->
<details><summary><b>Nemotron-Nano-12B-v2 (base), legal</b> (<code>nemotron12b-legal</code>, 16 runs)</summary>

| Run | Steps | Pool / k | Selection | Learning rate | replay_share | Val per origin | Data |
|---|---|---|---|---|---|---|---|
| `specialist` → **`specialist.yaml`** | 2898 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 580 | 0 | 200 |  |
| `fixed-replay-10-long` | 3220 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 644 | 0.111 | 200 |  |
| `fixed-replay-20-long` | 3620 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 724 | 0.25 | 200 |  |
| `max-replay` | 5795 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 1159 | 1 | 200 |  |
| `fixed-replay-10` → **`fixed-replay-10.yaml`** | 2958 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 615 | 0.111 | 200 |  |
| `fixed-replay-20` | 2958 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 615 | 0.25 | 200 |  |
| `fixed-replay-50` | 2958 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 615 | 1 | 200 |  |
| `rod-m1.5` | 3538 | 1536 / 1024 (m=1.5) | top_k | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 500 |  |
| `rod-m2` | 3538 | 2048 / 1024 (m=2) | top_k | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 500 |  |
| `rod-m4` | 3840 | 4096 / 1024 (m=4) | top_k | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 500 |  |
| `rod-m8` | 3538 | 8192 / 1024 (m=8) | top_k | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 500 |  |
| `rod` → **`rod.yaml`** | 2958 | 2048 / 1024 (m=2) | top_k | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 500 |  |
| `ablation-fixed-share` | 3538 | 4096 / 1024 (m=4) | fixed_ratio (0.072) | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 500 |  |
| `ablation-fixed-share-replay-rho` | 3538 | 4096 / 1024 (m=4) | replay_only_rho (0.072) | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 500 |  |
| `ablation-fixed-share-rho` → **`ablation-fixed-share-rho.yaml`** | 3538 | 4096 / 1024 (m=4) | fixed_ratio_rho (0.072) | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 500 |  |
| `merge-eval` → **`merge-eval.yaml`** | 0 (val only) | 1024 | – | – | 0 | 200 | merged model |

</details>

<details><summary><b>Nemotron-Nano-12B-v2 (base), German</b> (<code>nemotron12b-german</code>, 10 runs)</summary>

| Run | Steps | Pool / k | Selection | Learning rate | replay_share | Val per origin | Data |
|---|---|---|---|---|---|---|---|
| `specialist` | 2980 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 591 | 0 | 200 |  |
| `fixed-replay-17-long` | 3285 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 657 | 0.111 | 200 |  |
| `fixed-replay-32-long` | 3695 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 739 | 0.25 | 200 |  |
| `max-replay` | 5913 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 1183 | 1 | 200 |  |
| `fixed-replay-17` | 2980 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 591 | 0.111 | 200 |  |
| `fixed-replay-32` | 2980 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 591 | 0.25 | 200 |  |
| `fixed-replay-65` | 2980 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 591 | 1 | 200 |  |
| `rod-m2` | 3538 | 2048 / 1024 (m=2) | top_k | 1.2e-4 → 1.2e-5, cosine over last 616 | 0.533 | 500 |  |
| `rod` | 2980 | 2048 / 1024 (m=2) | top_k | 1.2e-4 → 1.2e-5, cosine over last 592 | 0.533 | 500 |  |
| `merge-eval` | 0 (val only) | 1024 | – | – | 1 | 500 | merged model |

</details>

<details><summary><b>Qwen3.5-9B (base), legal</b> (<code>qwen35-9b-legal</code>, 7 runs)</summary>

| Run | Steps | Pool / k | Selection | Learning rate | replay_share | Val per origin | Data |
|---|---|---|---|---|---|---|---|
| `specialist` | 2824 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 549 | 0 | 200 |  |
| `max-replay` | 5735 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 1147 | 1 | 200 |  |
| `fixed-replay-10` | 2824 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 505 | 0.111 | 200 |  |
| `fixed-replay-20` | 2824 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 505 | 0.25 | 200 |  |
| `fixed-replay-50` | 2824 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 299 | 1 | 200 |  |
| `rod` | 2824 | 2048 / 1024 (m=2) | top_k | 1.2e-4 → 1.2e-5, cosine over last 550 | 1 | 500 |  |
| `merge-eval` | 0 (val only) | 1024 | – | – | 1 | 500 | merged model |

</details>

<details><summary><b>Qwen3.5-4B (base), German</b> (<code>qwen35-4b-german</code>, 9 runs)</summary>

| Run | Steps | Pool / k | Selection | Learning rate | replay_share | Val per origin | Data |
|---|---|---|---|---|---|---|---|
| `specialist` | 2862 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 587 | 0 | 200 |  |
| `fixed-replay-17-long` | 3551 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 710 | 0.111 | 200 |  |
| `fixed-replay-17` | 2862 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 587 | 0.111 | 200 |  |
| `fixed-replay-32-long` | 4323 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 865 | 0.25 | 200 |  |
| `fixed-replay-32` | 2862 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 587 | 0.25 | 200 |  |
| `max-replay` | 8488 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 1698 | 1 | 200 |  |
| `fixed-replay-65` | 2862 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 587 | 1 | 200 |  |
| `rod` | 2862 | 2048 / 1024 (m=2) | top_k | 1.2e-4 → 1.2e-5, cosine over last 588 | 0.528 | 500 |  |
| `merge-eval` | 0 (val only) | 1024 | – | – | 1 | 500 | merged model |

</details>

<details><summary><b>Qwen3.5-9B (base), German</b> (<code>qwen35-9b-german</code>, 11 runs)</summary>

| Run | Steps | Pool / k | Selection | Learning rate | replay_share | Val per origin | Data |
|---|---|---|---|---|---|---|---|
| `specialist` | 2935 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 587 | 0 | 200 |  |
| `fixed-replay-17-long` | 3261 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 652 | 0.111 | 200 |  |
| `fixed-replay-32-long` | 3668 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 734 | 0.25 | 200 |  |
| `max-replay` | 5869 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 1174 | 1 | 200 |  |
| `fixed-replay-17` | 2960 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 587 | 0.111 | 200 |  |
| `fixed-replay-32` | 2960 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 587 | 0.25 | 200 |  |
| `fixed-replay-65` | 2960 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 587 | 1 | 200 |  |
| `rod` | 2960 | 2048 / 1024 (m=2) | top_k | 1.2e-4 → 1.2e-5, cosine over last 588 | 0.528 | 500 |  |
| `rod-4b-reference` | 2960 | 2048 / 1024 (m=2) | top_k | 1.2e-4 → 1.2e-5, cosine over last 588 | 0.528 | 500 |  |
| `curriculum-from-4b` | 2862 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 588 | 1 | 200 | stream of `qwen35-4b-german/rod` |
| `merge-eval` | 0 (val only) | 1024 | – | – | 1 | 500 | merged model |

</details>

<details><summary><b>Nemotron-3-Nano-30B-A3B (base), German</b> (<code>nemotron30b-german</code>, 3 runs)</summary>

| Run | Steps | Pool / k | Selection | Learning rate | replay_share | Val per origin | Data |
|---|---|---|---|---|---|---|---|
| `specialist` | 3538 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 616 | 0 | 200 |  |
| `fixed-replay-32` | 3538 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 640 | 0.25 | 200 |  |
| `curriculum-from-12b` | 3538 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 616 | 1 | 200 | stream of `nemotron12b-german/rod-m2` |

</details>

<details><summary><b>Qwen3.5-35B-A3B (base), German</b> (<code>qwen35-35b-german</code>, 3 runs)</summary>

| Run | Steps | Pool / k | Selection | Learning rate | replay_share | Val per origin | Data |
|---|---|---|---|---|---|---|---|
| `specialist` | 2862 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 612 | 0 | 200 |  |
| `fixed-replay-32` | 2862 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 612 | 0.25 | 200 |  |
| `curriculum-from-4b` → **`curriculum-from-4b.yaml`** | 2862 | 1024 | – | 1.2e-4 → 1.2e-5, cosine over last 612 | 1 | 200 | stream of `qwen35-4b-german/rod` |

</details>
<!-- RUNS:END -->
