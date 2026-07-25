# PRISM

Implementation of **PRISM: Process-Aware Offline Reinforcement Learning with Confidence-Calibrated Self-Correction for Long-Horizon Tool-Using Language Agents**.

PRISM trains a tool-using LLM agent on ALFWorld, WebShop and ScienceWorld by combining (i) Monte Carlo step-level reward labels, (ii) a process reward model, (iii) a Platt-calibrated trajectory confidence head, (iv) implicit Q-learning critics, (v) advantage-weighted regression over a LoRA-adapted LLaMA-3.1-8B-Instruct policy, and (vi) confidence-gated self-correction at inference time.

The full pipeline is implemented in a single file: `prism.py`.


---

## 1. Requirements

* Python >= 3.10
* CUDA-capable GPU (one NVIDIA A100 80GB; smaller GPUs work with reduced `--max_tasks_per_env` and 4-bit loading)
* ~250 GB of disk for model weights, trajectory buffer and step-label cache

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install the three benchmark environments:

```bash
pip install alfworld scienceworld
alfworld-download
git clone https://github.com/princeton-nlp/WebShop && cd WebShop && bash setup.sh
```

Add the WebShop repository root to `PYTHONPATH` and make sure `ALFWORLD_DATA` points at the downloaded ALFWorld task suite.

Authenticate with the Hugging Face Hub so the LLaMA-3.1-8B-Instruct weights can be pulled:

```bash
huggingface-cli login
```

---

## 2. Datasets

The benchmarks used in the paper are publicly available through Hugging Face and the original repositories. See the **Data availability** section of the paper for the exact links.

ALFWorld and ScienceWorld task suites are downloaded automatically through their official packages on first run. WebShop requires the local server set up via the script above (`bash setup.sh` inside the `WebShop` repo).

Split sizes follow Sec. V-A of the paper:

| Benchmark | Training tasks | Evaluation tasks | Evaluation split | Step budget |
|---|---|---|---|---|
| ALFWorld | 3,553 | 134 | unseen (`eval_out_of_distribution`) | 30 |
| WebShop | 1,000 | 500 | canonical test instructions | 25 |
| ScienceWorld | 1,371 variants | 270 variants | official test variations | 30 |

WebShop training sessions are offset to 500–1499 so that they are disjoint from the 0–499 sessions used for evaluation. ScienceWorld train and test variants come from `getVariationsTrain()` and `getVariationsTest()` and never overlap.

---

## 3. Running the full pipeline

```bash
python prism.py \
  --base_model_name meta-llama/Llama-3.1-8B-Instruct \
  --encoder_model_name microsoft/deberta-v3-base \
  --benchmarks alfworld webshop scienceworld \
  --output_dir ./prism_outputs \
  --run_sft \
  --search_thresholds
```

This executes Stages A -> B1 -> B2 -> B3 -> B4 -> C (Sec. IV of the paper).

### Stages

| Stage | What it does | Section in paper |
|---|---|---|
| A | Hold out the calibration tasks, then collect trajectories with pi(0) (Kcollect = 4) and SFT-finetuned pi_1 (K_collect/2 = 2) | IV-B, V-F |
| B1 | Compute Monte Carlo step labels (M = 8 continuations from pi_1) | IV-C |
| B2 | Train the PRM (BCE + MSE) and the confidence head, fit per-benchmark Platt scalars, grid-search the thresholds | IV-C, IV-D, V-F |
| B3 | Train IQL value and action-value critics (expectile tau = 0.7, gamma = 0.99) | IV-E |
| B4 | Extract the LoRA-adapted policy with AWR (beta = 3.0) | IV-F |
| C | Evaluate with PRM re-scoring and confidence-gated reflection (K = 3, tau_low = 0.45, tau_abort = 0.20, r_max = 2) | IV-G |

The calibration pool is collected **before** any component is trained and is excluded from Monte Carlo labeling, PRM training, confidence-head training, the IQL buffer and the AWR update (Sec. V-F). It comprises 670 ALFWorld trajectories from 335 held-out tasks, 500 WebShop trajectories from 250 tasks and 500 ScienceWorld trajectories from 250 tasks: 1,670 trajectories in total, two rollouts per task.

pi_0, pi_1 and pi_eta are three states of one backbone: pi_0 is the base model with all adapters disabled, pi_1 is the `sft` LoRA adapter, and pi_eta is a **separate** `awr` adapter initialized from the base model rather than from pi_1, matching the `pi_eta <- pi_0` input of Algorithm 1.

### Useful CLI flags

* `--max_tasks_per_env N` — cap the number of training tasks per benchmark (handy for smoke tests; use a small N like 5)
* `--max_calib_tasks_per_env N` — cap the number of calibration tasks per benchmark
* `--max_eval_tasks N` — cap the number of evaluation tasks
* `--run_sft` — run the one-epoch SFT update for pi_1 in Stage A (required to reproduce the paper)
* `--search_thresholds` — grid-search tau_low and tau_abort on the calibration pool (Sec. V-F); without it the defaults 0.45 / 0.20 are used directly
* `--reuse_buffer` — reload `trajectory_buffer.json` and `calibration_pool.json` instead of re-collecting Stage A
* `--step_labels_path PATH` — read/write the Stage B1 label cache at an explicit path so it is shared across seeds and ablations
* `--recompute_step_labels` — force Stage B1 to re-run even when a cache exists
* `--n_step_labels 60000` — number of prefixes sampled for Monte Carlo labeling
* `--m_continuations 8` — continuations per prefix
* `--label_noise_eps 0.0` — symmetric step-label corruption rate for the robustness study
* `--iql_steps 50000` — number of IQL gradient steps
* `--awr_steps 50000` — number of AWR gradient steps
* `--prm_epochs 20` / `--conf_epochs 20` — Stage B2 epoch budgets (early stopping patience 3)
* `--awr_beta`, `--iql_expectile_tau`, `--alpha_reward_mix`, `--lambda_mse`, `--k_candidates`, `--tau_low`, `--tau_abort`, `--r_max` — the swept hyperparameters
* `--seed 42` — base random seed
* `--seeds_eval 1 7 13 23 42` — evaluation seeds (paper default)

### Cross-base ablation (Mistral-7B)

```bash
python prism.py --base_model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --output_dir ./prism_outputs_mistral --run_sft --search_thresholds
```

---

## 4. Output artifacts

Written to `--output_dir`:

| File | Contents |
|---|---|
| `config.json` | the fully resolved configuration for the run |
| `task_split.json` | training and calibration task ids per benchmark |
| `trajectory_buffer.json`, `calibration_pool.json` | Stage A trajectories (disjoint by task) |
| `buffer_composition.json` | tasks, trajectories, success rate and mean length per benchmark (Table II) |
| `step_labels.json` | Monte Carlo step-label triples (goal, history, action, `q_hat`, `y_label`) |
| `stage_b2_report.json` | PRM and confidence training curves, plus ECE and Brier before/after Platt scaling per benchmark |
| `prm.pt`, `confidence_head.pt` | PRM weights; confidence weights with the fitted Platt scalars |
| `stage_b3_report.json`, `iql_critics.pt` | IQL loss curves and the V/Q heads |
| `stage_b4_report.json`, `policy_adapter/` | AWR loss curve and the LoRA-adapted policy |
| `thresholds.json` | selected tau_low and tau_abort with the full grid |
| `results.json` | per-seed and aggregated task success, mean reward, steps per successful task, decoded tokens per task, reflection rate and abort rate, plus per-task success for paired testing |

`results.json` reports the primary metric per benchmark: success rate for ALFWorld and WebShop, mean reward in [0, 100] for ScienceWorld. `per_task_success` holds the per-task success fraction over the five seeds, which is the paired unit used for the t-test and bootstrap in Sec. VI-A.

---

## 5. Hyperparameter reference (Table IV)

| Stage | Optimizer | LR | Batch | Horizon | Weight decay | Warm-up | Method-specific |
|---|---|---|---|---|---|---|---|
| PRM phi | AdamW | 1.5e-5 | 32 | 20 epochs | 0.01 | 10% | lambda_BCE = 1.0, lambda_MSE = 0.5 |
| Conf xi | AdamW | 1.0e-5 | 32 | 20 epochs | 0.01 | 10% | per-benchmark Platt (a, b) |
| IQL (V, Q) | AdamW | 5.0e-5 | 64 | 50k steps | 0.01 | 10% | tau = 0.7, gamma = 0.99, Polyak 0.005 |
| AWR pi_eta | AdamW | 1.0e-4 | 16 | 50k steps (3.0 epochs) | 0.0 | 5% | beta = 3.0, Delta_max = 4, negative-advantage factor 0.1 |

Shared settings (Sec. V-D): AdamW betas (0.9, 0.95), gradient clipping 1.0, linear warm-up then linear decay, bfloat16, LoRA rank 16 with alpha 32 and dropout 0.05 on the query, key, value and output projections of every block. Decoding uses temperature 0.7, top-p 0.95 and at most 256 new tokens per action.

Other quantities fixed by the paper: K_collect = 4 and K_collect/2 = 2 (Sec. IV-B); N_step = 60,000 prefixes split 54,000 train / 6,000 validation with early stopping patience 3 (Sec. IV-C); M = 8 continuations; alpha = 0.7 for the behavioral reward of Eq. (13); the encoder truncates prefixes head-tail, always keeping the task goal in full and never truncating the action being scored (Sec. IV-C).

---

## 6. Reproducing the main table

The numbers in **Table V** (ALFWorld 76.9%, WebShop 57.8%, ScienceWorld 70.8) are the mean over the five evaluation seeds {1, 7, 13, 23, 42}. Run the full pipeline once; evaluation iterates over all five seeds by default.

Stage B1 is the expensive stage: 60,000 prefixes x 8 continuations x 4.1 mean remaining steps is approximately 2.0 million environment interactions and roughly 103 GPU-hours. It is computed once and cached, so pass the same `--step_labels_path` to every subsequent seed, ablation and sweep to reuse it at zero marginal cost (Sec. IV-J).

```bash
python prism.py --run_sft --search_thresholds \
  --step_labels_path ./cache/step_labels.json \
  --output_dir ./prism_outputs
```

### Sensitivity analysis (Table VII)

```bash
for BETA in 1.0 3.0 10.0; do
  python prism.py --awr_beta $BETA --step_labels_path ./cache/step_labels.json \
    --reuse_buffer --output_dir ./sweeps/beta_$BETA --run_sft
done

for TAU in 0.5 0.7 0.9; do
  python prism.py --iql_expectile_tau $TAU --step_labels_path ./cache/step_labels.json \
    --reuse_buffer --output_dir ./sweeps/tau_$TAU --run_sft
done

for K in 1 3 5; do
  python prism.py --k_candidates $K --step_labels_path ./cache/step_labels.json \
    --reuse_buffer --output_dir ./sweeps/k_$K --run_sft
done

for TL in 0.35 0.45 0.55; do
  python prism.py --tau_low $TL --step_labels_path ./cache/step_labels.json \
    --reuse_buffer --output_dir ./sweeps/tau_low_$TL --run_sft
done
```

### Robustness to rollout count and label noise (Table IX)

`M` changes the labeling budget and therefore requires recomputing Stage B1; the corruption rate `epsilon` is applied to cached labels and does not.

```bash
for M in 4 8 16; do
  python prism.py --m_continuations $M --recompute_step_labels \
    --step_labels_path ./cache/step_labels_m$M.json \
    --reuse_buffer --output_dir ./sweeps/m_$M --run_sft
done

for EPS in 0.10 0.20; do
  python prism.py --label_noise_eps $EPS --step_labels_path ./cache/step_labels.json \
    --reuse_buffer --output_dir ./sweeps/eps_$EPS --run_sft
done
```

### Ablations (Table VI)

* *w/o self-correction loop*: `--k_candidates 1 --r_max 0`
* *w/o PRM re-scoring*: `--k_candidates 1` with reflection retained (`--r_max 2`)
* *w/o offline RL*: skip Stages B3 and B4 and evaluate the PRM as a best-of-K verifier over the SFT adapter
* *w/o Platt scaling*: delete the `platt` entry from `confidence_head.pt` before evaluation, so the raw xi outputs are used
* *outcome reward only*: `--alpha_reward_mix 0.0`

---

## 7. Repository layout

```
prism.py            single-file end-to-end implementation
requirements.txt    Python dependencies
README_PRISM.md     this file
```

---

## 8. License

The implementation is released for research use. The base model (LLaMA-3.1-8B-Instruct) and encoder (DeBERTa-v3-base) follow their respective upstream licenses. 
