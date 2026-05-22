# PRISM

Implementation of **PRISM: Process-Aware Offline Reinforcement Learning with Confidence-Calibrated Self-Correction for Long-Horizon Tool-Using Language Agents**.

PRISM trains a tool-using LLM agent on ALFWorld, WebShop, and ScienceWorld by combining (i) Monte Carlo step-level reward labels, (ii) a process reward model, (iii) a Platt-calibrated trajectory confidence head, (iv) implicit Q-learning critics, (v) advantage-weighted regression over a LoRA-adapted LLaMA-3.1-8B-Instruct policy, and (vi) confidence-gated self-correction at inference time.

The full pipeline is implemented in a single file: `prism.py`.

---

## 1. Requirements

* Python ≥ 3.10
* CUDA-capable GPU (one NVIDIA A100 80GB; smaller GPUs work with reduced `--max_tasks_per_env` and 4-bit loading)
* ~250 GB of disk for model weights, trajectory buffer, and step-label cache

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

Authenticate with the Hugging Face Hub so the LLaMA-3.1-8B-Instruct weights can be pulled:

```bash
huggingface-cli login
```

---

## 2. Datasets

The benchmarks used in the paper are publicly available through Hugging Face and the original repositories. See the **Data availability** section of the paper for the exact links.

The implementation downloads ALFWorld and ScienceWorld task suites automatically through their official packages on first run. WebShop requires the local server set up via the script above (`bash setup.sh` inside the `WebShop` repo).

---

## 3. Running the full pipeline

```bash
python prism.py \
  --base_model_name meta-llama/Llama-3.1-8B-Instruct \
  --encoder_model_name microsoft/deberta-v3-base \
  --benchmarks alfworld webshop scienceworld \
  --output_dir ./prism_outputs \
  --run_sft
```

This executes Stages A → B1 → B2 → B3 → B4 → C (Sec. IV of the paper).

### Stages

| Stage | What it does | Section in paper |
|---|---|---|
| A | Collect trajectories with π₀ and SFT-finetuned π₁ | IV-B |
| B1 | Compute Monte Carlo step labels (M = 8 continuations) | IV-C |
| B2 | Train the PRM (BCE + MSE) and the Platt-calibrated confidence head | IV-C, IV-D |
| B3 | Train IQL value & action-value critics (expectile τ = 0.7) | IV-E |
| B4 | Extract the LoRA-adapted policy with AWR (β = 3.0) | IV-F |
| C | Evaluate with PRM re-scoring and confidence-gated reflection (K = 3, τ(low) = 0.45, τ(abort) = 0.20) | IV-G |

### Useful CLI flags

* `--max_tasks_per_env N` — cap the number of training tasks per benchmark (handy for smoke tests; use a small N like 5)
* `--max_eval_tasks N` — cap the number of evaluation tasks
* `--n_step_labels 60000` — number of prefixes sampled for Monte Carlo labeling 
* `--iql_steps 50000` — number of IQL gradient steps 
* `--prm_epochs 20` — PRM training epochs 
* `--awr_epochs 1` — AWR training epochs 
* `--seed 42` — base random seed (eval seeds are fixed to {1, 7, 13, 23, 42} per paper)
* `--run_sft` — run the one-epoch SFT update for π₁ in Stage A

### Cross-base ablation (Mistral-7B)

```bash
python prism.py --base_model_name mistralai/Mistral-7B-Instruct-v0.3 ...
```

---

## 4. Output artifacts

Written to `--output_dir`:

* `step_labels.json` — Monte Carlo step-label triples
* `results.json` — task-success / score / steps for each benchmark, averaged over the five evaluation seeds
* checkpoints for PRM, confidence head, IQL critics, and the LoRA-adapted policy

---

## 5. Hyperparameter reference 

| Stage | Optimizer | LR | Batch | Horizon | Other |
|---|---|---|---|---|---|
| PRM φ | AdamW | 1.5e-5 | 32 | 20 epochs | λ_MSE = 0.5 |
| Conf ξ | AdamW | 1.0e-5 | 32 | 20 epochs | Platt scaling on 670-traj pool |
| IQL (V, Q) | AdamW | 5.0e-5 | 64 | 50k steps | τ = 0.7, γ = 0.99 |
| AWR π(η) | AdamW | 1.0e-4 | 16 | 1 epoch | β = 3.0, LoRA rank 16, α = 32 |

---

## 6. Reproducing the main table

The numbers in Table III (ALFWorld 76.9%, WebShop 57.8%, ScienceWorld 70.8) are the mean of five evaluation seeds. To reproduce, run the full pipeline once and let evaluation iterate over all five seeds (default behavior).

---

## 7. Repository layout

```
prism.py            single-file end-to-end implementation
requirements.txt    Python dependencies
README.md           this file
```

---

## 8. License

The implementation is released for research use. The base model (LLaMA-3.1-8B-Instruct) and encoder (DeBERTa-v3-base) follow their respective upstream licenses. The benchmarks are distributed by their original authors under their own licenses.
