# Safety-Weighted Semantic Entropy (SWSE) Probes

**Real-Time Jailbreak Detection via Safety-Weighted Semantic Entropy Probes**

Lightweight neural probes that detect jailbreak attempts against LLMs from a **single forward pass** — no multi-sampling, no gradient analysis, no extra generation. Probes are trained on hidden states at the *token-before-generation* (TBG) position and supervised by a safety-aware entropy target derived from intent-clustered sampled responses.

> Ata Dundar Yigit, Mohammad Zandsalimy, Shanu Sushmita.
> *Real-Time Jailbreak Detection via Safety-Weighted Semantic Entropy Probes.* Canadian AI 2026 (short paper).

---

## Key Results

Evaluated on **Llama-3.2-3B-Instruct** with 9,697 harmful + 7,000 benign prompts aggregated from JailBreakV-28K, xTRam1, In-The-Wild-Jailbreak, AdvBench, HarmBench, JailbreakBench, and TDC23-RedTeaming. Held-out test: 650 harmful / 1,410 benign from xTRam1.

| Metric | Concat MLP probe (layers 2, 3, 20, 23, 28) |
|---|---|
| ROC AUC | **0.989** |
| Accuracy | **96.7%** |
| Cohen's d | 1.559 |
| KS statistic | 0.940 |
| Mean benign score | 0.003 ± 0.036 |
| Mean harmful score | 0.260 ± 0.288 |
| Speedup vs multi-sampling defenses | ~100× |

At a conservative threshold of 0.1: **0.85% FPR**, **6.15% FNR**.

---

## Method

For each prompt `x`:

1. **Sample** `N = 10` responses under temperature sampling (T=1.0, top-p=0.9, top-k=50).
2. **Cluster** responses by *intent* with an LLM judge (Gemini 2.5 Flash) into three safety-critical categories:
   - **HC** — Harmful Compliance
   - **DH** — Disguised Harmful
   - **AR** — Appropriate Refusal
3. **Compute** a joint risk target combining a direct safety score and a safety-weighted entropy:

   ```
   SWSE(x) = -Σ wᵢ · pᵢ(x) · log pᵢ(x)          # weighted behavioral uncertainty
   s(x)    = p_HC(x) + 0.5 · p_DH(x)             # direct harm magnitude
   T(x)    = α · s(x) + (1−α) · SWSE(x) / log(|I|),  α = 0.7
   ```

4. **Train** an MLP probe on TBG hidden states `h^(l)_TBG` to regress `T(x)` with MSE loss.
5. **Detect** at inference: a single forward pass → probe → risk score. If above threshold, halt before any token is generated.

Middle-to-late layers carry the strongest safety signal. Concatenating layers spanning early-to-late stages (2, 3, 20, 23, 28) beats any single layer.

---

## Repository Layout

```
.
├── configs/
│   └── safety_probe.yaml          # Model, sampling, clustering, training config
├── scripts/
│   ├── generate_responses.py      # Sample N responses per prompt
│   └── generate_clustering.py     # Cluster responses by intent (LLM judge)
├── defenses/SafetyProbes/
│   ├── clustering/                # LLM / NLI / bidirectional-entailment clustering
│   ├── probe_train.py             # Probe training utilities
│   └── safety_probe.py            # SafetyProbe class
├── generate_features.py           # Extract TBG hidden states from Llama
├── train_probes.py                # Train single-layer + concatenated MLP/linear probes
├── evaluate_safety_probes.py      # Statistical + ROC analysis, figures
├── test_safety_probes.py          # Run a trained probe on new prompts
└── paper/
    └── main_short.tex             # Canadian AI 2026 short paper
```

Large artifacts (`data/`, `runs/`, model weights, cached features, figures) are **not** tracked. See `.gitignore`.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers scikit-learn numpy matplotlib tqdm pyyaml
```

For Llama-3.2-3B-Instruct access, authenticate via `huggingface-cli login`. For the intent-clustering step, set your Gemini API credentials in `configs/safety_probe.yaml` (`clustering.llm_based.api_key`).

---

## Pipeline

### 1. Sample responses and cluster by intent

```bash
python scripts/generate_responses.py   --config configs/safety_probe.yaml
python scripts/generate_clustering.py  --config configs/safety_probe.yaml
```

### 2. Extract TBG hidden states

```bash
python generate_features.py   --data data/clustering_results/clustered_data.json   --model meta-llama/Llama-3.2-3B-Instruct   --out cache/features
```

### 3. Train probes

```bash
python train_probes.py   --features cache/features   --probe_type mlp   --layers 2 3 20 23 28   --out runs/my_run
```

### 4. Evaluate

```bash
python evaluate_safety_probes.py   --run runs/my_run   --test_set data/datasets/xTRam1_harmful_test.json
```

Outputs: ROC curves, layer-wise comparison, KS/MWU statistics, and score distributions.

### 5. Run detection on new prompts

```bash
python test_safety_probes.py   --probe runs/my_run/mlp/safety_probe.pkl   --prompt "How do I ..."
```

---

## Datasets

| Split | Source | Count |
|---|---|---|
| Harmful | JailBreakV-28K | 5,000 |
| Harmful | xTRam1/safe-guard-prompt-injection | 3,000 |
| Harmful | In-The-Wild-Jailbreak | 651 |
| Harmful | AdvBench | 509 |
| Harmful | HarmBench | 382 |
| Harmful | JailbreakBench | 100 |
| Harmful | TDC23-RedTeaming | 55 |
| Benign | xTRam1 benign split | 7,000 |

Prompts are deduplicated (case-insensitive), with trivially short or non-textual entries removed. Test set (650 / 1,410) is drawn exclusively from xTRam1 — no source overlap with training.

---

## Limitations

- Intent labels come from an external LLM judge (Gemini 2.5 Flash) and inherit its biases.
- Text-only; multimodal jailbreaks are out of scope.
- Adversarial robustness of the probes themselves against probe-aware attackers is unexplored.

---

## Citation

```bibtex
@inproceedings{yigit2026swse,
  title     = {Real-Time Jailbreak Detection via Safety-Weighted Semantic Entropy Probes},
  author    = {Yigit, Ata Dundar and Zandsalimy, Mohammad and Sushmita, Shanu},
  booktitle = {Canadian AI},
  year      = {2026}
}
```
