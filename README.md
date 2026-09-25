# CS4.406 IRE — Assignment 2: Learning from Click-Logs on MIND and EB-NeRD

Two-stage retrieve-then-rank news recommendation with behavioural features, on the
**MIND** (English, Microsoft) and **EB-NeRD** (Danish, Ekstra Bladet) datasets.

**Team**
- Shivam Patel — 2025202030
- Suyash Pande — 2025201063

## Overview

Building on Assignment 1's lexical/semantic retrieval, we add behavioural signals from
click-logs and a learned re-ranker. Assignment 1's candidate generator supplies top-K
candidates; a re-ranker scores them over engineered behavioural features (click-history,
recency-decay, popularity, CTR, freshness, category, session). All features are computed
point-in-time (strictly-before-T) under an enforced behaviour-window boundary. Both datasets
are trained on the large bundles; extended evaluation is on labelled dev; final predictions
are submitted to both Codabench leaderboards.

**Work split:** Shivam ran the MIND pipeline end-to-end; Suyash ran the EB-NeRD pipeline
end-to-end. Design, feature taxonomy, and evaluation protocol are shared.

## Repository structure

```
.
├── IRE_A2_MIND_notebooks/        # MIND pipeline (Shivam) — Kaggle notebooks
├── IRE A2_EBNERD_IPYNB_FILES/    # EB-NeRD pipeline (Suyash) — Kaggle notebooks + EB-NeRD design note
├── IRE_ASS2.pdf                  # Combined design note (6 pages, both datasets)
└── README.md
```

### MIND notebooks (`IRE_A2_MIND_notebooks/`)

| Notebook | Purpose |
|----------|---------|
| A2_01_features_mind | Q1: 14 behavioural features, leakage test, feature npz |
| A2_02_reranker_gbdt_mind | Q2: LightGBM + XGBoost, before/after, extended eval |
| A2_03_reranker_neural_mind / A2_03_finalize_mind | Q2: NRMS (title) + MLP |
| A2_03c/03d/03e_nrms_* | Q3 improvement + ablations (title+abstract; category-null; recency-regression) |
| A2_03f_nrms_hybrid_mind | Q3 experiment: hybrid NRMS (text + features) |
| A2_04_baseline_beat_ci_mind | Q3.4: paired bootstrap 95% CIs |
| A2_slice_comparison_mind | Q5: per-reranker × cold/warm/head/tail slices |
| A2_retrieve_rerank_mind | Q2 two-stage: FAISS ANN retrieve → rerank |
| A2_Q4_serving_scale_mind | Q4: serving latency, cost/QPS, 10× analysis |
| A2_submission_mind | Q5: LightGBM → 2.37M test → Codabench prediction |

### EB-NeRD notebooks (`IRE A2_EBNERD_IPYNB_FILES/`)

Suyash's EB-NeRD pipeline, developed as a sequence of iterations building up to the final
CatBoost re-ranker. All run top-to-bottom on Kaggle with the `ebnerd_large` dataset attached;
EB-NeRD supplies pre-computed 768-d contrastive article embeddings (no GPU required for
feature construction).

| Notebook | Purpose |
|----------|---------|
| `ire-a1-ebnerd (4).ipynb` | Assignment 1 EB-NeRD reference pipeline (lexical/semantic + freshness/popularity blend) — the A1 baseline the A2 re-ranker is measured against |
| `ire-a2-ebnerd.ipynb` | A2 iteration 1: unified schema, behavioural + semantic/BM25 history features, temporal protocol, first re-ranking + evaluation |
| `ire-a2-ebnerd-run2.ipynb` | A2 iteration 2: feature/blend refinement, staleness-matched (STALE) evaluation, ablations |
| `ire-a2-ebnerd-run3-pretrain-embedding.ipynb` | A2 iteration 3: NRMS-DocVec neural ranker using the frozen pretrained contrastive vectors |
| `ire-a2-ebnerd-run4-catboost.ipynb` | A2 final: CatBoost PairLogit re-ranker, paired-bootstrap CIs, history/pageviews masking ablations, beyond-accuracy + slices, and the full 13.5M-impression test prediction + submission |

The folder also contains the EB-NeRD A2 design note (`EBNeRD_A2_Design_Note.pdf/.docx`) and
the A1 design note (`2025201063_A1_Design_Note.pdf`).

**Key EB-NeRD features:** A1 hybrid score, log-pageviews/inviews, 24h-half-life decayed CTR
(Bayesian smoothing), 10-bucket article age, semantic/BM25 history, category/subcategory
affinity, sentiment, position, causal session depth. **Final re-ranker:** CatBoost PairLogit
(700 iterations, depth 8).

## Datasets

| | MIND | EB-NeRD |
|--|------|---------|
| Articles | ~104K | 125,541 |
| Splits | MINDlarge (dev labelled; test 2.37M unlabelled) | 12.06M / 12.57M / 13.54M impressions (one week each) |
| Language | English | Danish |

Splits are temporal (never random). Large data files, embeddings, model checkpoints, and
prediction ZIPs are **not** committed (see `.gitignore`); model/feature artifacts are hosted
on Hugging Face and pulled by the notebooks.

## Models

- **GBDT re-rankers:** LightGBM (LambdaMART), XGBoost (pairwise) — MIND; CatBoost (PairLogit) — EB-NeRD
- **Neural re-rankers:** NRMS (multi-head self-attention + additive attention), MLP — MIND; NRMS-DocVec — EB-NeRD
- **Retrieval:** BM25 (own inverted index) + dense embeddings (MiniLM / contrastive) via FAISS (Flat exact vs HNSW approximate)

## Results (headline)

| Dataset | Baseline → Improved (dev AUC) | Leaderboard |
|---------|-------------------------------|-------------|
| MIND | NRMS 0.585 → LightGBM 0.710 (paired CI +0.121, excludes 0) | test AUC **0.6581** (Codabench 13967) |
| EB-NeRD | A1 0.681 → CatBoost 0.713 (paired CI +0.032, excludes 0) | 0.6817 scored; A2 CatBoost submitted (comp. 2469) |

Full metrics, ablations (with paired bootstrap 95% CIs), slice analysis, serving/scale
measurements, and the 10× scaling discussion are in `IRE_ASS2.pdf`.

## Reproducing

Each notebook is self-contained and runs on Kaggle with the relevant dataset mounted.
Feature matrices, trained models, and dev-score caches are pulled from a public Hugging Face
dataset (downloads need no token); uploads require an `HF_TOKEN` Kaggle secret. Paths are
hardcoded per notebook; see each folder for run order (MIND: 01 → 02 → 03 → 04 → slice →
retrieve → Q4 → submission).

## Anti-gaming & correctness (Q9)

- Behaviour-window boundary enforced; executable leakage test (MIND: 0 future-click violations;
  EB-NeRD: split-history separation assertion).
- Metrics reported with and without serving-unavailable features (e.g. EB-NeRD pageviews
  masking diagnostic).
- Metrics computed within-impression; official scorer agreement verified.

## References

1. Wu et al., *MIND: A Large-Scale Dataset for News Recommendation*, ACL 2020.
2. Kruse et al., *EB-NeRD: A Large-Scale Dataset for News Recommendation*, RecSys Challenge 2024.
3. Wu et al., *Neural News Recommendation with Multi-Head Self-Attention (NRMS)*, EMNLP-IJCNLP 2019.
