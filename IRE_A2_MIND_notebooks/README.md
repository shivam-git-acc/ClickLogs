# Assignment 2 — MIND Notebooks (Kaggle)

Team: Shivam Patel (2025202030) & Suyash Pande (2025201063)
CS4.406 Information Retrieval & Extraction — Assignment 2

These are the MIND-side A2 notebooks (Shivam). Each is self-contained; they load/save
artifacts (feature matrices, models, dev-score caches) to a public HF dataset
(donbosoc/mind-artifacts): downloads need no token, uploads use an HF_TOKEN Kaggle secret.
All need the MINDlarge dataset mounted. Recommended run order:

| # | Notebook | Purpose | Output |
|---|----------|---------|--------|
| 01 | A2_01_features_mind | Q1: build 14 behavioural features (600k train + 40k dev), leakage test, save npz (+iids) | mind_large_features.npz |
| 02 | A2_02_reranker_gbdt_mind | Q2: LightGBM + XGBoost rerankers, before/after, extended eval, save models | lgbm/xgb models |
| 03 | A2_03_reranker_neural_mind | Q2 Option B: NRMS (title) + MLP training (full run) | nrms/mlp models |
| 03-finalize | A2_03_finalize_mind | Load best NRMS + MLP, final eval + dev-score cache | nrms_devscores.pkl, mlp |
| 03b | A2_03b_resume_nrms_mind | (utility) resume NRMS training from checkpoint (lr fix) | — |
| 03c | A2_03c_nrms_titleabstract_mind | Q3 Improvement 2: NRMS title+abstract (800k/2ep) | nrms_ta model + cache |
| 03d | A2_03d_nrms_category_mind | Q3 ablation: NRMS +category (null result) | nrms_cat model + cache |
| 03e | A2_03e_nrms_freshness_mind | Q3 ablation: NRMS +recency (regression) | nrms_fresh model + cache |
| 03f | A2_03f_nrms_hybrid_mind | Q3 experiment: hybrid NRMS (text + features) | nrms_hybrid model + cache |
| 04 | A2_04_baseline_beat_ci_mind | Q3.4: paired bootstrap 95% CIs (baseline->beat) | CI tables |
| slice | A2_slice_comparison_mind | Q5: per-reranker x cold/warm/head/tail slice table | slice tables |
| retrieve | A2_retrieve_rerank_mind | Q2 two-stage: FAISS ANN retrieve -> 3 rerankers + extended eval | recall@K, cond MRR, latency |
| Q4 | A2_Q4_serving_scale_mind | Q4: serving latency, cost/QPS, 10x scaling analysis | latency/cost tables |
| submission | A2_submission_mind | Q5: LightGBM on 2.37M test -> Codabench prediction.zip | prediction.zip (LB 0.6581) |

## Notes
- Core pipeline: 01 -> 02 -> 03/03-finalize -> 03c/03d/03e (ablations) -> 04 -> slice -> retrieve -> Q4 -> submission.
- 03b is a utility (resume); 03f is an extra experiment (hybrid, leak-fixed).
- MIND leaderboard: test AUC 0.6581 (Codabench comp. 13967).
