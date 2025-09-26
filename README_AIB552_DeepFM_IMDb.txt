README — AIB552 End-of-Course Assessment (July 2025): Residual MLP + DeepFM for IMDb Reviews
============================================================================================

Project summary
---------------
This Jupyter notebook implements a leakage-safe, two-view pipeline for binary sentiment classification on IMDb reviews.
It combines a Factorization Machine (FM) on a sparse bag-of-words view with a Residual-MLP on a compact dense view 
(TruncatedSVD-32). The branches are fused (DeepFM-style) to produce a logit, with optional post-hoc calibration.
Evaluation covers ROC/PR curves, confusion matrices, and business-aligned thresholds.

Requirements
------------
Python 3.9+ and the following packages (detected from imports):
- matplotlib
- numpy
- pandas
- scipy
- sklearn
- torch
GPU is optional; a CUDA build of PyTorch accelerates training.

Data
----
Expect an IMDb-like reviews dataset (text + binary label). Create a high-dimensional sparse bag-of-words (CSR).
Use a stratified 70/15/15 split. Fit all transforms (variance filter, SVD, scaler) on **train only** and apply to val/test.

Pipeline (major steps)
----------------------
- STEP #1 — Integrity checks & schema audit cell
- STEP #2 — Stratified split (70/15/15)
- STEP #3 — Drop zero-variance columns
- STEP #4 — Prepare the FM/linear view (sparse CSR)
- STEP #5 — Low-rank projection for the deep view (SVD, fit only on the training dataset only)
- STEP #6 — Scale the dense factors (fit only on the training dataset only)
- STEP #7 — Imbalance handling (class weights, TRAIN only)
- STEP #8 — Persist preprocessor (train-fitted mask + SVD + scaler + metadata)
- STEP #9 — Build PyTorch Dataset (two-view Dataset)
- STEP #10 — Integrity checks (shapes, dtypes, alignment, finiteness)
- STEP #9a Model: FM block (Linear + pairwise (factorization trick)
- STEP #9b Model: Build the Residual MLP (Deep path with skip connections)
- STEP #9c Model: Full model (make_model) (Fuse FM + deep → final logit)
- STEP #9d Model: Model Instantiation & callbacks (Optimizer, scheduler, early stop)
- STEP #9e Model: Model Training loop-(fit) (Train/validate; early stopping)
- STEP #10: One-step helpers (Sigmoid/Probability/Class)
- STEP #11: Choose threshold τ (on validation) (Pick τ to maximize F1; report Youden J as backup)
- Step #12: Test forecasts (apply best τ)
- Step #13: Metrics & report (Val + Test)
- Step #13 (plots): Visualise confusion matrices + validation/test classification reports
- STEP #14: Plotting (Train/val loss curves, ROC curves, PR curves)
- STEP #15: Fine-Tuning Hyper-Parameters (compact grid + LR candidates)

Model architecture
------------------
- FM: bias w0, linear weights w, latent matrix V (rank k) for pairwise interactions.
- Residual-MLP: Z (e.g., 32-D SVD) → Linear(256) → residual blocks (LayerNorm, GELU, Dropout).
- Fusion: concat(FM logit, deep features) → Linear(1) → logit; BCEWithLogitsLoss + AdamW; early stopping.

Metrics & typical results
-------------------------
- ROC-AUC ≈ 0.926, PR-AUC ≈ 0.923; F1/Accuracy ≈ 0.85 at validation-chosen τ;
  log-loss ≈ 0.47; Brier ≈ 0.116 (values vary with seed/threshold).

How to run
----------
1) Open the notebook and run cells top-to-bottom.
2) Set dataset paths and verify environment (first cells).
3) Execute STEPs #1–#9 to build the two-view dataset, then STEPs #9a–#9e to define/train the model.
4) Run evaluation (threshold selection, test, metrics/plots) and optional calibration/dashboard.

Enhancements (optional)
-----------------------
- Enhancement 1: Replace SVD with sentence embeddings + ANN retrieval for candidates; feed embeddings to the deep path.
- Enhancement 2: Sparse FM via EmbeddingBag (with feature hashing), AMP/compile, ONNX + INT8 for fast inference.
Re-calibrate probabilities after either change to keep CTR/precision@k stable.

Reproducibility
---------------
- Set seeds; expect small variance across runs. Pick τ on validation to match business costs (precision/recall).

License & acknowledgements
--------------------------
Academic use only. References: Rendle (2010), He et al. (2016), Deerwester et al. (1990), Kingma & Ba (2015),
Saito & Rehmsmeier (2015), Brier (1950).