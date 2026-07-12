# Hydrogen Yield Predictor

**Machine learning pipeline to predict H₂ yield from dark fermentation process parameters — with leakage-aware validation, synthetic data expansion, and an interactive prediction tool.**

> Built for real experimental constraints: small sample size (66 runs), grouped observations, and a dominant process variable (F/M ratio).

---

## Why this project matters

Dark fermentation experiments are expensive and slow. This project asks: *given substrate, inoculum, and process chemistry, can we predict hydrogen yield before running the reactor?*

What makes it interview-ready:

| Highlight | What it shows employers |
|-----------|-------------------------|
| **Leakage-aware validation** | Source-level train/test split + GroupKFold CV — no duplicate experimental groups leaking across folds |
| **Synthetic data with guardrails** | Expanded 66 experiments → 6,000 rows using statistically anchored generation, then validated before modeling |
| **Honest model comparison** | Compared 10+ models; tuned Extra Trees selected on CV, not holdout cherry-picking |
| **Domain insight over hype** | F/M ratio dominates; a simple F/M baseline (R² 0.83) beats the full 15-feature model (R² 0.72) on original holdout |
| **Production-style delivery** | Serialized sklearn pipeline + interactive notebook predictor with correct physical units |

---

## Results at a glance

Evaluated on **13 held-out original experimental source groups** (never seen during training):

| Model | Original holdout R² | MAE (mL H₂/g VS) |
|-------|---------------------|------------------|
| F/M group-mean baseline | **0.83** | 38.1 |
| F/M-only Extra Trees | 0.83 | 38.8 |
| **Full 15-feature Extra Trees (deployed)** | **0.72** | 47.9 |

Selected model: **Tuned Extra Trees** — GroupKFold CV R² = 0.67, balances complexity and generalization.

**Key takeaway:** The full model adds value for multi-variable scenarios, but F/M ratio explains most of the variance. Documenting this trade-off shows scientific honesty, not failure.

---

## For recruiters (no install required)

You do **not** need to run anything to evaluate this project:

1. Read this README (2 min)
2. Open [`data/05_h2_prediction_model.ipynb`](data/05_h2_prediction_model.ipynb) on GitHub — scroll to saved outputs for model comparison, holdout metrics, and feature importance
3. Skim the notebook pipeline: 01 → 06 tells the full story from raw data to prediction

**Optional demo:** Clone and run only Notebook 06 Cell 1 (~2 min setup) to try the interactive predictor.

---

## Project pipeline

```
01  Data integration & quality audit     →  66 experimental runs cleaned
02  Group-wise statistical characterization
03  Synthetic data generator benchmarking
04  Synthetic data validation
05  ML modeling, tuning & pipeline export  →  h2_yield_pipeline.joblib
06  Interactive H₂ yield prediction        →  tkinter popup inputs
```

---

## Live demo (after deployment)

**Streamlit app:** `streamlit run streamlit_app.py` locally, or deploy to [Streamlit Community Cloud](https://share.streamlit.io) — see [`DEPLOY.md`](DEPLOY.md).

Once deployed, add your public URL here: `https://YOUR-APP.streamlit.app`

## Interactive prediction (Notebook 06)

Run **Cell 1** in [`data/06_h2_yield_prediction.ipynb`](data/06_h2_yield_prediction.ipynb).

- Asks for **10 primary inputs** (MC, VS, SCOD, TCOD, etc.) with units
- Auto-derives TS, FS, and SCOD/TCOD ratio
- Returns predicted H₂ yield in **mL H₂/g VS**

![Add a screenshot here after running Cell 1](docs/demo-screenshot.png)

*Tip: Take a screenshot of the popup + prediction output and save it as `docs/demo-screenshot.png` before pushing to GitHub.*

---

## Quick start (technical reviewers)

### Option A — pip

```bash
git clone https://github.com/YOUR_USERNAME/hydrogen-yield-predictor.git
cd hydrogen-yield-predictor
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
jupyter notebook data/06_h2_yield_prediction.ipynb
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate h2-yield-predictor
jupyter notebook data/06_h2_yield_prediction.ipynb
```

The saved model is already included at `data/models/h2_yield_pipeline.joblib`. Re-training is optional:

```bash
python data/train_and_save_pipeline.py
```

---

## Dataset

| File | Description |
|------|-------------|
| `data/data.csv` | Original 66 experimental observations |
| `data/h2_data.csv` | Modeling dataset (6,000 rows, 61 source groups) |
| `data/reference_experimental_data.csv` | Reference subset for holdout recovery |

**Predictors (15):** substrate/inoculum MC, TS, VS, FS; SCOD, TCOD, SCOD/TCOD ratio; TRS, lignin, HMF; F/M ratio

**Target:** H₂ yield (mL H₂/g VS)

**F/M ratios:** 0.5, 1.0, 1.5, 2.0, 2.5, 3.0

---

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Jupyter · tkinter

---

## Resume bullets (copy-paste)

- Built an end-to-end ML pipeline predicting H₂ yield from dark fermentation parameters, expanding 66 experiments to 6,000 statistically validated synthetic observations
- Designed leakage-aware evaluation using source-level stratified splits and GroupKFold cross-validation on experimental source IDs
- Benchmarked 10+ regressors with hyperparameter tuning; deployed tuned Extra Trees (holdout R² 0.72) with serialized pipeline and unit-aware interactive predictor
- Identified F/M ratio as the dominant predictor via permutation importance and baseline comparison — demonstrating when simpler models outperform complex ones

---

## Author

**Your Name** — [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [Email](mailto:you@email.com)

Replace placeholders above before sharing with employers.

---

## License

MIT — free to review, fork, and discuss in interviews.
