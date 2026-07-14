# Hydrogen Yield Predictor

Predict H₂ yield from dark fermentation parameters — leakage-aware ML pipeline with synthetic data expansion and a live-style demo.

## Highlights

- Source-level splits / GroupKFold (no row leakage)
- F/M alone ≈ R² 0.83; full Extra Trees holdout ≈ 0.72
- End-to-end: clean → synthesize → model → Streamlit/notebook

## Results (13 held-out experimental groups)

| Model | Holdout R² | MAE |
|-------|------------|-----|
| F/M group-mean baseline | **0.83** | 38.1 |
| F/M-only Extra Trees | 0.83 | 38.8 |
| Full Extra Trees (deployed) | 0.72 | 47.9 |

Deployed model: **Tuned Extra Trees** (GroupKFold CV R² ≈ 0.67).

## Live demo

**Try the app:** https://hydrogen-yield-predictor-wubzgkxntf4hxc2zqtenqf.streamlit.app/

## Layout

`notebooks/` · `data/raw|processed|models/` · `scripts/` · `streamlit_app.py`

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Notebook demo: [`notebooks/06_h2_yield_prediction.ipynb`](notebooks/06_h2_yield_prediction.ipynb)  
Deploy: [`DEPLOY.md`](DEPLOY.md) · Repo: https://github.com/1996arpit/Hydrogen-Yield-Predictor  
Retrain: `python scripts/train_and_save_pipeline.py`

## Data

`data/raw/data.csv` (66 runs) · `data/processed/h2_data.csv` (6,000 rows) · target: mL H₂/g VS

## Resume bullets

- H₂-yield pipeline from 66 experiments → 6,000 validated synthetic rows
- Leakage-aware source-level CV; tuned Extra Trees (holdout R² 0.72)
- Unit-aware Streamlit/notebook predictor; F/M baselines beat the full model on holdout

## Author

**Arpit Sharma** — [LinkedIn](https://www.linkedin.com/in/arpit-sharma-132547315/) · [Email](asharma1996tkd@gmail.com)

MIT
