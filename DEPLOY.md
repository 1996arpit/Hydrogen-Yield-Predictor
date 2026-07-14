# Deploy

Streamlit Cloud so others can try the app with no install.

## Local test

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Cloud

1. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub  
2. Deploy repo `1996arpit/Hydrogen-Yield-Predictor`, branch `main`, file `streamlit_app.py`

Needs: `streamlit_app.py`, `requirements.txt`, `data/models/h2_yield_pipeline.joblib`

If model missing → confirm the `.joblib` is pushed. Logs: Manage app → Logs.
