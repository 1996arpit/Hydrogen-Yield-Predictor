# Deploy the H₂ Yield Predictor

This guide deploys the **Streamlit web app** so recruiters can try predictions in a browser — no install required.

---

## 1. Test locally first

```powershell
cd "D:\Data Science Projects\Hydrogen yield predictor"
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL shown (usually `http://localhost:8501`). Enter values and click **Predict H₂ yield**.

---

## 2. Push to GitHub

If you have not already:

```powershell
git init
git add .
git commit -m "Add Streamlit deployment for H2 yield predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hydrogen-yield-predictor.git
git push -u origin main
```

**Required files for deployment:**

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main app |
| `requirements.txt` | Python dependencies |
| `data/models/h2_yield_pipeline.joblib` | Trained model (must be in the repo) |

---

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Set:
   - **Repository:** `YOUR_USERNAME/hydrogen-yield-predictor`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **Deploy**.

First deploy takes 2–5 minutes. You get a public URL like:

`https://YOUR-APP-NAME.streamlit.app`

Add that link to your resume and README.

---

## 4. After deployment

- [ ] Test the live URL on your phone or another browser
- [ ] Add the live link to `README.md` under a **Live demo** section
- [ ] Put the link on your resume next to the GitHub repo

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model not found | Ensure `data/models/h2_yield_pipeline.joblib` is committed and pushed |
| sklearn version warning | `requirements.txt` pins `scikit-learn>=1.9` to match the saved model |
| App crashes on start | Check Streamlit Cloud logs (Manage app → Logs) |
| Slow first load | Normal — model is cached after first request |

---

## Alternative hosts

| Platform | Notes |
|----------|-------|
| **Streamlit Community Cloud** | Easiest for Streamlit; free tier |
| **Hugging Face Spaces** | Needs a `Dockerfile` or Gradio wrapper |
| **Render / Railway** | Run `streamlit run streamlit_app.py --server.port $PORT` |

Streamlit Community Cloud is recommended for this project.
