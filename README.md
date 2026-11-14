# NeuroScan

**Brain MRI tumor classification** — a Flask web app that classifies an MRI slice as
**Glioma**, **Meningioma**, **Pituitary Tumor** or **No Tumor**.

Features are extracted with **HOG + LBP + GLCM** and classified by a **Support Vector Machine**.

| Metric | Score |
|---|---|
| Accuracy | 90.75% |
| Precision | 91.58% |
| Recall | 90.75% |
| F1 score | 90.40% |
| 5-fold CV accuracy | 94.52% |

> For research and education only. Not a medical device.

---

## Project structure

```
neuroscan/
├── app.py                 # Flask app (routes: /, /predict, /health)
├── predictor.py           # Feature extraction + SVM prediction
├── download_models.py     # Fetches the model files if missing / Git LFS pointers
├── requirements.txt       # Pinned versions (scikit-learn 1.6.1 matches the model)
├── gunicorn.conf.py       # Production server settings
├── render.yaml            # Render blueprint (one-click deploy)
├── Procfile
├── .python-version        # Python 3.11.11
├── saved_models/          # SVM model, scaler, label encoder (.pkl)
├── templates/index.html   # UI
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── samples/           # 4 sample MRIs for the "Try a sample" buttons
│   └── uploads/           # user uploads (auto-pruned to the last 50)
├── results/Model_Comparison.csv
└── notebook/Brain_Tumor_Model.ipynb
```

---

## Run locally

```bash
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

Use Python 3.10 – 3.12 (3.11 recommended).

---

## Deploy on Render

### 1. Push to GitHub

`Best_BrainTumor_Model.pkl` is ~257 MB, above GitHub's 100 MB limit, so track it with Git LFS:

```bash
git init
git lfs install
git lfs track "*.pkl"          # already set in .gitattributes
git add .
git commit -m "NeuroScan"
git branch -M main
git remote add origin https://github.com/<your-username>/neuroscan.git
git push -u origin main
```

> No Git LFS? You can also skip committing `saved_models/*.pkl` entirely:
> the build runs `download_models.py`, which downloads any missing model files
> automatically.

### 2. Create the service

**Option A — Blueprint (easiest):** Render dashboard → **New → Blueprint** → pick your repo.
`render.yaml` sets everything up.

**Option B — Manual Web Service:** Render dashboard → **New → Web Service** → pick your repo, then:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt && python download_models.py` |
| Start Command | `gunicorn app:app -c gunicorn.conf.py` |
| Health Check Path | `/health` |
| Environment variable | `PYTHON_VERSION` = `3.11.11` |

### Notes

* **Memory:** the SVM model uses ~400 MB of RAM once loaded, so the app runs a
  single gunicorn worker (2 threads) to stay under the free plan's 512 MB.
  On a paid plan with more RAM you can raise `WEB_CONCURRENCY`.
* **Cold starts:** free instances sleep after inactivity; the first request
  after waking takes ~30–60 s while the model loads.
* **Uploads** are stored on Render's ephemeral disk and are wiped on each deploy — that's fine,
  they're only used for the prediction.
* **Custom model host:** set the `MODEL_BASE_URL` env var to change where
  `download_models.py` fetches the `.pkl` files from.

---

## API

`POST /predict` — multipart form with either `image` (file) or `sample`
(`glioma` | `meningioma` | `notumor` | `pituitary`).

```json
{
  "success": true,
  "prediction": "meningioma",
  "label": "Meningioma",
  "confidence": 95.5,
  "probabilities": {"glioma": 4.31, "meningioma": 95.5, "notumor": 0.17, "pituitary": 0.02},
  "summary": "A usually slow-growing tumour of the meninges...",
  "image": "/static/uploads/<id>.jpg"
}
```

`GET /health` → `{"status": "ok", "app": "NeuroScan"}`
