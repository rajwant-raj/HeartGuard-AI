# ❤️ HeartGuard AI – Heart Disease Predictor

A premium machine learning web app that predicts cardiovascular disease risk using clinical parameters.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (saves model.pkl + features.pkl to /models)
python train.py

# 3. Launch the web app
streamlit run app.py
```

## 🤖 ML Algorithms

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based model |
| **Random Forest** | **Best model – used for predictions** |

## 📊 Metrics & Outputs

- Model accuracy comparison across all 3 models
- Confusion Matrix → `images/confusion_matrix.png`
- ROC Curve with AUC score → `images/roc_curve.png`
- Trained model → `models/model.pkl`
- Feature list → `models/features.pkl`

## 🗂️ Dataset

UCI Heart Disease dataset (`datasets/heart.csv`).  
Target: `num` column (0 = no disease, 1–4 = disease → converted to binary).

## 🛠️ Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Frontend:** Streamlit (dark premium UI)
- **Persistence:** joblib