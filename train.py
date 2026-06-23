import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


##################################################
# Paths
##################################################

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datasets" / "heart.csv"
IMAGES_DIR = BASE_DIR / "images"
MODELS_DIR = BASE_DIR / "models"

# Ensure output directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


##################################################
# Load Dataset
##################################################

print("Loading dataset from:", DATA_PATH)

df = pd.read_csv(DATA_PATH)


##################################################
# Basic Information
##################################################

print("\nColumns:\n")
print(df.columns.tolist())

print("\nMissing Values:\n")
print(df.isnull().sum())


##################################################
# Target Variable
##################################################

# UCI dataset target column = "num"
# Convert multiclass (0–4) to binary (0 = no disease, 1 = disease)
y = df["num"]
y = (y > 0).astype(int)


##################################################
# Features
##################################################

# Drop target + any meta columns that may be present
cols_to_drop = ["num"]
for col in ["id", "dataset"]:
    if col in df.columns:
        cols_to_drop.append(col)

X = df.drop(cols_to_drop, axis=1)


##################################################
# Missing Values – fill numeric with median, drop remaining
##################################################

X = X.fillna(X.median(numeric_only=True))

# Drop any leftover non-numeric columns (e.g. string columns with NaN)
X = X.select_dtypes(include=["number"])


##################################################
# Verify
##################################################

print("\nFeature Shape:", X.shape)
print("Features used:", X.columns.tolist())
print("\nTarget Distribution:")
print(y.value_counts())


##################################################
# Train / Test Split
##################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


##################################################
# Models
##################################################

lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)


##################################################
# Training
##################################################

print("\nTraining Models...\n")

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)


##################################################
# Accuracy
##################################################

models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
}

print("\nModel Accuracy\n")

for name, model in models.items():
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"  {name}: {acc:.4f}")


##################################################
# Confusion Matrix
##################################################

pred_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, pred_rf)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest – Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "confusion_matrix.png")
plt.close()
print("\nSaved: images/confusion_matrix.png")


##################################################
# ROC Curve
##################################################

prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, prob)
auc = roc_auc_score(y_test, prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#e74c3c", linewidth=2)
plt.plot([0, 1], [0, 1], "r--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig(IMAGES_DIR / "roc_curve.png")
plt.close()
print("Saved: images/roc_curve.png")


##################################################
# Save Model + Feature Names
##################################################

model_path = MODELS_DIR / "model.pkl"
joblib.dump(rf, model_path)
print(f"\nModel saved → {model_path}")

# Save feature column names so the app can align inputs correctly
feature_path = MODELS_DIR / "features.pkl"
joblib.dump(X.columns.tolist(), feature_path)
print(f"Features saved → {feature_path}")

print("\nDone ✓")