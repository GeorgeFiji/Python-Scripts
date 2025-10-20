# ================= FIX TKINTER WARNINGS =================
import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, matthews_corrcoef, precision_recall_curve, roc_auc_score
)
from imblearn.combine import SMOTEENN  # <--- Change here

# ================= CONFIG =================
DATA_FILE = "final.csv"
OUTPUT_DIR = "model_10fold_smoteenn"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# ================= LOAD DATA =================
df = pd.read_csv(DATA_FILE)

# ================= SEPARATE FEATURES AND TARGET =================
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# ================= ENCODE CATEGORICAL VARIABLES =================
label_encoders = {}
X_encoded = X.copy()
for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ================= SCALE FOR LOGISTIC REGRESSION =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# ================= ORIGINAL CLASS DISTRIBUTION =================
original_counts = pd.Series(y_encoded).value_counts().sort_index()
print("\n=== Original Class Distribution ===")
for i, count in original_counts.items():
    print(f"Class {target_encoder.classes_[i]}: {count}")
print(f"Total: {len(y_encoded)}")

# ================= SAMPLE FOR HYPERPARAMETER TUNING =================
SAMPLE_SIZE = 100000
sample_idx = np.random.RandomState(42).choice(len(X_encoded), size=SAMPLE_SIZE, replace=False)
X_sample = X_encoded.iloc[sample_idx]
y_sample = y_encoded[sample_idx]

# ================= DEFINE MODELS AND PARAM GRIDS =================
models = {
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2']
    },
    "LogisticRegression": {
        'C': [0.1, 1.0, 10],
        'solver': ['lbfgs']
    }
}

# ================= HYPERPARAMETER TUNING =================
best_params = {}
for name in models.keys():
    print(f"\n==== Hyperparameter tuning: {name} ====")
    model = models[name]
    grid = param_grids[name]
    grid_search = GridSearchCV(model, grid, cv=3, scoring='accuracy', n_jobs=-1)
    X_tune = X_scaled[sample_idx] if name == "LogisticRegression" else X_sample
    grid_search.fit(X_tune, y_sample)
    best_params[name] = grid_search.best_params_
    print(f"Best params for {name}: {best_params[name]}")

# ================= 10-FOLD CV WITH SMOTE-ENN =================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
smote_enn = SMOTEENN(random_state=42)  # <--- Use SMOTE-ENN here

for name, model in models.items():
    print(f"\n==== {name} 10-Fold CV (With SMOTE-ENN) ====")
    model.set_params(**best_params[name])

    X_cv = X_scaled if name == "LogisticRegression" else X_encoded.to_numpy()
    all_y_test, all_y_pred = [], []
    if name == "RandomForest":
        fold_importances = []

    fold_num = 1
    for train_idx, test_idx in cv.split(X_cv, y_encoded):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # === Before SMOTE-ENN ===
        before_counts = pd.Series(y_train).value_counts().sort_index()

        # === Apply SMOTE-ENN ===
        X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
        after_counts = pd.Series(y_train_res).value_counts().sort_index()

        print(f"\nFold {fold_num} - Class Distribution:")
        for i in before_counts.index:
            print(f"  Class {target_encoder.classes_[i]}: Before={before_counts[i]}, After={after_counts[i]}")
        print(f"  Total Before={len(y_train)}, After={len(y_train_res)}")

        # === Visualize Before vs After ===
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(["Before_0", "Before_1"], before_counts.values, alpha=0.7, label='Before SMOTE-ENN')
        ax.bar(["After_0", "After_1"], after_counts.values, alpha=0.7, label='After SMOTE-ENN')
        ax.set_title(f"{name} - Fold {fold_num} Class Counts")
        ax.set_ylabel("Samples")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{name}_fold{fold_num}_smoteenn_counts.png", dpi=300)
        plt.close()

        # === Train model ===
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        if name == "RandomForest":
            fold_importances.append(model.feature_importances_)

        fold_num += 1

    # ================= METRICS & VISUALS =================
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    full_acc = accuracy_score(all_y_test, all_y_pred)
    full_precision = precision_score(all_y_test, all_y_pred, average=None, zero_division=0)
    full_recall = recall_score(all_y_test, all_y_pred, average=None, zero_division=0)
    full_f1 = f1_score(all_y_test, all_y_pred, average=None, zero_division=0)
    full_cm = confusion_matrix(all_y_test, all_y_pred)

    print(f"\n{name} CV Accuracy (SMOTE-ENN): {full_acc:.4f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(full_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} Confusion Matrix (10-Fold CV)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_confusion_matrix_smoteenn.png", dpi=300)
    plt.close()

    # ROC curve
    if hasattr(model, "predict_proba"):
        all_y_proba = model.predict_proba(X_cv)[:, 1]
        fpr, tpr, _ = roc_curve(y_encoded, all_y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0,1], [0,1], linestyle='--', color='gray', lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name} ROC Curve (Full Dataset)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{name}_roc_curve_smoteenn.png", dpi=300)
        plt.close()

    # Feature importance (RF only)
    if name == "RandomForest":
        mean_importance = np.mean(fold_importances, axis=0)
        sorted_idx = np.argsort(mean_importance)[::-1]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(range(len(mean_importance)), mean_importance[sorted_idx], color='#4ECDC4', alpha=0.8)
        ax.set_xticks(range(len(mean_importance)))
        ax.set_xticklabels(X.columns[sorted_idx], rotation=90)
        ax.set_ylabel("Average Importance")
        ax.set_title("Random Forest Feature Importance (SMOTE-ENN CV)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/RandomForest_feature_importance_smoteenn.png", dpi=300)
        plt.close()

    # Logistic regression coefficients
    if name == "LogisticRegression":
        coef = model.coef_[0]
        sorted_idx = np.argsort(np.abs(coef))[::-1]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(range(len(coef)), coef[sorted_idx], color='#FF6B6B', alpha=0.8)
        ax.set_xticks(range(len(coef)))
        ax.set_xticklabels(np.array(X.columns)[sorted_idx], rotation=90)
        ax.set_ylabel("Coefficient Value")
        ax.set_title("Logistic Regression Coefficients (SMOTE-ENN CV)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/LogisticRegression_coefficients_smoteenn.png", dpi=300)
        plt.close()

print("\n✓ Completed SMOTE-ENN 10-Fold CV for Random Forest & Logistic Regression.")
print(f"All outputs saved in '{OUTPUT_DIR}'")
