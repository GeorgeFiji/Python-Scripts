# ================= FIX TKINTER WARNINGS =================
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# ================= SKLEARN METRICS =================
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score
)

# ================= CONFIG =================
DATA_FILE = "undersampled_dataset.csv"
OUTPUT_DIR = "model_outputs_undersampled"
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

# ================= USE FULL UNDERSAMPLED DATA FOR TUNING =================
X_sample = X_encoded
y_sample = y_encoded
X_sample_scaled = X_scaled

# ================= DEFINE MODELS AND PARAM GRIDS =================
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

param_grids = {
    "DecisionTree": {
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 5, 10]
    },
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
    if grid:
        X_tune = X_sample_scaled if name == "LogisticRegression" else X_sample
        grid_search = GridSearchCV(model, grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_tune, y_sample)
        best_params[name] = grid_search.best_params_
        print(f"Best params for {name}: {best_params[name]}")
    else:
        best_params[name] = {}

# ================= 10-FOLD CROSS-VALIDATION =================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n==== {name} 10-Fold CV (Full Undersampled Dataset) ====")
    model.set_params(**best_params[name])

    X_cv = X_scaled if name == "LogisticRegression" else X_encoded.to_numpy()
    all_y_test, all_y_pred, fold_importances = [], [], []

    for train_idx, test_idx in cv.split(X_cv, y_encoded):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        if name in ["DecisionTree", "RandomForest"]:
            fold_importances.append(model.feature_importances_)

    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)

    # ================= METRICS =================
    acc = accuracy_score(all_y_test, all_y_pred)
    precision = precision_score(all_y_test, all_y_pred, average=None, zero_division=0)
    recall = recall_score(all_y_test, all_y_pred, average=None, zero_division=0)
    f1 = f1_score(all_y_test, all_y_pred, average=None, zero_division=0)
    mcc = matthews_corrcoef(all_y_test, all_y_pred)
    fpr, tpr, _ = roc_curve(all_y_test, all_y_pred)
    roc_auc = auc(fpr, tpr)
    pr, re, _ = precision_recall_curve(all_y_test, all_y_pred)
    prc_auc = auc(re, pr)
    cm = confusion_matrix(all_y_test, all_y_pred)

    # ---- Save text metrics ----
    metrics_file = os.path.join(OUTPUT_DIR, f"{name}_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {acc:.3f}\nMCC: {mcc:.3f}\nROC AUC: {roc_auc:.3f}\nPRC AUC: {prc_auc:.3f}\n\n")
        f.write("Class\tPrecision\tRecall\tF1\n")
        for i, cls in enumerate(target_encoder.classes_):
            f.write(f"{cls}\t{precision[i]:.3f}\t{recall[i]:.3f}\t{f1[i]:.3f}\n")

    # ================= ENHANCED METRICS PNG =================
    metrics_data = []
    for i, cls in enumerate(target_encoder.classes_):
        metrics_data.append({
            "Class": cls,
            "TP Rate": round(recall[i], 3),
            "FP Rate": round(1 - recall[i], 3),
            "Precision": round(precision[i], 3),
            "Recall": round(recall[i], 3),
            "F1": round(f1[i], 3),
            "MCC": round(mcc, 3),
            "ROC AUC": round(roc_auc, 3),
            "PRC AUC": round(prc_auc, 3),
            "Accuracy": round(acc, 3)
        })
    metrics_df = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(metrics_df)))
    ax.axis('off')
    tbl = ax.table(cellText=metrics_df.values,
                   colLabels=metrics_df.columns,
                   cellLoc='center',
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.2)
    for (row, col), cell in tbl.get_celld().items():
        if col == 0:
            cell.set_width(0.15)
        else:
            cell.set_width(0.1)
    plt.title(f"{name} Enhanced Metrics", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_enhanced_metrics.png"), dpi=300)
    plt.close()

    # ================= CONFUSION MATRIX =================
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"), dpi=300)
    plt.close()

    # ================= FEATURE IMPORTANCE =================
    if name in ["DecisionTree", "RandomForest"]:
        mean_importance = np.mean(fold_importances, axis=0)
        sorted_idx = np.argsort(mean_importance)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(mean_importance)), mean_importance[sorted_idx], color='#4ECDC4', alpha=0.8)
        ax.set_xticks(range(len(mean_importance)))
        ax.set_xticklabels(X.columns[sorted_idx], rotation=90)
        ax.set_ylabel("Average Importance")
        ax.set_title(f"{name} Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_feature_importance.png"), dpi=300)
        plt.close()

# ================= LOGISTIC REGRESSION COEFFICIENT PLOT =================
log_model = models["LogisticRegression"]
log_model.set_params(**best_params["LogisticRegression"])
log_model.fit(X_scaled, y_encoded)

coefs = log_model.coef_[0]
sorted_idx = np.argsort(np.abs(coefs))[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=coefs[sorted_idx], y=X.columns[sorted_idx], ax=ax, palette="viridis")
ax.set_title("Logistic Regression Coefficients (Sorted by Magnitude)")
ax.set_xlabel("Coefficient Value")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "LogisticRegression_coefficients.png"), dpi=300)
plt.close()

# ================= DECISION TREE SVG =================
dt_model = models["DecisionTree"]
dt_model.set_params(**best_params["DecisionTree"])
dt_model.fit(X_encoded, y_encoded)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns,
          class_names=target_encoder.classes_,
          filled=True, rounded=True, fontsize=10, label='none')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "DecisionTree_tree.svg"), format='svg', dpi=300)
plt.close()

print(f"\n✓ All metrics, enhanced PNGs, logistic coefficients, and visualizations saved in '{OUTPUT_DIR}'")
