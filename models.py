# ================= FIX TKINTER WARNINGS =================
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Tkinter errors

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
    roc_auc_score  # Added to fix undefined variable error
)
# ================= CONFIG =================

DATA_FILE = "11selected.csv"
OUTPUT_DIR = "model_output_11undersample"
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

# ================= SAMPLE FOR HYPERPARAMETER TUNING =================
SAMPLE_SIZE = 100000  # 100K for faster tuning
sample_idx = np.random.RandomState(42).choice(len(X_encoded), size=SAMPLE_SIZE, replace=False)
X_sample = X_encoded.iloc[sample_idx]
y_sample = y_encoded[sample_idx]

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

# ================= HYPERPARAMETER TUNING ON SAMPLE =================
best_params = {}
for name in ["DecisionTree", "RandomForest", "LogisticRegression"]:
    print(f"\n==== Hyperparameter tuning: {name} ====")
    model = models[name]
    grid = param_grids[name]
    if len(grid) > 0:
        grid_search = GridSearchCV(model, grid, cv=3, scoring='accuracy', n_jobs=-1)
        X_tune = X_scaled[sample_idx] if name == "LogisticRegression" else X_sample
        grid_search.fit(X_tune, y_sample)
        best_params[name] = grid_search.best_params_
        print(f"Best params for {name}: {best_params[name]}")
    else:
        best_params[name] = {}

# ================= 10-FOLD CROSS-VALIDATION ON FULL DATASET =================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n==== {name} 10-Fold CV (Full Dataset) ====")

    # Apply best params from tuning
    model.set_params(**best_params[name])

    # Choose scaled or original features
    X_cv = X_scaled if name == "LogisticRegression" else X_encoded.to_numpy()

    # Lists to store predictions for all records
    all_y_test = []
    all_y_pred = []

    # Feature importance collection for tree models
    if name in ["DecisionTree", "RandomForest"]:
        fold_importances = []

    # 10-Fold CV
    for train_idx, test_idx in cv.split(X_cv, y_encoded):
        X_train_cv, X_test_cv = X_cv[train_idx], X_cv[test_idx]
        y_train_cv, y_test_cv = y_encoded[train_idx], y_encoded[test_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)

        all_y_test.extend(y_test_cv)
        all_y_pred.extend(y_pred_cv)

        # Collect feature importance per fold
        if name in ["DecisionTree", "RandomForest"]:
            fold_importances.append(model.feature_importances_)

    # Convert to NumPy arrays
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)

    # ================= FULL DATASET METRICS =================
    full_acc = accuracy_score(all_y_test, all_y_pred)
    full_precision = precision_score(all_y_test, all_y_pred, average=None, zero_division=0)
    full_recall = recall_score(all_y_test, all_y_pred, average=None, zero_division=0)
    full_f1 = f1_score(all_y_test, all_y_pred, average=None, zero_division=0)
    full_cm = confusion_matrix(all_y_test, all_y_pred)

    # Save metrics
    with open(f"{OUTPUT_DIR}/{name}_full_dataset_metrics.txt", "w") as f:
        f.write(f"Accuracy: {full_acc:.4f}\n\n")
        f.write("Class\tPrecision\tRecall\tF1\n")
        for i, class_name in enumerate(target_encoder.classes_):
            f.write(f"{class_name}\t{full_precision[i]:.4f}\t{full_recall[i]:.4f}\t{full_f1[i]:.4f}\n")

    print(f"Full Dataset Accuracy: {full_acc:.4f}")

    # ================= CONFUSION MATRIX =================
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(full_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} Confusion Matrix (Full Dataset)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_confusion_matrix_full_dataset.png", dpi=300)
    plt.close()

    # ================= FEATURE IMPORTANCE =================
    if name in ["DecisionTree", "RandomForest"]:
        mean_importance = np.mean(fold_importances, axis=0)
        feature_names = X.columns
        sorted_idx = np.argsort(mean_importance)[::-1]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(range(len(mean_importance)), mean_importance[sorted_idx], color='#4ECDC4', alpha=0.8)
        ax.set_xticks(range(len(mean_importance)))
        ax.set_xticklabels(feature_names[sorted_idx], rotation=90)
        ax.set_ylabel("Average Importance")
        ax.set_title(f"{name} Feature Importance (Full Dataset Average)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{name}_feature_importance_full_dataset.png", dpi=300)
        plt.close()

from sklearn.tree import plot_tree

# DecisionTree visualization (trained on full dataset)
if name == "DecisionTree":
    # Fit on full dataset
    model.fit(X_encoded, y_encoded)

    # Plot tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=X.columns,       # show feature names
        class_names=target_encoder.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
        label='none'                   # remove class/value labels for readability
    )

    plt.tight_layout()
    svg_path = f"{OUTPUT_DIR}/{name}_tree.svg"
    fig.savefig(svg_path, format='svg', dpi=300)
    plt.close()
    print(f"Decision Tree saved as SVG: {svg_path}")


    # ================= ROC CURVE (FULL DATASET) =================
    if hasattr(model, "predict_proba"):
        all_y_proba = model.predict_proba(X_cv)[:, 1]
    else:
        all_y_proba = all_y_pred

    fpr, tpr, _ = roc_curve(all_y_test, all_y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0,1], [0,1], linestyle='--', color='gray', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} ROC Curve (Full Dataset)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_roc_curve_full_dataset.png", dpi=300)
    plt.close()

from sklearn.metrics import matthews_corrcoef, precision_recall_curve

# ================= FULL DATASET METRICS =================
def compute_metrics(y_true, y_pred, y_proba=None):
    metrics_list = []
    classes = np.unique(y_true)
    for cls in classes:
        # Binary arrays for current class
        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)
        
        tp = np.sum((y_true_cls == 1) & (y_pred_cls == 1))
        fp = np.sum((y_true_cls == 0) & (y_pred_cls == 1))
        fn = np.sum((y_true_cls == 1) & (y_pred_cls == 0))
        tn = np.sum((y_true_cls == 0) & (y_pred_cls == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
        mcc = matthews_corrcoef(y_true_cls, y_pred_cls)
        roc_area = roc_auc_score(y_true_cls, y_proba[:, cls] if y_proba is not None else y_pred_cls)
        
        # PRC Area
        if y_proba is not None:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_cls, y_proba[:, cls])
            prc_area = np.trapz(precision_curve, recall_curve)
        else:
            prc_area = 0
        
        metrics_list.append([tpr, fpr, precision, recall, f1, mcc, roc_area, prc_area, cls])
    return metrics_list

# ================= SAVE METRICS AS PNG =================


import matplotlib.table as tbl

def save_metrics_png(metrics_list, class_labels, accuracy, filename, title):
    """
    Saves a detailed metrics table as PNG with rounded values, readable column widths,
    and overall accuracy shown at the top.
    """
    # Round numeric values to 3 decimals and replace class index with class name
    rounded_data = [
        [f"{row[0]:.3f}", f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.3f}",
         f"{row[4]:.3f}", f"{row[5]:.3f}", f"{row[6]:.3f}", f"{row[7]:.3f}",
         str(class_labels[int(row[8])])]  # use readable class name
        for row in metrics_list
    ]
    
    # Header row
    headers = ["TP Rate", "FP Rate", "Precision", "Recall", "F-Measure",
               "MCC", "ROC Area", "PRC Area", "Class"]
    table_data = [headers] + rounded_data

    # Create figure
    fig_height = max(2, 1 + len(rounded_data)*0.6)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis('off')

    # Add accuracy title
    plt.title(f"{title}\nOverall Accuracy: {accuracy:.3f}", fontsize=14, pad=20)

    # Create table
    table = tbl.table(ax, cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    # Widen the "Class" column for readability
    col_widths = [0.9]*8 + [1.8]
    for i, width in enumerate(col_widths):
        for key, cell in table.get_celld().items():
            if key[1] == i:  # column index
                cell.set_width(width)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



# ================= LOGISTIC REGRESSION COEFFICIENTS =================
def plot_logreg_coefficients(model, feature_names, filename):
    coef = model.coef_[0]
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(range(len(coef)), coef[sorted_idx], color='#FF6B6B', alpha=0.8)
    ax.set_xticks(range(len(coef)))
    ax.set_xticklabels(np.array(feature_names)[sorted_idx], rotation=90)
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Logistic Regression Feature Coefficients")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ================= APPLY METRICS & VISUALIZATION =================
for name, model in models.items():
    # Choose scaled or original features
    X_cv = X_scaled if name == "LogisticRegression" else X_encoded.to_numpy()

    model.set_params(**best_params[name])
    model.fit(X_cv, y_encoded)

    if hasattr(model, "predict_proba"):
        y_proba_full = model.predict_proba(X_cv)
    else:
        y_proba_full = None

    y_pred_full = model.predict(X_cv)
    metrics_list = compute_metrics(y_encoded, y_pred_full, y_proba_full)
    overall_acc = accuracy_score(y_encoded, y_pred_full)

    save_metrics_png(
        metrics_list,
        target_encoder.classes_,
        overall_acc,
        f"{OUTPUT_DIR}/{name}_metrics_table.png",
        f"{name} Full Dataset Metrics"
    )
    if name == "LogisticRegression":
        plot_logreg_coefficients(model, X.columns, f"{OUTPUT_DIR}/{name}_coefficients.png")

print("✓ Metrics table PNG and Logistic Regression coefficients PNG saved.")


print(f"\n✓ All metrics, plots, feature importance, ROC curves, and tree diagram saved in '{OUTPUT_DIR}'")
