# ================= FIX TKINTER WARNINGS =================
import matplotlib
matplotlib.use('Agg')
import matplotlib.table as tbl


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
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
OUTPUT_DIR = "model_outputs_smote_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# ================= LOAD SMOTE TRAINING AND ORIGINAL TEST SET =================
X_train = pd.read_csv("smote_combined_visualizationall/X_train_smote_features.csv")
y_train = pd.read_csv("smote_combined_visualizationall/y_train_smote.csv")['Diabetes_binary']
X_test = pd.read_csv("smote_combined_visualizationall/X_test_original.csv")
y_test = X_test.pop('Diabetes_binary')

# ✅ FIX: Convert string labels to numeric if needed
if y_train.dtype == 'object':
    y_train = y_train.map({'No_diabetes': 0, 'Pre_diabetic_or_has_diabetes': 1})
if y_test.dtype == 'object':
    y_test = y_test.map({'No_diabetes': 0, 'Pre_diabetic_or_has_diabetes': 1})

# ================= ENCODE CATEGORICAL VARIABLES =================
label_encoders = {}
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

# ================= SCALE FOR LOGISTIC REGRESSION =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= DEFINE MODELS AND PARAM GRIDS =================
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

param_grids = {
    "DecisionTree": {'max_depth': [10, 20, None], 'min_samples_leaf': [1, 5, 10]},
    "RandomForest": {'n_estimators': [100, 200], 'max_depth': [10, 20, None],
                     'min_samples_leaf': [1, 5, 10], 'max_features': ['sqrt', 'log2']},
    "LogisticRegression": {'C': [0.1, 1.0, 10], 'solver': ['lbfgs']}
}

# ================= HYPERPARAMETER TUNING =================
best_params = {}
SAMPLE_SIZE = min(100000, len(X_train))
sample_idx = np.random.RandomState(42).choice(len(X_train), size=SAMPLE_SIZE, replace=False)

for name, model in models.items():
    print(f"\n==== Hyperparameter tuning: {name} ====")
    grid = param_grids[name]
    if len(grid) > 0:
        grid_search = GridSearchCV(model, grid, cv=3, scoring='accuracy', n_jobs=-1)
        X_tune = X_train_scaled[sample_idx] if name == "LogisticRegression" else X_train.iloc[sample_idx]
        grid_search.fit(X_tune, y_train.iloc[sample_idx])
        best_params[name] = grid_search.best_params_
        print(f"Best params for {name}: {best_params[name]}")
    else:
        best_params[name] = {}

# ================= METRIC COMPUTATION =================
def compute_metrics(y_true, y_pred, y_proba=None):
    metrics_list = []
    classes = np.unique(y_true)
    for cls in classes:
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
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mcc = matthews_corrcoef(y_true_cls, y_pred_cls)

        # ROC and PRC areas
        if y_proba is not None:
            roc_area = roc_auc_score(y_true_cls, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_cls, y_proba)
            prc_area = np.trapz(precision_curve, recall_curve)
        else:
            roc_area, prc_area = 0, 0

        metrics_list.append([tpr, fpr, precision, recall, f1, mcc, roc_area, prc_area, cls])
    return metrics_list


# ================= SAVE METRICS AS PNG =================
def save_metrics_png(metrics_list, class_labels, accuracy, filename, title):
    rounded_data = [
        [f"{row[0]:.3f}", f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.3f}",
         f"{row[4]:.3f}", f"{row[5]:.3f}", f"{row[6]:.3f}", f"{row[7]:.3f}",
         str(class_labels[int(row[8])])]
        for row in metrics_list
    ]
    headers = ["TP Rate", "FP Rate", "Precision", "Recall", "F-Measure",
               "MCC", "ROC Area", "PRC Area", "Class"]
    table_data = [headers] + rounded_data

    fig_height = max(2, 1 + len(rounded_data)*0.6)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis('off')
    plt.title(f"{title}\nOverall Accuracy: {accuracy:.3f}", fontsize=14, pad=20)

    table = tbl.table(ax, cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    col_widths = [0.9]*8 + [1.8]
    for i, width in enumerate(col_widths):
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ================= LOGISTIC REGRESSION COEFFICIENT PLOT =================
def plot_logreg_coefficients(model, feature_names, filename):
    coef = model.coef_[0]
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(coef)), coef[sorted_idx], color='#FF6B6B', alpha=0.8)
    ax.set_xticks(range(len(coef)))
    ax.set_xticklabels(np.array(feature_names)[sorted_idx], rotation=90)
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Logistic Regression Coefficients (Trained on SMOTE)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ================= TRAIN ON FULL SMOTE TRAINING SET & EVALUATE ON ORIGINAL TEST SET =================
for name, model in models.items():
    print(f"\n==== {name} Training on SMOTE & Evaluating on Original Test Set ====")
    model.set_params(**best_params[name])

    X_train_model = X_train_scaled if name == "LogisticRegression" else X_train
    X_test_model = X_test_scaled if name == "LogisticRegression" else X_test

    # Train on full SMOTE training set
    model.fit(X_train_model, y_train)
    y_pred_test = model.predict(X_test_model)

    # ================= METRICS =================
    acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred_test, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average=None, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    with open(f"{OUTPUT_DIR}/{name}_test_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Class\tPrecision\tRecall\tF1\n")
        for i, class_label in enumerate(np.unique(y_test)):
            f.write(f"{class_label}\t{precision[i]:.4f}\t{recall[i]:.4f}\t{f1[i]:.4f}\n")

    print(f"{name} Test Set Accuracy: {acc:.4f}")

        # ================= CONFUSION MATRIX =================
    class_labels = ['No_diabetes', 'Pre_diabetic_or_has_diabetes']
    cm = confusion_matrix(y_test, y_pred_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{name} Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_test_confusion_matrix.png", dpi=300)
    plt.close()

    # ================= FEATURE IMPORTANCE =================
    if name in ["DecisionTree", "RandomForest"]:
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importances)), importances[sorted_idx], color='#4ECDC4', alpha=0.8)
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(X_train.columns[sorted_idx], rotation=90)
        ax.set_ylabel("Importance")
        ax.set_title(f"{name} Feature Importance (Trained on SMOTE)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{name}_feature_importance.png", dpi=300)
        plt.close()

      # ================= DECISION TREE VISUALIZATION =================
    if name == "DecisionTree":
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=X_train.columns,
            class_names=None,           # hide class names
            filled=False,               # no color fill
            rounded=True,
            fontsize=10,
            impurity=False,             # hide Gini index
            proportion=False,           # hide proportions
            label='none',               # hide class labels
            precision=0                 # clean numeric formatting (optional)
        )
        plt.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/{name}_tree.svg", format='svg', dpi=300)
        plt.close()


    # ================= ROC CURVE =================
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_model)[:, 1]
    else:
        y_proba = y_pred_test

    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_roc_curve.png", dpi=300)
    plt.close()

        # ================= METRICS TABLE PNG =================
    if hasattr(model, "predict_proba"):
        y_proba_full = model.predict_proba(X_test_model)[:, 1]
    else:
        y_proba_full = y_pred_test

    metrics_list = compute_metrics(y_test, y_pred_test, y_proba_full)
    overall_acc = accuracy_score(y_test, y_pred_test)
    save_metrics_png(
        metrics_list,
        ['No_diabetes', 'Pre_diabetic_or_has_diabetes'],
        overall_acc,
        f"{OUTPUT_DIR}/{name}_metrics_table.png",
        f"{name} Metrics (Test Set)"
    )

    # ================= LOGISTIC REGRESSION COEFFICIENTS =================
    if name == "LogisticRegression":
        plot_logreg_coefficients(model, X_train.columns, f"{OUTPUT_DIR}/{name}_coefficients.png")


print(f"\n✓ All models trained on SMOTE and evaluated on original test set. Outputs saved in '{OUTPUT_DIR}'")
 