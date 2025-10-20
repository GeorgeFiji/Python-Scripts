import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# ================== CONFIG ==================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
OUTPUT_DIR = "smote_eleven"  # folder for plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== LOAD DATA ==================
df = pd.read_csv("11selected.csv")

# ================== SEPARATE FEATURES AND TARGET ==================
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# ================== ENCODE CATEGORICAL VARIABLES ==================
label_encoders = {}
X_encoded = X.copy()
for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

# Encode target if it's text
target_encoder = None
if y.dtype == 'object':
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

# ================== TRAIN/TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print("Before SMOTE:", Counter(y_train))

# ================== APPLY SMOTE ==================
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_res))

# ================== PREPARE LABELS FOR PLOTS ==================
before_smote = Counter(y_train)
after_smote = Counter(y_res)
classes_before = list(before_smote.keys())
counts_before = list(before_smote.values())
classes_after = sorted(after_smote.keys())
counts_after = [after_smote[c] for c in classes_after]

if target_encoder is not None:
    class_labels = target_encoder.inverse_transform(classes_before)
else:
    class_labels = classes_before

colors = ['#FF6B6B', '#4ECDC4']

# ================== SAVE PLOTS ==================

# 1. Bar - Before SMOTE
fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(range(len(class_labels)), counts_before, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(class_labels)))
ax.set_xticklabels(class_labels, rotation=45, ha='right')
ax.set_ylabel('Samples')
ax.set_title('Class Distribution Before SMOTE')
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{counts_before[i]}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_before_smote.png', dpi=300)
plt.close()

# 2. Bar - After SMOTE
fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(range(len(class_labels)), counts_after, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(class_labels)))
ax.set_xticklabels(class_labels, rotation=45, ha='right')
ax.set_ylabel('Samples')
ax.set_title('Class Distribution After SMOTE')
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{counts_after[i]}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_after_smote.png', dpi=300)
plt.close()

# 3. Side-by-side comparison
fig, ax = plt.subplots(figsize=(10,6))
x_pos = np.arange(len(class_labels))
width = 0.35
ax.bar(x_pos - width/2, counts_before, width, label='Before', color='#FF6B6B', alpha=0.8)
ax.bar(x_pos + width/2, counts_after, width, label='After', color='#4ECDC4', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(class_labels, rotation=45, ha='right')
ax.set_ylabel('Samples')
ax.set_title('Before vs After SMOTE')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_before_vs_after.png', dpi=300)
plt.close()

# 4. Pie - Before SMOTE
fig, ax = plt.subplots(figsize=(10,8))
ax.pie(counts_before, labels=class_labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Pie Chart - Before SMOTE')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_pie_before_smote.png', dpi=300)
plt.close()

# 5. Pie - After SMOTE
fig, ax = plt.subplots(figsize=(10,8))
ax.pie(counts_after, labels=class_labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Pie Chart - After SMOTE')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_pie_after_smote.png', dpi=300)
plt.close()

# 6. Percentage change
fig, ax = plt.subplots(figsize=(10,6))
pct_change = [(counts_after[i]-counts_before[i])/counts_before[i]*100 for i in range(len(counts_before))]
colors_change = ['green' if x>0 else 'red' for x in pct_change]
ax.bar(range(len(class_labels)), pct_change, color=colors_change)
ax.set_xticks(range(len(class_labels)))
ax.set_xticklabels(class_labels, rotation=45, ha='right')
ax.set_ylabel('% Change')
ax.set_title('Percentage Change After SMOTE')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_percentage_change.png', dpi=300)
plt.close()

# 7. Cumulative distribution
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(range(len(class_labels)), np.cumsum(counts_before), marker='o', label='Before', color='#FF6B6B')
ax.plot(range(len(class_labels)), np.cumsum(counts_after), marker='s', label='After', color='#4ECDC4')
ax.set_xticks(range(len(class_labels)))
ax.set_xticklabels(class_labels, rotation=45, ha='right')
ax.set_ylabel('Cumulative Samples')
ax.set_title('Cumulative Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_cumulative_distribution.png', dpi=300)
plt.close()

# 8. Class balance improvement ratio
fig, ax = plt.subplots(figsize=(10,6))
ratio_before = counts_before[1]/counts_before[0]
ratio_after = counts_after[1]/counts_after[0]
ax.bar(['Before', 'After'], [ratio_before, ratio_after], color=['#FF6B6B','#4ECDC4'])
ax.axhline(1.0, linestyle='--', color='green', label='Perfect Balance')
ax.set_ylabel('Minority/Majority Ratio')
ax.set_title('Class Balance Improvement')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_class_balance_ratio.png', dpi=300)
plt.close()

# 9. Training vs Test counts
fig, ax = plt.subplots(figsize=(10,6))
train_counts_before = list(Counter(y_train).values())
test_counts = list(Counter(y_test).values())
labels = class_labels
x_pos = np.arange(len(labels))
width = 0.35
ax.bar(x_pos - width/2, train_counts_before, width, label='Train (before SMOTE)', color='#FF6B6B', alpha=0.8)
ax.bar(x_pos + width/2, test_counts, width, label='Test', color='#4ECDC4', alpha=0.8)
for bar, count in zip(ax.patches[:len(train_counts_before)], train_counts_before):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count}', ha='center', va='bottom')
for bar, count in zip(ax.patches[len(train_counts_before):], test_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count}', ha='center', va='bottom')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Samples')
ax.set_title('Training vs Test Class Counts')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_training_vs_test_counts.png', dpi=300)
plt.close()

# ================== SAVE RESAMPLED & TEST DATASETS ==================
OUTPUT_CSV_DIR = "smote_combined_visualizationall"
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# Decode the resampled data back to original format
X_res_decoded = pd.DataFrame(X_res, columns=X_encoded.columns)
for col in X_res_decoded.columns:
    if col in label_encoders:
        X_res_decoded[col] = label_encoders[col].inverse_transform(X_res_decoded[col].astype(int))

if target_encoder is not None:
    y_res_decoded = target_encoder.inverse_transform(y_res)
else:
    y_res_decoded = y_res

X_res_decoded['Diabetes_binary'] = y_res_decoded

# Decode test set back to original format
X_test_decoded = pd.DataFrame(X_test, columns=X_encoded.columns)
for col in X_test_decoded.columns:
    if col in label_encoders:
        X_test_decoded[col] = label_encoders[col].inverse_transform(X_test_decoded[col].astype(int))

if target_encoder is not None:
    y_test_decoded = target_encoder.inverse_transform(y_test)
else:
    y_test_decoded = y_test

X_test_decoded['Diabetes_binary'] = y_test_decoded

# Save datasets
X_res_decoded.to_csv(f'{OUTPUT_CSV_DIR}/X_train_smote.csv', index=False)
X_test_decoded.to_csv(f'{OUTPUT_CSV_DIR}/X_test_original.csv', index=False)
X_res_decoded.drop('Diabetes_binary', axis=1).to_csv(f'{OUTPUT_CSV_DIR}/X_train_smote_features.csv', index=False)
pd.DataFrame(y_res_decoded, columns=['Diabetes_binary']).to_csv(f'{OUTPUT_CSV_DIR}/y_train_smote.csv', index=False)

print(f"\n✓ SMOTE datasets saved in '{OUTPUT_CSV_DIR}'")
print(f"✓ All visualizations saved in '{OUTPUT_DIR}'")
