import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
from collections import Counter
import os

# --- Parameters ---
sample_size = 50000  # subset for RFECV speed
cv_splits = 3
output_dir = "feature_selection_images2"
os.makedirs(output_dir, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("final_cleaned.csv")
y = df['Diabetes_binary'].map({'No_diabetes':0, 'Pre_diabetic_or_has_diabetes':1})
X = df.drop('Diabetes_binary', axis=1)

# --- Ordinal encoding ---
ordinal_mapping = {
    'BMI': ['Healthy Weight', 'Overweight', 'Class 1 Obesity', 'Class 2 Obesity', 'Class 3 Obesity'],
    'Age': ['Age_18_to_24','Age_25_to_29','Age_30_to_34','Age_35_to_39','Age_40_to_44',
            'Age_45_to_49','Age_50_to_54','Age_55_to_59','Age_60_to_64','Age_65_to_69',
            'Age_70_to_74','Age_75_to_79','Age_80_or_Older'],
    'Education': ['Never_Attended','Grade_1_to_8','Grade_9_to_11','Grade_12_or_GED',
                  'College_1_to_3_years','College_4_or_more_years'],
    'Income': ['less_than_10K','10K_to_less_than_15K','15K_to_less_than_20K','20K_to_less_than_25K',
               '25K_to_less_than_35K','35K_to_less_than_50K','50K_to_less_than_75K','more_than_75K']
}
for col, order in ordinal_mapping.items():
    X[col] = X[col].astype(pd.CategoricalDtype(categories=order, ordered=True)).cat.codes

# --- Label encode remaining categorical columns ---
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes

# --- CV and scorer ---
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score)

# --- Helper to save charts ---
def save_feature_chart(features, values, method_name):
    plt.figure(figsize=(12,6))
    values.plot(kind='bar', color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Importance / Score")
    plt.title(f"Top Features by {method_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method_name}_top_features.png"))
    plt.close()

# --- 1️⃣ RFECV (subset + F1-score) ---
if len(df) > sample_size:
    df_sample = pd.concat([
        df[y==1].sample(n=int(sample_size * sum(y==1)/len(df)), random_state=42),
        df[y==0].sample(n=int(sample_size * sum(y==0)/len(df)), random_state=42)
    ])
else:
    df_sample = df.copy()

y_sample = df_sample['Diabetes_binary'].map({'No_diabetes':0, 'Pre_diabetic_or_has_diabetes':1})
X_sample = df_sample.drop('Diabetes_binary', axis=1)

# Re-encode ordinals and categories for sample
for col, order in ordinal_mapping.items():
    if col in X_sample.columns:
        X_sample[col] = X_sample[col].astype(pd.CategoricalDtype(categories=order, ordered=True)).cat.codes
for col in X_sample.select_dtypes(include='object').columns:
    X_sample[col] = X_sample[col].astype('category').cat.codes

rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
rfecv = RFECV(estimator=rf_model, cv=cv, scoring=f1_scorer, step=1, min_features_to_select=5)
rfecv.fit(X_sample, y_sample)
rfecv_features = X_sample.columns[rfecv.support_].tolist()

# --- 2️⃣ Mutual Information ---
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_features = X.columns[mi_scores > 0.01].tolist()
mi_importances = pd.Series(mi_scores, index=X.columns)
save_feature_chart(mi_features, mi_importances[mi_features], "Mutual_Info")

# --- 3️⃣ Pearson correlation ---
pearson_corr = X.corrwith(y).abs()
pearson_features = pearson_corr[pearson_corr > 0.1].index.tolist()
save_feature_chart(pearson_features, pearson_corr[pearson_features], "Pearson")

# --- Combine all selections ---
all_features = rfecv_features + mi_features + pearson_features
counter = Counter(all_features)
final_features = [feat for feat, count in counter.items() if count >= 2]  # selected by all 3 methods

# --- Map back to original attributes ---
def map_to_original(feature_list):
    original_features = set()
    for f in feature_list:
        if '_' in f and f not in ordinal_mapping:
            original_features.add(f.split('_')[0])
        else:
            original_features.add(f)
    return sorted(original_features)

final_original_features = map_to_original(final_features)

# --- Evaluate final features ---
final_acc = cross_val_score(rf_model, X[final_features], y, cv=cv, scoring='accuracy').mean()
final_f1 = cross_val_score(rf_model, X[final_features], y, cv=cv, scoring=f1_scorer).mean()

# --- Save combined frequency chart ---
feature_freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['Frequency']).sort_values(by='Frequency', ascending=False)
plt.figure(figsize=(14,6))
plt.bar(feature_freq_df.index, feature_freq_df['Frequency'], color='coral')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Methods Selecting Feature")
plt.title("Feature Selection Frequency Across Methods (RFECV, MI, Pearson)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_selection_frequency_all_methods.png"))
plt.close()

# --- Output ---
print("RFECV features:", rfecv_features)
print("Mutual Info features:", mi_features)
print("Pearson features:", pearson_features)
print("\nFinal robust features (encoded):", final_features)
print("Final robust attributes (original):", final_original_features)
print(f"Final Accuracy/F1 on full dataset: {final_acc:.3f}/{final_f1:.3f}")

# --- Save final original attributes ---
pd.DataFrame({'Final_Selected_Attributes': final_original_features}).to_csv(
    os.path.join(output_dir, "final_selected_original_attributes.csv"), index=False
)
