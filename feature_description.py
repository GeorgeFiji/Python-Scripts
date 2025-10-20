import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_path = "final_cleaned.csv"
df = pd.read_csv(file_path)

# Create output folder
output_dir = "feature_images"
os.makedirs(output_dir, exist_ok=True)

# Features where x-axis labels may cluster
rotate_labels_features = ['Income', 'BMI', 'Education', 'Age']

for col in df.columns:
    plt.figure(figsize=(8,5))  # Increase figure size for clarity
    if df[col].dtype in ['int64', 'float64']:
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f"Distribution of {col}")
    else:
        sns.countplot(x=col, data=df, palette='viridis')
        plt.title(f"Counts of {col}")
        # Rotate x-axis labels if needed
        if col in rotate_labels_features:
            plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{col}.png"))
    plt.close()

print(f"Images saved in folder: {output_dir}")
