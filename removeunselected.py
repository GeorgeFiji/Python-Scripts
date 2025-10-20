import pandas as pd

# --- Step 1: Load your dataset ---
file_path = "final_cleaned.csv"  # Replace with your CSV file path
df = pd.read_csv(file_path)

# --- Step 2: Specify columns to keep ---
columns_to_keep = [
   
    "Diabetes_binary", #leave this here
    
     'Age', 'BMI', 'DiffWalk', 'Education', 'Fruits', 'GenHlth', 'HeartDiseaseorAttack', 'HighBP', 'HighChol', 'Income', 'PhysHlth'
]

# --- Step 3: Keep only the specified columns ---
df = df[columns_to_keep]

# --- Step 4: Save the filtered dataset (optional) ---
output_file = "11selected.csv"
df.to_csv(output_file, index=False)

print(f"Columns kept: {columns_to_keep}")
print(f"Filtered dataset saved to {output_file}")
