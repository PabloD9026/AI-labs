from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

analyze_cfg = params["analyzeData"]

# Extract paths
output_base_dir = analyze_cfg["output_base_dir"]
normalized_data_dir = analyze_cfg["normalized_data_dir"]

os.makedirs(output_base_dir, exist_ok=True)

df = pd.read_csv(normalized_data_dir)

# Encode categorical text into numbers so we can calculate correlations
encoder = LabelEncoder()
categorical_cols =['name','year','selling_price','km_driven','fuel','seller_type','transmission','owner']

df_encoded = df.copy()
for col in categorical_cols:
    if col in df_encoded.columns:
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))

# Plot Correlation Heatmap
plt.figure(figsize=(7, 6))

# SAFETY CHECK: Select ONLY numeric columns before calculating correlation
numeric_df = df_encoded.select_dtypes(include=[np.number])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
plt.savefig(os.path.join(output_base_dir, 'correlation_heatmap.png'))