# Run once. Merge all datasets into a single file with a consistent schema. This will create 'merged.csv' in the same directory.
import pandas as pd
import os

os.makedirs('output/prepareOutput', exist_ok=True)

# Load datasets
df = pd.read_csv('datasets/Car_data.csv')
df1 = pd.read_csv('datasets/Car_details_v3.csv')
df2 = pd.read_csv('datasets/CAR_DETAILS_FROM_CAR_DEKHO.csv')

# Rename columns in df to match df2's schema
df_renamed = df.rename(columns={
    'Car_Name': 'name',
    'Year': 'year',
    'Selling_Price': 'selling_price',
    'Kms_Driven': 'km_driven',
    'Fuel_Type': 'fuel',
    'Seller_Type': 'seller_type',
    'Transmission': 'transmission',
    'Owner': 'owner'
})

# Keep only the common columns defined by df2
common_cols = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']

df_renamed = df_renamed[common_cols]
df1_selected = df1[common_cols]
df2_selected = df2[common_cols]   # df2 already has exactly these columns

# Concatenate all rows
merged = pd.concat([df_renamed, df1_selected, df2_selected], ignore_index=True)

# Export to the same directory
merged.to_csv('output/prepareOutput/merged.csv', index=False)

merged_df = pd.read_csv('output/prepareOutput/merged.csv')
print("Dataset shape:", merged_df.shape)

# Encode categorical text into numbers so we can calculate correlations
from sklearn.preprocessing import LabelEncoder

# Make a copy for encoding
df_encoded = merged_df.copy()

# Apply label encoding to all categorical columns
label_encoders = {}
categorical_cols = df_encoded.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

# Save the encoded dataset
df_encoded.to_csv('output/prepareOutput/encoded.csv', index=False)

print("Dataset shape:", df_encoded.shape)

#  normalize the data and shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the encoded dataset
encoded_df = pd.read_csv('output/prepareOutput/encoded.csv')

# Normalize all columns using Min-Max scaling
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(encoded_df)

# Convert back to DataFrame with original column names
normalized_df = pd.DataFrame(normalized_data, columns=encoded_df.columns)

# Save the full normalized dataset
normalized_df.to_csv('output/prepareOutput/normalized.csv', index=False)

# Split into train and test sets (80% train, 20% test) with shuffling
train_df, test_df = train_test_split(normalized_df, test_size=0.2, shuffle=True, random_state=42)

# Save the splits
train_df.to_csv('output/prepareOutput/train.csv', index=False)
test_df.to_csv('output/prepareOutput/test.csv', index=False)

