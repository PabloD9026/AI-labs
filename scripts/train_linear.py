import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Setup paths
train_path = 'output/prepareOutput/train.csv'
test_path = 'output/prepareOutput/test.csv'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# 2. Load data
# We assume the data is already encoded/normalized from your prepare script
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Separate features (X) and target (y)
# Adjust 'selling_price' if your target column name is different
X_train = train_df.drop(columns=['selling_price'])
y_train = train_df['selling_price']
X_test = test_df.drop(columns=['selling_price'])
y_test = test_df['selling_price']

# 3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Determine Weights (Coefficients)
print("--- Linear Regression Weights ---")
weights = pd.DataFrame({'Feature': X_train.columns, 'Weight': model.coef_})
print(weights)
print(f"Intercept: {model.intercept_}")

# 5. Predict and Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "MAE": mae,
    "MSE": mse,
    "R2": r2,
}


print("\n--- Metrics ---")
print(json.dumps(metrics, indent=4))

# 6. Save Model and Metrics
joblib.dump(model, os.path.join(model_dir, 'linear_model.joblib'))

with open('metrics_linear.json', 'w') as f:
    json.dump(metrics, f)

print("\nModel saved to models/linear_model.joblib")