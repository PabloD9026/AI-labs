import pandas as pd
import numpy as np
import joblib
import os
import json
import yaml
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Extract XGBoost configuration
xgb_cfg = params["train_xgboost"]

output_base_dir = xgb_cfg["output_base_dir"]
train_data_dir = xgb_cfg["train_data_dir"]
test_data_dir = xgb_cfg["test_data_dir"]
images_dir = xgb_cfg["images_dir"]  # Pulling directory from params.yaml

# 1. Setup paths
os.makedirs(output_base_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# 2. Load data
train_df = pd.read_csv(train_data_dir)
test_df = pd.read_csv(test_data_dir)

X_train = train_df.drop(columns=['selling_price'])
y_train = train_df['selling_price']
X_test = test_df.drop(columns=['selling_price'])
y_test = test_df['selling_price']

# 3. Train Model
model = XGBRegressor(
    n_estimators=xgb_cfg["n_estimators"],
    learning_rate=xgb_cfg["learning_rate"],
    max_depth=xgb_cfg["max_depth"],
    subsample=xgb_cfg["subsample"],
    colsample_bytree=xgb_cfg["colsample_bytree"],
    random_state=42
)

model.fit(X_train, y_train)

# 4. Predict and Evaluate
y_pred = model.predict(X_test)
metrics = {
    "MAE": float(mean_absolute_error(y_test, y_pred)),
    "MSE": float(mean_squared_error(y_test, y_pred)),
    "R2": float(r2_score(y_test, y_pred)),
}

# 5. Visualizations - Feature Importance
importances = model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title('XGBoost Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='orange', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance Score')
plt.tight_layout()

# Save to the directory specified in params.yaml
plt.savefig(os.path.join(images_dir, "Feature_importance.png"))
plt.close()

# 6. Save Model and Metrics
joblib.dump(model, os.path.join(output_base_dir, 'xgboost_model.joblib'))

with open(os.path.join(output_base_dir, 'metrics_xgboost.json'), 'w') as f:
    json.dump(metrics, f, indent=4)
