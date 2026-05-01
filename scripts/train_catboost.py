import pandas as pd
import numpy as np
import joblib
import os
import json
import yaml
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Extract CatBoost configuration
cb_cfg = params["train_catboost"]

output_base_dir = cb_cfg["output_base_dir"]
train_data_dir = cb_cfg["train_data_dir"]
test_data_dir = cb_cfg["test_data_dir"]
images_dir = cb_cfg["images_dir"]


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
# Initializing CatBoost with the 5 most important params from params.yaml
model = CatBoostRegressor(
    iterations=cb_cfg["iterations"],
    learning_rate=cb_cfg["learning_rate"],
    depth=cb_cfg["depth"],
    l2_leaf_reg=cb_cfg["l2_leaf_reg"],
    loss_function=cb_cfg["loss_function"],
    random_seed=42,
    verbose=False  # Keeps the console clean
)

model.fit(X_train, y_train)

# 4. Predict and Evaluate
y_pred = model.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),
}

# 5. Visualizations - Feature Importance
importances = model.get_feature_importance()
feature_names = X_train.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title('CatBoost Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='green', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "Feature_importance.png"))
plt.close()

# 6. Save Model and Metrics
joblib.dump(model, os.path.join(output_base_dir, 'catboost_model.joblib'))

with open(os.path.join(output_base_dir, 'metrics_catboost.json'), 'w') as f:
    json.dump(metrics, f, indent=4)
    