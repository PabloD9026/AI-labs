import pandas as pd
import numpy as np
import joblib
import os
import json
import yaml
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Extract Decision Tree configuration
tree_cfg = params["train_tree"]

output_base_dir = tree_cfg["output_base_dir"]
train_data_dir = tree_cfg["train_data_dir"]
test_data_dir = tree_cfg["test_data_dir"]

# Hyperparameters
criterion = tree_cfg["criterion"]
max_depth = tree_cfg["max_depth"]
min_samples_split = tree_cfg["min_samples_split"]
min_samples_leaf = tree_cfg["min_samples_leaf"]
output_levels = tree_cfg.get("output_levels", 3)  # Default to 3 if not found
images_dir = tree_cfg["images_dir"]
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
model = DecisionTreeRegressor(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
model.fit(X_train, y_train)

# 4. Predict and Evaluate
y_pred = model.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),
}

# 5. Visualizations

# --- A. Full Tree Plot ---
plt.figure(figsize=(25, 15))
plot_tree(model, 
          feature_names=X_train.columns.tolist(), 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title(f"Decision Tree (Full - Depth {max_depth})")
plt.savefig(os.path.join(images_dir, "Tree_model_full.png"), bbox_inches='tight')
plt.close()

# --- B. Shallow Tree Plot ---
plt.figure(figsize=(20, 10))
plot_tree(model, 
          max_depth=output_levels, 
          feature_names=X_train.columns.tolist(), 
          filled=True, 
          rounded=True, 
          fontsize=12)
plt.title(f"Decision Tree (Shallow - Depth {output_levels})")
plt.savefig(os.path.join(images_dir, "Tree_model_shallow.png"), bbox_inches='tight')
plt.close()

# --- C. Feature Importance Bar Chart ---
importances = model.feature_importances_
indices = np.argsort(importances)
features = X_train.columns

plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "Feature_importance.png"))
plt.close()

# 6. Save Model and Metrics
joblib.dump(model, os.path.join(output_base_dir, 'tree_model.joblib'))

with open(os.path.join(output_base_dir, 'metrics_tree.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

