import pandas as pd
import numpy as np
import os
import json
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train_nn"]

# Setup Directories
os.makedirs(params["output_base_dir"], exist_ok=True)
os.makedirs(params["images_dir"], exist_ok=True)
log_subdir = os.path.join(params["log_dir"], datetime.now().strftime("%Y%m%d-%H%M%S"))

# 1. Load data
train_df = pd.read_csv(params["train_data_dir"])
test_df = pd.read_csv(params["test_data_dir"])

X_train = train_df.drop(columns=['selling_price']).values
y_train = train_df['selling_price'].values
X_test = test_df.drop(columns=['selling_price']).values
y_test = test_df['selling_price'].values

# 2. Build Dynamic Model
model = models.Sequential()
model.add(layers.Input(shape=(X_train.shape[1],)))

# Add hidden layers based on params
num_layers = params["amount_of_hidden_layers"]
for i in range(1, num_layers + 1):
    model.add(layers.Dense(
        params[f"neurons_l{i}"], 
        activation=params[f"activation_l{i}"],
        kernel_initializer=params["kernel_initializer"]
    ))
    if params["dropout_rate"] > 0:
        model.add(layers.Dropout(params["dropout_rate"]))

# Output layer for regression
model.add(layers.Dense(1))

# Optimizer
if params["optimizer"] == "adam":
    opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
else:
    opt = tf.keras.optimizers.RMSprop(learning_rate=params["learning_rate"])

model.compile(optimizer=opt, loss=params["loss_function"], metrics=['mae'])

# 3. Callbacks
tensorboard_callback = callbacks.TensorBoard(log_dir=log_subdir, histogram_freq=1)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=params["patience"], restore_best_weights=True)

# 4. Train
history = model.fit(
    X_train, y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    validation_split=params["validation_split"],
    callbacks=[tensorboard_callback, early_stop],
    verbose=1
)

# 5. Evaluate
y_pred = model.predict(X_test).flatten()
metrics = {
    "MAE": float(mean_absolute_error(y_test, y_pred)),
    "MSE": float(mean_squared_error(y_test, y_pred)),
    "R2": float(r2_score(y_test, y_pred))
}

# 6. Visualizations

# A. Learning Curves
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Learning Curve (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(params["images_dir"], "learning_curve.png"))
plt.close()

# B. Weight Histograms & Interpretation
# We extract weights from the first and last hidden layers
layers_to_plot = [l for l in model.layers if isinstance(l, layers.Dense)]
fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(15, 4))
fig.suptitle('Weight Histograms (Interpretation: Normal distribution suggests healthy training)')

for i, layer in enumerate(layers_to_plot):
    weights = layer.get_weights()[0].flatten()
    axes[i].hist(weights, bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Layer {i+1}')

plt.tight_layout()
plt.savefig(os.path.join(params["images_dir"], "weight_histograms.png"))
plt.close()

# 7. Save Model, Metrics and Tensorboard Graph
model.save(os.path.join(params["output_base_dir"], 'nn_model.h5'))

with open(os.path.join(params["output_base_dir"], 'metrics_nn.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

