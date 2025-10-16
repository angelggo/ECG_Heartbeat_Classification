# ==========================================================
# Explainable and Reliable AI for ECG Heartbeat Classification
# CNN + BiLSTM + Attention + SHAP explainability
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import shap

# ==========================================================
# 1. DOWNLOAD DATASET FROM KAGGLE (optional: skip if already downloaded)
# ==========================================================
# Dataset: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
# Place mitbih_train.csv and mitbih_test.csv in your working directory

train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df = pd.read_csv("mitbih_test.csv", header=None)

# ==========================================================
# 2. LOAD AND PREPROCESS THE DATASET
# ==========================================================
print("Sample data rows:")
print(train_df.head())

# Convert to binary classification: Normal(0) vs Abnormal(1)
X_train = train_df.iloc[:, :-1].values
y_train = (train_df.iloc[:, -1] != 0).astype(int).values
X_test = test_df.iloc[:, :-1].values
y_test = (test_df.iloc[:, -1] != 0).astype(int).values

# Visualize label distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title('Training Labels (0=Normal, 1=Abnormal)')
sns.countplot(x=y_test, ax=axes[1])
axes[1].set_title('Testing Labels (0=Normal, 1=Abnormal)')
plt.show()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==========================================================
# 3. NORMALIZE DATA
# ==========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for Conv1D: (samples, timesteps, features)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print(f"Reshaped data: {X_train.shape}")

# ==========================================================
# 4. DEFINE ATTENTION LAYER
# ==========================================================
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(input_shape[-1],),
            initializer="random_normal",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        att_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        weighted_output = tf.reduce_sum(inputs * tf.expand_dims(att_weights, -1), axis=1)
        return weighted_output

# ==========================================================
# 5. BUILD CNN + BiLSTM + ATTENTION MODEL
# ==========================================================
def build_explainable_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # CNN feature extraction
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)

    # BiLSTM temporal modeling
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # Attention for interpretability
    x = AttentionLayer()(x)

    # Dense classifier
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_explainable_model(X_train.shape[1:])
model.summary()

# ==========================================================
# 6. TRAIN MODEL
# ==========================================================
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# ==========================================================
# 7. PLOT TRAINING HISTORY
# ==========================================================
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='x')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.show()

# ==========================================================
# 8. EVALUATE ON TEST DATA
# ==========================================================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Test Recall: {recall_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal'])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.show()

# ==========================================================
# 9. EXPLAINABILITY WITH SHAP
# ==========================================================
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:5])

shap.initjs()
shap.summary_plot(shap_values[0], X_test[:5].reshape(5, -1),
                  feature_names=[f"Signal_{i}" for i in range(X_test.shape[1])])

# Visualize SHAP for one ECG
plt.figure(figsize=(15, 6))
idx = 0
plt.plot(X_test[idx].squeeze(), label='ECG Signal', color='blue')
shap_val = shap_values[0][idx]
shap_val_scaled = 5 * (shap_val - np.min(shap_val)) / (np.max(shap_val) - np.min(shap_val))
plt.fill_between(range(len(shap_val_scaled)), 0, shap_val_scaled, color='red', alpha=0.3, label='SHAP Values')
plt.xlabel('Time-step'); plt.ylabel('Normalized Amplitude')
plt.title('ECG Signal with SHAP Explanation')
plt.legend(); plt.show()

# ==========================================================
# 10. SAVE MODEL
# ==========================================================
model.save('ecg_heartbeat_model.h5')
print("✅ Model saved to 'ecg_heartbeat_model.h5'")
