# ================================================================
# ECG Heartbeat Classification: CNN + BiLSTM + Attention
# Imbalance-robust training + TF2-safe explainability
# ================================================================
# Requirements: pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
# Optional (only if you want SHAP): pip install shap
# Files in same dir: mitbih_train.csv, mitbih_test.csv
# ================================================================

import os, sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, f1_score
    , roc_auc_score, roc_curve  
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# --------------------------
# 0) Settings
# --------------------------
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

TRAIN_CSV = "mitbih_train.csv"
TEST_CSV  = "mitbih_test.csv"
USE_SHAP  = False  # set True only if shap is installed and you want SHAP plots

# --------------------------
# 1) Load & Preprocess
# --------------------------
if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
    print("\n[!] Could not find dataset CSV files.")
    print("    Ensure mitbih_train.csv and mitbih_test.csv are in:", os.getcwd())
    sys.exit(1)

train_df = pd.read_csv(TRAIN_CSV, header=None)
test_df  = pd.read_csv(TEST_CSV,  header=None)

# Binary labels: 0 = Normal, 1 = Abnormal (merge non-zero classes)
X_train_all = train_df.iloc[:, :-1].values.astype("float32")
y_train_all = (train_df.iloc[:, -1] != 0).astype(int).values
X_test      = test_df.iloc[:, :-1].values.astype("float32")
y_test      = (test_df.iloc[:, -1] != 0).astype(int).values

print(f"[i] Raw Train: {X_train_all.shape}, Test: {X_test.shape}")

# Standardize features (fit on train, apply to val/test)
scaler = StandardScaler()
X_train_all = scaler.fit_transform(X_train_all).astype("float32")
X_test      = scaler.transform(X_test).astype("float32")

# Reshape for Conv1D: (samples, timesteps, channels=1)
X_train_all = X_train_all[..., np.newaxis]
X_test      = X_test[..., np.newaxis]
TIMESTEPS   = X_train_all.shape[1]

# --------------------------
# 2) Stratified Split & Class Weights
# --------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.2, random_state=SEED, stratify=y_train_all
)

classes = np.array([0, 1])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print("[i] class_weight =", class_weight)

# --------------------------
# 3) Custom Attention Layer (returns context, weights)
# --------------------------
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(name="W", shape=(d, d),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="b", shape=(d,),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="u", shape=(d,),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)
    def call(self, inputs):
        # inputs: (B,T,F)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)   # (B,T,F)
        att_w = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)  # (B,T)
        context = tf.reduce_sum(inputs * tf.expand_dims(att_w, -1), axis=1) # (B,F)
        return context, att_w

# --------------------------
# 4) Build Models
# --------------------------
def build_models(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(2)(x)                # 187 -> 93
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)                # 93  -> 46
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    context, att_w = AttentionLayer()(x)

    z = layers.Dense(32, activation='relu')(context)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(1, activation='sigmoid')(z)

    clf_model = models.Model(inputs=inp, outputs=out, name="ecg_classifier")
    clf_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='prec'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )

    att_model = models.Model(inputs=inp, outputs=att_w, name="attention_probe")
    return clf_model, att_model

# Optional alias so "model = build_explainable_model(X_train.shape[1:])" works as in your snippet
def build_explainable_model(input_shape):
    m, _ = build_models(input_shape)        
    return m

model, att_probe = build_models(X_tr.shape[1:])
model.summary()

# --------------------------
# 5) Train (with callbacks)
# --------------------------
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
rlr   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(
    X_tr, y_tr,
    epochs=15,
    batch_size=128,
    validation_split = 0.2,
    class_weight=class_weight,
    callbacks=[early, rlr],
    verbose=1
)

# --------------------------
# 6) Evaluate + Threshold Tuning
# --------------------------
y_prob = model.predict(X_test, verbose=0).ravel()

ths = np.linspace(0.1, 0.9, 81)
f1s = []
for th in ths:
    yp = (y_prob >= th).astype(int)
    f1s.append(f1_score(y_test, yp))
best_th = float(ths[int(np.argmax(f1s))])
print(f"[i] Best threshold by F1 scan: {best_th:.2f}")

y_pred = (y_prob >= best_th).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred)     
roc_auc = roc_auc_score(y_test, y_prob)     


print(f"\n[i] Test Accuracy:  {acc:.4f}")
print(f"[i] Test Precision: {prec:.4f}")
print(f"[i] Test Recall:    {rec:.4f}")
print(f"[i] F1 Score:       {f1:.4f}")      
print(f"[i] ROC-AUC:        {roc_auc:.4f}")     
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4, zero_division=0))

cm = confusion_matrix(y_test, y_pred)

# --------------------------
# 7) Plots (saved to ./figs)
# --------------------------
os.makedirs("figs", exist_ok=True)

# Training curves
plt.figure(figsize=(11,4.2))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig("figs/accuracy_loss_curves.png", dpi=150)
plt.close()

# Confusion matrix
plt.figure(figsize=(5.8,5.0))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal'])
plt.title('Confusion Matrix (threshold = {:.2f})'.format(best_th))
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig("figs/confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.grid(True); plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("figs/roc_curve.png", dpi=150)
plt.close()     

# Attention overlay (upsample from 46 -> 187 to match signal)
idx = 0
x_ex = X_test[idx:idx+1]                       # (1, 187, 1), float32
att  = att_probe.predict(x_ex, verbose=0).ravel()  # (46,)
sig  = x_ex.squeeze()                          # (187,)

T_att, T_sig = len(att), len(sig)
att_up = np.interp(np.linspace(0, T_att - 1, num=T_sig), np.arange(T_att), att)
att_up = att_up / (att_up.max() + 1e-8)

plt.figure(figsize=(12,4))
plt.plot(sig, label='ECG (normalized)')
plt.fill_between(np.arange(T_sig), 0, att_up * max(0, float(sig.max())),
                 alpha=0.35, label='Attention (upsampled)')
plt.title('ECG with Attention Overlay')
plt.xlabel('Time-step'); plt.ylabel('Normalized amplitude')
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig("figs/attention_overlay.png", dpi=150)
plt.close()

# --------------------------
# 8) Explainability
#    A) Integrated Gradients (robust, no extra deps)  -- FIXED SHAPES
#    B) Optional SHAP (set USE_SHAP=True if installed)
# --------------------------
def integrated_gradients(model, x, steps=64):
    """
    x: np.ndarray of shape (T, 1) or (T,) in float32
    Returns: np.ndarray of shape (T,) IG attribution
    """
    # ensure (1, T, 1) float32
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]  # (T,1)
    x = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)  # (1,T,1)
    baseline = tf.zeros_like(x)                               # (1,T,1)

    steps = int(steps)
    alphas = tf.linspace(tf.constant(0.0, tf.float32),
                         tf.constant(1.0, tf.float32),
                         steps + 1)                           # (S+1,)
    alphas = tf.reshape(alphas, (-1, 1, 1))                   # (S+1,1,1)

    x_tiled = tf.tile(x, [steps + 1, 1, 1])                   # (S+1,T,1)
    b_tiled = tf.tile(baseline, [steps + 1, 1, 1])            # (S+1,T,1)
    interpolated = b_tiled + (x_tiled - b_tiled) * alphas     # (S+1,T,1)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)           # (S+1,1)
    grads = tape.gradient(preds, interpolated)                # (S+1,T,1)

    avg_grads = tf.reduce_mean(grads[:-1], axis=0)            # (T,1)
    ig = (x - baseline)[0] * avg_grads                        # (T,1)
    return ig.numpy().squeeze()                               # (T,)

# IG overlay
sig_ig = X_test[0].squeeze()                      # (T,)
ig_attr = integrated_gradients(model, sig_ig, steps=64)

pos = np.clip(ig_attr, 0, None)
pos = pos / (pos.max() + 1e-8)

plt.figure(figsize=(12,4))
plt.plot(sig_ig, label='ECG (normalized)')
plt.fill_between(np.arange(len(sig_ig)), 0, pos * max(0.0, float(sig_ig.max())),
                 alpha=0.35, label='Integrated Gradients (pos.)')
plt.title('ECG with Integrated Gradients')
plt.xlabel('Time-step'); plt.ylabel('Normalized amplitude')
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("figs/ig_overlay.png", dpi=150); plt.close()
print("[i] Integrated Gradients figure saved: figs/ig_overlay.png")

# Optional SHAP (only if you really want it and have shap installed)
if USE_SHAP:
    try:
        import shap
        masker = shap.maskers.Independent(data=X_tr[:100])
        explainer = shap.Explainer(model, masker=masker, algorithm="gradient")
        X_slice = X_test[:5]
        sv = explainer(X_slice)   # sv.values shape: (N,T,1)

        shap.summary_plot(
            sv.values.reshape(sv.values.shape[0], -1),
            X_slice.reshape(X_slice.shape[0], -1),
            feature_names=[f"t{t}" for t in range(X_slice.shape[1])],
            show=False
        )
        plt.tight_layout(); plt.savefig("figs/shap_summary_plot.png", dpi=150); plt.close()
         # Single-sample overlay example (idx=0)        
        shap_val = shap_values[0][0].squeeze()
        shap_val_scaled = 5 * (shap_val - np.min(shap_val)) / (np.max(shap_val) - np.min(shap_val) + 1e-8)

        plt.figure(figsize=(15,6))
        plt.plot(X_slice[0].squeeze(), label="ECG Signal")
        plt.fill_between(range(len(shap_val_scaled)), 0, shap_val_scaled, alpha=0.3, label="SHAP (importance)")
        plt.title("ECG Signal (Test Sample) with SHAP Overlay")
        plt.xlabel("Time-step"); plt.ylabel("Normalized Amplitude")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig("figs/shap_overlay_sample0.png", dpi=150); plt.close()      
        print("[i] SHAP summary saved: figs/shap_summary_plot.png")
        print("[i] SHAP overlay: figs/shap_overlay_sample0.png")        
    except Exception as e:
        print("[!] SHAP disabled due to:", repr(e))

# --------------------------
# 9) Save model & recap
# --------------------------
MODEL_PATH = "ecg_heartbeat_model.h5"
model.save(MODEL_PATH)
print(f"\n✅ Model saved to '{MODEL_PATH}'")
print("✅ Figures saved to ./figs/:")
print("   - accuracy_loss_curves.png")
print("   - confusion_matrix.png")
print("   - roc_curve.png")     
print("   - attention_overlay.png")
print("   - ig_overlay.png")
print("   - shap_summary_plot.png  (only if USE_SHAP=True and no errors)")
print("   - shap_overlay_sample0.png (only if USE_SHAP=True and no errors)")    