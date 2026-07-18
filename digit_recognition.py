"""
============================================================
  Handwritten Digit Recognition Using Deep Learning
  Models: CNN | MLP Neural Network | SVM | KNN
  Dataset: MNIST (70,000 samples, 28x28 grayscale)
============================================================
  Note: SVM and KNN use a 15k stratified subset of the
  training set. RBF-SVM and KNN are O(n^2/n^3) and become
  impractically slow on the full 60k. CNN and MLP use the
  full 60k training set.
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import time

# ──────────────────────────────────────────────────────────
# STYLING CONSTANTS
# ──────────────────────────────────────────────────────────
BG    = '#0d0d0d'
PANEL = '#1a1a2e'
COLORS = {'CNN': '#a29bfe', 'MLP': '#ff6b35', 'SVM': '#4ecdc4', 'KNN': '#ffe66d'}

print("=" * 60)
print("  HANDWRITTEN DIGIT RECOGNITION — MNIST")
print("=" * 60)

# ──────────────────────────────────────────────────────────
# 1. LOAD MNIST DATASET
# ──────────────────────────────────────────────────────────
print("\n[1/6] Loading MNIST dataset...")
(X_train_raw, y_train_full), (X_test_raw, y_test_full) = mnist.load_data()

# Flatten 28x28 -> 784 and normalize for classical ML
X_train_flat = X_train_raw.reshape(-1, 784) / 255.0
X_test_flat  = X_test_raw.reshape(-1, 784) / 255.0

print(f"      Train samples : {X_train_flat.shape[0]}")
print(f"      Test  samples : {X_test_flat.shape[0]}")
print(f"      Image size    : 28x28 = 784 features")
print(f"      Classes       : {np.unique(y_train_full)}")

# 15k stratified subset for SVM and KNN
X_sub, _, y_sub, _ = train_test_split(
    X_train_flat, y_train_full,
    train_size=15000, stratify=y_train_full, random_state=42
)

# ──────────────────────────────────────────────────────────
# 2. SAMPLE DIGIT GRID
# ──────────────────────────────────────────────────────────
print("\n[2/6] Saving sample digit grid...")

fig, axes = plt.subplots(4, 10, figsize=(14, 6))
fig.suptitle("MNIST — Sample Digits (0-9)", fontsize=15,
             fontweight='bold', color='white', y=1.01)
fig.patch.set_facecolor(BG)

for digit in range(10):
    idx = np.where(y_train_full == digit)[0][:4]
    for row, i in enumerate(idx):
        ax = axes[row, digit]
        ax.imshow(X_train_raw[i], cmap='inferno', interpolation='nearest')
        ax.axis('off')
        if row == 0:
            ax.set_title(str(digit), color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved -> sample_digits.png")

# ──────────────────────────────────────────────────────────
# 3. TRAIN ALL MODELS
# ──────────────────────────────────────────────────────────
print("\n[3/6] Training models...\n")
results = {}

# ── CNN ──────────────────────────────────────────────────
print("  > CNN (Convolutional Neural Network) — full 60k")
X_train_cnn = X_train_raw.reshape(-1, 28, 28, 1) / 255.0
X_test_cnn  = X_test_raw.reshape(-1, 28, 28, 1) / 255.0
y_train_cat = to_categorical(y_train_full, 10)
y_test_cat  = to_categorical(y_test_full, 10)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

t0 = time.time()
cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cat,
    epochs=5, batch_size=128,
    validation_split=0.1, verbose=1
)
cnn_time = time.time() - t0

_, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn, verbose=0), axis=1)
results['CNN'] = {'acc': cnn_acc, 'time': cnn_time, 'preds': y_pred_cnn}

cnn_model.save('cnn_model.h5')
print(f"      Accuracy : {cnn_acc*100:.2f}%  |  Time : {cnn_time:.1f}s")
print("      Saved -> cnn_model.h5")

# ── MLP ──────────────────────────────────────────────────
print("\n  > MLP Neural Network — full 60k")
scaler_mlp  = StandardScaler()
X_tr_mlp    = scaler_mlp.fit_transform(X_train_flat)
X_te_mlp    = scaler_mlp.transform(X_test_flat)

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu', solver='adam',
    max_iter=30, learning_rate_init=0.001,
    batch_size=128, random_state=42, verbose=False
)
t0 = time.time()
mlp.fit(X_tr_mlp, y_train_full)
mlp_time = time.time() - t0

y_pred_mlp = mlp.predict(X_te_mlp)
mlp_acc    = accuracy_score(y_test_full, y_pred_mlp)
results['MLP'] = {'acc': mlp_acc, 'time': mlp_time, 'preds': y_pred_mlp}
print(f"      Accuracy : {mlp_acc*100:.2f}%  |  Time : {mlp_time:.1f}s")

# ── SVM ──────────────────────────────────────────────────
print("\n  > SVM (RBF kernel) — 15k subset")
scaler_svm = StandardScaler()
X_tr_svm   = scaler_svm.fit_transform(X_sub)
X_te_svm   = scaler_svm.transform(X_test_flat)

svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
t0 = time.time()
svm.fit(X_tr_svm, y_sub)
svm_time = time.time() - t0

y_pred_svm = svm.predict(X_te_svm)
svm_acc    = accuracy_score(y_test_full, y_pred_svm)
results['SVM'] = {'acc': svm_acc, 'time': svm_time, 'preds': y_pred_svm}
print(f"      Accuracy : {svm_acc*100:.2f}%  |  Time : {svm_time:.1f}s")

# ── KNN ──────────────────────────────────────────────────
print("\n  > KNN (k=5) — 15k subset")
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
t0 = time.time()
knn.fit(X_sub, y_sub)
knn_time = time.time() - t0

y_pred_knn = knn.predict(X_test_flat)
knn_acc    = accuracy_score(y_test_full, y_pred_knn)
results['KNN'] = {'acc': knn_acc, 'time': knn_time, 'preds': y_pred_knn}
print(f"      Accuracy : {knn_acc*100:.2f}%  |  Time : {knn_time:.1f}s")

# ──────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ──────────────────────────────────────────────────────────
print("\n[4/6] Generating evaluation plots...")

# ── Confusion Matrices (4 panels) ─────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Confusion Matrices — All Models",
             fontsize=15, fontweight='bold', color='white', y=1.02)

for ax, (name, res) in zip(axes, results.items()):
    cm   = confusion_matrix(y_test_full, res['preds'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=np.arange(10))
    disp.plot(ax=ax, colorbar=False, cmap='YlOrRd')
    ax.set_title(f"{name}  ({res['acc']*100:.2f}%)",
                 color=COLORS[name], fontsize=12, fontweight='bold')
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved -> confusion_matrices.png")

# ── Accuracy & Training Time Comparison ───────────────────
fig = plt.figure(figsize=(14, 5), facecolor=BG)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

names = list(results.keys())
accs  = [results[n]['acc'] * 100 for n in names]
times = [results[n]['time'] for n in names]
cols  = [COLORS[n] for n in names]

ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(PANEL)
bars = ax1.bar(names, accs, color=cols, edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom',
             color='white', fontsize=11, fontweight='bold')
ax1.set_ylim(min(accs) - 3, 101)
ax1.set_title('Model Accuracy Comparison', color='white',
              fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color='white')
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#333')
ax1.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')

ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(PANEL)
bars2 = ax2.bar(names, times, color=cols, edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars2, times):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f}s', ha='center', va='bottom',
             color='white', fontsize=11, fontweight='bold')
ax2.set_title('Training Time Comparison', color='white',
              fontsize=13, fontweight='bold')
ax2.set_ylabel('Time (seconds)', color='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#333')
ax2.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')

fig.suptitle('Performance Overview — All Models',
             color='white', fontsize=15, fontweight='bold')
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved -> model_comparison.png")

# ── CNN Training History (Accuracy + Loss) ────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

for ax in [ax1, ax2]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')

ax1.plot(cnn_history.history['accuracy'],     color='#a29bfe', linewidth=2, label='Train')
ax1.plot(cnn_history.history['val_accuracy'], color='#fd79a8', linewidth=2, label='Validation')
ax1.set_title('CNN Accuracy per Epoch', color='white', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', color='white')
ax1.set_ylabel('Accuracy', color='white')
ax1.legend(facecolor=PANEL, labelcolor='white')

ax2.plot(cnn_history.history['loss'],     color='#ff6b35', linewidth=2, label='Train')
ax2.plot(cnn_history.history['val_loss'], color='#ffeaa7', linewidth=2, label='Validation')
ax2.set_title('CNN Loss per Epoch', color='white', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch', color='white')
ax2.set_ylabel('Loss', color='white')
ax2.legend(facecolor=PANEL, labelcolor='white')

fig.suptitle('CNN Training History', color='white',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved -> cnn_training_history.png")

# ── CNN Sample Predictions ────────────────────────────────
fig, axes = plt.subplots(3, 10, figsize=(16, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('CNN Predictions — Green=Correct | Red=Wrong',
             color='white', fontsize=13, fontweight='bold')

for col in range(10):
    digit_idxs = np.where(y_test_full == col)[0][:3]
    for row, idx in enumerate(digit_idxs):
        ax   = axes[row, col]
        pred = y_pred_cnn[idx]
        true = y_test_full[idx]
        ax.imshow(X_test_raw[idx], cmap='inferno', interpolation='nearest')
        ax.axis('off')
        color = '#00ff88' if pred == true else '#ff4455'
        label = f'P:{pred}' if pred == true else f'P:{pred}\nT:{true}'
        ax.set_title(label, fontsize=7, color=color, pad=1)

plt.tight_layout()
plt.savefig('cnn_predictions.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved -> cnn_predictions.png")

# ──────────────────────────────────────────────────────────
# 5. CLASSIFICATION REPORTS
# ──────────────────────────────────────────────────────────
print("\n[5/6] Classification Reports")
print("=" * 60)
for name, res in results.items():
    print(f"\n  +-- {name} -----------------------------------------------")
    print(classification_report(y_test_full, res['preds'],
                                 target_names=[str(i) for i in range(10)],
                                 zero_division=0))

# ──────────────────────────────────────────────────────────
# 6. FINAL SUMMARY
# ──────────────────────────────────────────────────────────
print("\n[6/6] Final Summary")
print("=" * 60)
print(f"  {'Model':<8}  {'Accuracy':>10}  {'Train Time':>12}  {'Dataset':>15}")
print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*15}")
dataset_note = {'CNN': 'Full 60k', 'MLP': 'Full 60k', 'SVM': '15k subset', 'KNN': '15k subset'}
for name, res in results.items():
    print(f"  {name:<8}  {res['acc']*100:>9.2f}%  {res['time']:>10.1f}s  {dataset_note[name]:>15}")

best = max(results, key=lambda n: results[n]['acc'])
print(f"\n  Best Model : {best} ({results[best]['acc']*100:.2f}%)")
print("=" * 60)
print("\nAll files saved to current folder.")