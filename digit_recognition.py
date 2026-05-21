"""
============================================================
  Handwritten Digit Recognition Using Deep Learning
  Models: MLP Neural Network | SVM | KNN
  Dataset: MNIST
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────
from sklearn.datasets import fetch_openml
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
# 2. LOAD & PREPARE MNIST DATASET
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("  HANDWRITTEN DIGIT RECOGNITION — MNIST")
print("=" * 60)

print("\n[1/5] Loading MNIST dataset (synthetic via sklearn)...")
from sklearn.datasets import load_digits

# sklearn's digits dataset: 1797 samples, 8x8 images (64 features), 10 classes
raw = load_digits()
X_raw = raw.data          # already float64, range 0-16
y     = raw.target.astype(int)

# Upscale to 28x28 via interpolation for realistic simulation
from skimage.transform import resize as sk_resize
X_28 = np.array([
    sk_resize(img.reshape(8, 8), (28, 28), anti_aliasing=True).ravel()
    for img in X_raw
])
X = X_28 / X_28.max()    # normalise to [0, 1]

print(f"      Dataset shape : {X.shape}")
print(f"      Labels shape  : {y.shape}")
print(f"      Classes       : {np.unique(y)}")
print(f"      Note          : Using sklearn digits (8×8 → 28×28 upscaled)")

# Use all available data (no subset needed — dataset is small)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n      Train samples : {X_train.shape[0]}")
print(f"      Test  samples : {X_test.shape[0]}")

# ──────────────────────────────────────────────────────────
# 3. VISUALISE SAMPLE DIGITS
# ──────────────────────────────────────────────────────────
print("\n[2/5] Saving sample digit grid...")

fig, axes = plt.subplots(4, 10, figsize=(14, 6))
fig.suptitle("MNIST — Sample Digits (0–9)", fontsize=15, fontweight='bold', y=1.02)
fig.patch.set_facecolor('#0d0d0d')

for digit in range(10):
    idx = np.where(y_train == digit)[0][:4]
    for row, i in enumerate(idx):
        ax = axes[row, digit]
        ax.imshow(X_train[i].reshape(28, 28), cmap='inferno', interpolation='nearest')
        ax.axis('off')
        if row == 0:
            ax.set_title(str(digit), color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices.png',
            dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print("      Saved → sample_digits.png")

# ──────────────────────────────────────────────────────────
# 4. TRAIN MODELS
# ──────────────────────────────────────────────────────────
print("\n[3/5] Training models...\n")

results = {}

# ── 4a. MLP Neural Network ──────────────────────────────
print("  ► MLP Neural Network")
scaler_mlp = StandardScaler()
X_train_sc = scaler_mlp.fit_transform(X_train)
X_test_sc  = scaler_mlp.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    max_iter=30,
    learning_rate_init=0.001,
    batch_size=128,
    random_state=42,
    verbose=False
)
t0 = time.time()
mlp.fit(X_train_sc, y_train)
mlp_time = time.time() - t0

y_pred_mlp = mlp.predict(X_test_sc)
mlp_acc = accuracy_score(y_test, y_pred_mlp)
results['MLP'] = {'acc': mlp_acc, 'time': mlp_time,
                  'preds': y_pred_mlp, 'model': mlp}
print(f"      Accuracy : {mlp_acc*100:.2f}%  |  Time : {mlp_time:.1f}s")

# ── 4b. SVM ─────────────────────────────────────────────
print("\n  ► Support Vector Machine (SVM)")
scaler_svm = StandardScaler()
X_train_svm = scaler_svm.fit_transform(X_train)
X_test_svm  = scaler_svm.transform(X_test)

svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
t0 = time.time()
svm.fit(X_train_svm, y_train)
svm_time = time.time() - t0

y_pred_svm = svm.predict(X_test_svm)
svm_acc = accuracy_score(y_test, y_pred_svm)
results['SVM'] = {'acc': svm_acc, 'time': svm_time,
                  'preds': y_pred_svm, 'model': svm}
print(f"      Accuracy : {svm_acc*100:.2f}%  |  Time : {svm_time:.1f}s")

# ── 4c. KNN ─────────────────────────────────────────────
print("\n  ► K-Nearest Neighbours (KNN)")
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
t0 = time.time()
knn.fit(X_train, y_train)
knn_time = time.time() - t0

y_pred_knn = knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)
results['KNN'] = {'acc': knn_acc, 'time': knn_time,
                  'preds': y_pred_knn, 'model': knn}
print(f"      Accuracy : {knn_acc*100:.2f}%  |  Time : {knn_time:.1f}s")

# ──────────────────────────────────────────────────────────
# 5. EVALUATION & VISUALISATION
# ──────────────────────────────────────────────────────────
print("\n[4/5] Generating evaluation plots...")

COLORS = {'MLP': '#ff6b35', 'SVM': '#4ecdc4', 'KNN': '#ffe66d'}
BG = '#0d0d0d'
PANEL = '#1a1a2e'

# ── 5a. Confusion Matrices (3-panel) ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Confusion Matrices", fontsize=16, fontweight='bold', color='white', y=1.02)

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['preds'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(ax=ax, colorbar=False, cmap='YlOrRd')
    ax.set_title(f"{name}  ({res['acc']*100:.2f}%)",
                 color=COLORS[name], fontsize=13, fontweight='bold')
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

plt.tight_layout()
plt.savefig('confusion_matrices.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved → confusion_matrices.png")

# ── 5b. Accuracy & Training-Time Comparison ────────────
fig = plt.figure(figsize=(14, 5), facecolor=BG)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

names = list(results.keys())
accs  = [results[n]['acc'] * 100 for n in names]
times = [results[n]['time'] for n in names]
cols  = [COLORS[n] for n in names]

# Accuracy bar chart
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(PANEL)
bars = ax1.bar(names, accs, color=cols, edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', va='bottom', color='white',
             fontsize=11, fontweight='bold')
ax1.set_ylim(min(accs) - 5, 101)
ax1.set_title('Model Accuracy Comparison', color='white', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color='white')
ax1.tick_params(colors='white')
ax1.set_facecolor(PANEL)
for spine in ax1.spines.values():
    spine.set_edgecolor('#333')
ax1.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')

# Training time bar chart
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(PANEL)
bars2 = ax2.bar(names, times, color=cols, edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars2, times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}s', ha='center', va='bottom', color='white',
             fontsize=11, fontweight='bold')
ax2.set_title('Training Time Comparison', color='white', fontsize=13, fontweight='bold')
ax2.set_ylabel('Time (seconds)', color='white')
ax2.tick_params(colors='white')
ax2.set_facecolor(PANEL)
for spine in ax2.spines.values():
    spine.set_edgecolor('#333')
ax2.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')

fig.suptitle('Performance Overview — All Models', color='white',
             fontsize=15, fontweight='bold', y=1.03)
plt.savefig('model_comparison.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved → model_comparison.png")

# ── 5c. MLP Learning Curve ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(PANEL)
ax.plot(mlp.loss_curve_, color='#ff6b35', linewidth=2, label='Training Loss')
ax.set_xlabel('Epoch', color='white')
ax.set_ylabel('Loss', color='white')
ax.set_title('MLP — Training Loss Curve', color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
ax.legend(facecolor=PANEL, labelcolor='white')
ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
plt.tight_layout()
plt.savefig('mlp_loss_curve.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved → mlp_loss_curve.png")

# ── 5d. Sample Predictions (MLP) ───────────────────────
fig, axes = plt.subplots(3, 10, figsize=(16, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('MLP Predictions — Green=Correct | Red=Wrong',
             color='white', fontsize=13, fontweight='bold')

X_test_display = scaler_mlp.inverse_transform(X_test_sc)  # back to normalised scale

for col in range(10):
    digit_idxs = np.where(y_test == col)[0][:3]
    for row, idx in enumerate(digit_idxs):
        ax = axes[row, col]
        ax.imshow(X_test_display[idx].reshape(28, 28), cmap='inferno', interpolation='nearest')
        ax.axis('off')
        pred = y_pred_mlp[idx]
        true = y_test[idx]
        color = '#00ff88' if pred == true else '#ff4455'
        label = f'P:{pred}' if pred == true else f'P:{pred}\nT:{true}'
        ax.set_title(label, fontsize=7, color=color, pad=1)

plt.tight_layout()
plt.savefig('mlp_predictions.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("      Saved → mlp_predictions.png")

# ──────────────────────────────────────────────────────────
# 6. CLASSIFICATION REPORTS
# ──────────────────────────────────────────────────────────
print("\n[5/5] Classification Reports")
print("=" * 60)
for name, res in results.items():
    print(f"\n  ┌─ {name} ─────────────────────────────────────────")
    print(classification_report(y_test, res['preds'], zero_division=0,
                                 target_names=[str(i) for i in range(10)]))

# ──────────────────────────────────────────────────────────
# 7. FINAL SUMMARY TABLE
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Model':<8}  {'Accuracy':>10}  {'Train Time':>12}")
print(f"  {'─'*8}  {'─'*10}  {'─'*12}")
for name, res in results.items():
    print(f"  {name:<8}  {res['acc']*100:>9.2f}%  {res['time']:>10.1f}s")

best = max(results, key=lambda n: results[n]['acc'])
print(f"\n  ✔  Best Model : {best} ({results[best]['acc']*100:.2f}%)")
print("=" * 60)
print("\nAll plots saved to digit_recognition/ folder.")
