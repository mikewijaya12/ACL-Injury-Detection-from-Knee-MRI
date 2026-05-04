"""
    evaluate.py
    -----------
    Evaluasi lengkap model terbaik (v1) pada test set:
    - Confusion Matrix
    - AUC-ROC per kelas (one-vs-rest)
    - Precision, Recall, F1 per kelas
    - Sensitivity & Specificity per kelas
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders, load_metadata, split_dataset, LABEL_NAMES
from model import HybridCNNViT

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CHECKPOINT = r'D:\ACL\outputs\checkpoints\best_model.pth'
FIGURES    = r'D:\ACL\outputs\figures'
os.makedirs(FIGURES, exist_ok=True)

CLASSES = list(LABEL_NAMES.values())  # ['Healthy', 'Partial Tear', 'Complete Rupture']


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt['config']
    model = HybridCNNViT(
        num_classes = cfg['num_classes'],
        num_slices  = cfg['num_slices'],
        embed_dim   = cfg['embed_dim'],
        num_heads   = cfg['num_heads'],
        num_layers  = cfg['num_layers'],
        dropout     = cfg['dropout'],
        pretrained  = False
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded dari epoch {ckpt['epoch']} (val F1: {ckpt['val_f1']:.4f})")
    return model


# ─── INFERENSI ───────────────────────────────────────────────────────────────
def get_predictions(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


# ─── CONFUSION MATRIX ────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, save_path):
    cm   = confusion_matrix(labels, preds)
    cm_n = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # normalized

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confusion Matrix — Hybrid CNN-ViT (Test Set)',
                 fontsize=13, fontweight='bold')

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Normalized
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Normalized (per kelas aktual)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix tersimpan: {save_path}")
    return cm


# ─── AUC-ROC ─────────────────────────────────────────────────────────────────
def plot_roc_curves(labels, probs, save_path):
    # Binarize labels untuk one-vs-rest
    labels_bin = label_binarize(labels, classes=[0, 1, 2])
    colors     = ['steelblue', 'coral', 'seagreen']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('ROC Curves — Hybrid CNN-ViT (One-vs-Rest)',
                 fontsize=13, fontweight='bold')

    auc_scores = {}
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{cls} (AUC = {roc_auc:.4f})')

    ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"ROC curves tersimpan: {save_path}")
    return auc_scores


# ─── SENSITIVITY & SPECIFICITY ───────────────────────────────────────────────
def compute_sens_spec(cm):
    results = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        results[cls] = {'sensitivity': sensitivity, 'specificity': specificity,
                        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    return results


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  EVALUASI LENGKAP — Hybrid CNN-ViT v1")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data & model
    print("Memuat dataset...")
    _, _, test_loader = get_dataloaders(batch_size=8)

    print("\nMemuat model...")
    model = load_model(CHECKPOINT, device)

    # Prediksi
    print("\nMelakukan prediksi pada test set...")
    labels, preds, probs = get_predictions(model, test_loader, device)
    print(f"Total sampel test: {len(labels)}")

    # ── 1. Classification Report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  1. CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(labels, preds,
          target_names=CLASSES, digits=4))

    # ── 2. Confusion Matrix ───────────────────────────────────────────────────
    print("=" * 60)
    print("  2. CONFUSION MATRIX")
    print("=" * 60)
    cm = plot_confusion_matrix(
        labels, preds,
        os.path.join(FIGURES, 'confusion_matrix.png')
    )
    print("\nRaw confusion matrix:")
    print(f"{'':>20}", end='')
    for c in CLASSES:
        print(f"{c:>18}", end='')
    print()
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>20}", end='')
        for j in range(len(CLASSES)):
            print(f"{cm[i,j]:>18}", end='')
        print()

    # ── 3. Sensitivity & Specificity ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  3. SENSITIVITY & SPECIFICITY PER KELAS")
    print("=" * 60)
    sens_spec = compute_sens_spec(cm)
    print(f"\n{'Kelas':<22} {'Sensitivity':>12} {'Specificity':>12} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    print("-" * 70)
    for cls, vals in sens_spec.items():
        print(f"{cls:<22} {vals['sensitivity']:>12.4f} {vals['specificity']:>12.4f} "
              f"{vals['tp']:>6} {vals['fp']:>6} {vals['tn']:>6} {vals['fn']:>6}")

    # ── 4. AUC-ROC ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  4. AUC-ROC PER KELAS (One-vs-Rest)")
    print("=" * 60)
    auc_scores = plot_roc_curves(
        labels, probs,
        os.path.join(FIGURES, 'roc_curves.png')
    )
    for cls, score in auc_scores.items():
        print(f"  {cls:<22}: AUC = {score:.4f}")

    # ── 5. Ringkasan Final ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  5. RINGKASAN METRIK FINAL")
    print("=" * 60)
    from sklearn.metrics import accuracy_score, f1_score
    acc        = accuracy_score(labels, preds)
    f1_macro   = f1_score(labels, preds, average='macro',    zero_division=0)
    f1_weighted= f1_score(labels, preds, average='weighted', zero_division=0)
    mean_auc   = np.mean(list(auc_scores.values()))
    mean_sens  = np.mean([v['sensitivity'] for v in sens_spec.values()])
    mean_spec  = np.mean([v['specificity'] for v in sens_spec.values()])

    print(f"\n  {'Metrik':<25} {'Nilai':>10}")
    print(f"  {'-'*37}")
    print(f"  {'Accuracy':<25} {acc:>10.4f}")
    print(f"  {'F1 Macro':<25} {f1_macro:>10.4f}")
    print(f"  {'F1 Weighted':<25} {f1_weighted:>10.4f}")
    print(f"  {'Mean AUC-ROC':<25} {mean_auc:>10.4f}")
    print(f"  {'Mean Sensitivity':<25} {mean_sens:>10.4f}")
    print(f"  {'Mean Specificity':<25} {mean_spec:>10.4f}")

    print(f"\nSemua grafik tersimpan di: {FIGURES}")
    print("\nEvaluasi selesai! Lanjut ke gradcam.py")


if __name__ == '__main__':
    main()