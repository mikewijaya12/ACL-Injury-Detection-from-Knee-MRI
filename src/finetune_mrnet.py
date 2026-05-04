"""
    finetune_mrnet.py
    -----------------
    Fine-tune Hybrid CNN-ViT (trained di KneeMRI) pada MRNet.
    - Ganti classification head: 3 kelas → 2 kelas
    - Freeze CNN + Transformer, hanya train head dulu (5 epoch)
    - Unfreeze semua, fine-tune dengan lr kecil
    - Evaluasi di MRNet valid set
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (f1_score, classification_report,
                             roc_curve, auc, accuracy_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_mrnet import get_mrnet_dataloaders, LABEL_NAMES_MRNET
from model import HybridCNNViT

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CONFIG = {
    'kneemri_checkpoint': r'D:\ACL\outputs\checkpoints\best_model.pth',
    'checkpoint_dir'    : r'D:\ACL\outputs\checkpoints',
    'figures_dir'       : r'D:\ACL\outputs\figures',
    'plane'             : 'sagittal',
    'batch_size'        : 8,
    'head_epochs'       : 5,       # epoch untuk train head saja
    'finetune_epochs'   : 30,      # epoch fine-tune semua layer
    'lr_head'           : 1e-3,    # lr untuk head
    'lr_finetune'       : 1e-5,    # lr untuk fine-tune semua layer
    'weight_decay'      : 1e-2,
    'patience'          : 8,
    'num_classes_mrnet' : 2,       # binary: intact / tear
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['figures_dir'],    exist_ok=True)
CLASSES_MRNET = list(LABEL_NAMES_MRNET.values())


# ─── MODIFIKASI MODEL: 3 KELAS → 2 KELAS ────────────────────────────────────
def adapt_model_for_mrnet(checkpoint_path, device):
    """
    Load model KneeMRI, ganti head terakhir 3→2 kelas.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt['config']

    # Buat model dengan struktur asli (3 kelas)
    model = HybridCNNViT(
        num_classes = cfg['num_classes'],  # 3
        num_slices  = cfg['num_slices'],
        embed_dim   = cfg['embed_dim'],
        num_heads   = cfg['num_heads'],
        num_layers  = cfg['num_layers'],
        dropout     = cfg['dropout'],
        pretrained  = False
    )

    # Load bobot KneeMRI
    model.load_state_dict(ckpt['model_state'])
    print(f"  Bobot KneeMRI loaded dari epoch {ckpt['epoch']} "
          f"(val F1: {ckpt['val_f1']:.4f})")

    # Ganti classification head: 3 → 2 kelas
    embed_dim = cfg['embed_dim']
    model.classifier = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(cfg['dropout']),
        nn.Linear(embed_dim, embed_dim // 2),
        nn.GELU(),
        nn.Dropout(cfg['dropout']),
        nn.Linear(embed_dim // 2, CONFIG['num_classes_mrnet'])  # 2 kelas
    )
    print(f"  Classification head diganti: 3 kelas → 2 kelas")

    return model.to(device)


# ─── FREEZE / UNFREEZE ───────────────────────────────────────────────────────
def freeze_backbone(model):
    """Freeze CNN + Transformer, hanya head yang bisa belajar."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone frozen — trainable params: {trainable:,} (head only)")

def unfreeze_all(model):
    """Unfreeze semua layer untuk fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Semua layer unfreeze — trainable params: {trainable:,}")


# ─── TRAIN & EVAL ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / total,
            correct / total,
            f1_score(all_labels, all_preds, average='macro', zero_division=0))


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            probs  = torch.softmax(logits, dim=1)

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (total_loss / total,
            correct / total,
            f1_score(all_labels, all_preds, average='macro', zero_division=0),
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


# ─── PLOT ────────────────────────────────────────────────────────────────────
def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Fine-tune Curves — Hybrid CNN-ViT on MRNet',
                 fontsize=13, fontweight='bold')

    for ax, (tr, vl), title in zip(
        axes,
        [('train_loss','val_loss'),
         ('train_acc','val_acc'),
         ('train_f1','val_f1')],
        ['Loss', 'Accuracy', 'Macro F1']
    ):
        ax.plot(epochs, history[tr], label='Train', color='steelblue')
        ax.plot(epochs, history[vl], label='Val',   color='coral')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)
        if title != 'Loss':
            ax.set_ylim(0, 1)
        # Garis pemisah head-only vs fine-tune
        if CONFIG['head_epochs'] < len(history['train_loss']):
            ax.axvline(x=CONFIG['head_epochs'], color='green',
                      linestyle='--', alpha=0.5, label='Full fine-tune')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Kurva tersimpan: {save_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FINE-TUNE — Hybrid CNN-ViT pada MRNet")
    print("  KneeMRI → MRNet (3 kelas → 2 kelas)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # Data
    print("\nMemuat MRNet dataset...")
    train_loader, val_loader = get_mrnet_dataloaders(
        plane      = CONFIG['plane'],
        batch_size = CONFIG['batch_size']
    )

    # Model
    print("\nAdaptasi model KneeMRI untuk MRNet...")
    model = adapt_model_for_mrnet(CONFIG['kneemri_checkpoint'], device)

    # Class weights untuk imbalance MRNet
    train_labels  = train_loader.dataset.labels
    counts        = Counter(train_labels)
    total         = len(train_labels)
    class_weights = torch.tensor(
        [total / (2 * counts[i]) for i in range(2)],
        dtype=torch.float32
    ).to(device)
    print(f"\nClass weights MRNet:")
    for i, (name, w) in enumerate(zip(CLASSES_MRNET, class_weights)):
        print(f"  {name:<12}: {w.item():.4f}  (n={counts[i]})")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {k: [] for k in
               ['train_loss','val_loss','train_acc','val_acc',
                'train_f1','val_f1']}
    best_val_f1    = 0.0
    patience_count = 0
    best_epoch     = 0
    total_epoch    = 0

    # ── FASE 1: Train head saja ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FASE 1: Train head saja ({CONFIG['head_epochs']} epoch)")
    print(f"{'='*60}")
    freeze_backbone(model)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr_head'], weight_decay=CONFIG['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=CONFIG['head_epochs'], eta_min=1e-6)

    for epoch in range(1, CONFIG['head_epochs'] + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()
        total_epoch += 1

        for k, v in zip(
            ['train_loss','val_loss','train_acc','val_acc','train_f1','val_f1'],
            [tr_loss, vl_loss, tr_acc, vl_acc, tr_f1, vl_f1]
        ):
            history[k].append(v)

        print(f"[HEAD] Epoch {epoch:>2}/{CONFIG['head_epochs']} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
              f"F1 {tr_f1:.4f}/{vl_f1:.4f} | "
              f"{time.time()-t0:.1f}s")

        if vl_f1 > best_val_f1:
            best_val_f1    = vl_f1
            best_epoch     = total_epoch
            patience_count = 0

    # ── FASE 2: Fine-tune semua layer ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FASE 2: Fine-tune semua layer ({CONFIG['finetune_epochs']} epoch)")
    print(f"{'='*60}")
    unfreeze_all(model)
    patience_count = 0

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr_finetune'], weight_decay=CONFIG['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=CONFIG['finetune_epochs'], eta_min=1e-7)

    for epoch in range(1, CONFIG['finetune_epochs'] + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()
        total_epoch += 1

        for k, v in zip(
            ['train_loss','val_loss','train_acc','val_acc','train_f1','val_f1'],
            [tr_loss, vl_loss, tr_acc, vl_acc, tr_f1, vl_f1]
        ):
            history[k].append(v)

        print(f"[FULL] Epoch {epoch:>2}/{CONFIG['finetune_epochs']} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
              f"F1 {tr_f1:.4f}/{vl_f1:.4f} | "
              f"{time.time()-t0:.1f}s")

        if vl_f1 > best_val_f1:
            best_val_f1    = vl_f1
            best_epoch     = total_epoch
            patience_count = 0
            ckpt_path = os.path.join(
                CONFIG['checkpoint_dir'], 'best_model_mrnet.pth')
            torch.save({
                'epoch'      : total_epoch,
                'model_state': model.state_dict(),
                'val_f1'     : vl_f1,
                'val_acc'    : vl_acc,
            }, ckpt_path)
            print(f"  ★ Model terbaik tersimpan (val F1: {best_val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= CONFIG['patience']:
                print(f"\nEarly stopping (best: epoch {best_epoch})")
                break

    # Plot
    plot_curves(history, os.path.join(
        CONFIG['figures_dir'], 'training_curves_mrnet.png'))

    # ── EVALUASI FINAL ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUASI FINAL — MRNet Valid Set")
    print("=" * 60)

    ckpt = torch.load(os.path.join(
        CONFIG['checkpoint_dir'], 'best_model_mrnet.pth'))
    model.load_state_dict(ckpt['model_state'])
    print(f"Load model terbaik dari epoch {ckpt['epoch']}")

    _, test_acc, test_f1, preds, labels, probs = evaluate(
        model, val_loader, criterion, device)

    # AUC
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc     = auc(fpr, tpr)

    print(f"\nHasil di MRNet Valid Set:")
    print(f"  Accuracy    : {test_acc:.4f}")
    print(f"  F1 Macro    : {test_f1:.4f}")
    print(f"  AUC-ROC     : {roc_auc:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(labels, preds,
          target_names=CLASSES_MRNET, digits=4))

    # Plot ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'ACL Tear (AUC = {roc_auc:.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — MRNet Valid Set')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(CONFIG['figures_dir'], 'roc_mrnet.png')
    plt.savefig(roc_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nROC curve tersimpan: {roc_path}")

    # Tabel perbandingan final
    print("\n" + "=" * 60)
    print("  RINGKASAN AKHIR — Semua Eksperimen")
    print("=" * 60)
    print(f"\n  {'Model':<30} {'Dataset':<12} {'Acc':>7} {'F1':>7} {'AUC':>7}")
    print(f"  {'-'*65}")
    print(f"  {'CNN Only (baseline)':<30} {'KneeMRI':<12} {'?':>7} {'?':>7} {'?':>7}")
    print(f"  {'Hybrid CNN-ViT':<30} {'KneeMRI':<12} {'0.8014':>7} {'0.5975':>7} {'0.7488':>7}")
    print(f"  {'Hybrid CNN-ViT (fine-tuned)':<30} {'MRNet':<12} {test_acc:>7.4f} {test_f1:>7.4f} {roc_auc:>7.4f}")
    print(f"\n  * CNN Only hasil menyusul dari baseline.py")


if __name__ == '__main__':
    main()
