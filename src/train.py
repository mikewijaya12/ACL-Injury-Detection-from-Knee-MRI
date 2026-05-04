"""
    train.py
    --------
    Training loop untuk Hybrid CNN-ViT ACL detection.
    - Class-weighted loss untuk imbalanced data
    - AdamW optimizer + CosineAnnealingLR scheduler
    - Early stopping
    - Simpan model terbaik otomatis
    - Plot kurva training
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from collections import Counter

# Tambahkan src ke path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders, load_metadata, split_dataset, LABEL_NAMES
from model import HybridCNNViT, count_parameters

# ─── KONFIGURASI ─────────────────────────────────────────────────────────────
CONFIG = {
    'num_epochs'    : 50,
    'batch_size'    : 8,
    'learning_rate' : 1e-4,
    'weight_decay'  : 1e-2,
    'patience'      : 10,        # early stopping
    'num_classes'   : 3,
    'num_slices'    : 9,
    'embed_dim'     : 256,
    'num_heads'     : 8,
    'num_layers'    : 4,
    'dropout'       : 0.2,
    'checkpoint_dir': r'D:\ACL\outputs\checkpoints',
    'figures_dir'   : r'D:\ACL\outputs\figures',
}

# ─── SETUP ───────────────────────────────────────────────────────────────────
def setup():
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['figures_dir'],    exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


# ─── HITUNG CLASS WEIGHTS ─────────────────────────────────────────────────────
def get_class_weights(train_meta, device):
    labels  = train_meta['aclDiagnosis']
    counts  = Counter(labels)
    total   = len(labels)
    weights = torch.tensor(
        [total / (3 * counts[i]) for i in range(3)],
        dtype=torch.float32
    ).to(device)
    print(f"\nClass weights:")
    for i, (name, w) in enumerate(zip(LABEL_NAMES.values(), weights)):
        print(f"  {name:<20}: {w.item():.4f}  (n={counts[i]})")
    return weights


# ─── SATU EPOCH TRAINING ─────────────────────────────────────────────────────
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

        # Gradient clipping — mencegah exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, f1


# ─── EVALUASI ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, f1, all_preds, all_labels


# ─── PLOT TRAINING CURVES ────────────────────────────────────────────────────
def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training Curves — Hybrid CNN-ViT', fontsize=13, fontweight='bold')

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='coral')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train', color='steelblue')
    axes[1].plot(epochs, history['val_acc'],   label='Val',   color='coral')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # F1 Score
    axes[2].plot(epochs, history['train_f1'], label='Train', color='steelblue')
    axes[2].plot(epochs, history['val_f1'],   label='Val',   color='coral')
    axes[2].set_title('Macro F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Kurva tersimpan: {save_path}")


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TRAINING — Hybrid CNN-ViT untuk ACL Detection")
    print("=" * 60)

    device = setup()

    # DataLoaders
    print("\nMemuat dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=CONFIG['batch_size']
    )

    # Class weights
    metadata   = load_metadata()
    train_meta, _, _ = split_dataset(metadata)
    class_weights = get_class_weights(train_meta, device)

    # Model
    print("\nMembangun model...")
    model = HybridCNNViT(
        num_classes = CONFIG['num_classes'],
        num_slices  = CONFIG['num_slices'],
        embed_dim   = CONFIG['embed_dim'],
        num_heads   = CONFIG['num_heads'],
        num_layers  = CONFIG['num_layers'],
        dropout     = CONFIG['dropout'],
        pretrained  = True
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"Parameter: {trainable:,} trainable / {total:,} total")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr           = CONFIG['learning_rate'],
        weight_decay = CONFIG['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max = CONFIG['num_epochs'],
        eta_min = 1e-6
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : [],
        'train_f1'  : [], 'val_f1'  : []
    }

    best_val_f1    = 0.0
    patience_count = 0
    best_epoch     = 0

    print(f"\nMulai training ({CONFIG['num_epochs']} epoch, early stopping patience={CONFIG['patience']})...")
    print("-" * 60)

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        t_start = time.time()

        # Train
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        elapsed = time.time() - t_start

        # Simpan history
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(vl_f1)

        # Log
        print(
            f"Epoch {epoch:>3}/{CONFIG['num_epochs']} | "
            f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
            f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
            f"F1 {tr_f1:.4f}/{vl_f1:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Simpan model terbaik
        if vl_f1 > best_val_f1:
            best_val_f1    = vl_f1
            best_epoch     = epoch
            patience_count = 0
            ckpt_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model_v3.pth')
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_f1'     : vl_f1,
                'val_acc'    : vl_acc,
                'config'     : CONFIG
            }, ckpt_path)
            print(f"  ★ Model terbaik tersimpan (val F1: {best_val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= CONFIG['patience']:
                print(f"\nEarly stopping di epoch {epoch} (best epoch: {best_epoch})")
                break

    # ── Plot kurva ────────────────────────────────────────────────────────────
    curve_path = os.path.join(CONFIG['figures_dir'], 'training_curves_v3.png')
    plot_curves(history, curve_path)

    # ── Evaluasi final di test set ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUASI FINAL — Test Set")
    print("=" * 60)

    # Load model terbaik
    ckpt = torch.load(os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(ckpt['model_state'])
    print(f"Load model terbaik dari epoch {ckpt['epoch']}")

    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=list(LABEL_NAMES.values()),
        digits=4
    ))


if __name__ == '__main__':
    main()