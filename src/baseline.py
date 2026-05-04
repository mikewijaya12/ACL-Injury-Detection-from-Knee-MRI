"""
    baseline.py
    -----------
    Training CNN murni (EfficientNet-B0) tanpa Vision Transformer.
    Digunakan sebagai baseline untuk ablation study —
    membuktikan bahwa hybrid CNN-ViT lebih baik dari CNN saja.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders, load_metadata, split_dataset, LABEL_NAMES

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CONFIG = {
    'num_epochs'    : 50,
    'batch_size'    : 8,
    'learning_rate' : 1e-4,
    'weight_decay'  : 1e-2,
    'patience'      : 10,
    'num_classes'   : 3,
    'num_slices'    : 9,
    'dropout'       : 0.2,
    'checkpoint_dir': r'D:\ACL\outputs\checkpoints',
    'figures_dir'   : r'D:\ACL\outputs\figures',
}


# ─── CNN ONLY MODEL ──────────────────────────────────────────────────────────
class CNNOnly(nn.Module):
    """
    Baseline: EfficientNet-B0 saja tanpa Transformer.
    Tiap slice diproses CNN → rata-rata fitur → klasifikasi.
    Ini yang ingin kita kalahkan dengan Hybrid CNN-ViT.
    """
    def __init__(self, num_classes=3, num_slices=9, dropout=0.2):
        super(CNNOnly, self).__init__()
        self.num_slices = num_slices

        # CNN backbone — sama persis dengan hybrid model
        self.cnn = timm.create_model(
            'efficientnet_b0',
            pretrained  = True,
            in_chans    = 1,
            num_classes = 0,
            global_pool = 'avg'
        )
        cnn_out_dim = self.cnn.num_features  # 1280

        # Classification head langsung dari CNN features
        # Tanpa Transformer — ini bedanya dengan hybrid
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x: (B, S, H, W)
        """
        B, S, H, W = x.shape

        # CNN per slice
        x        = x.view(B * S, 1, H, W)   # (B*S, 1, H, W)
        features = self.cnn(x)               # (B*S, 1280)
        features = features.view(B, S, -1)   # (B, S, 1280)

        # Average pooling antar slice (tanpa Transformer)
        features = features.mean(dim=1)      # (B, 1280)

        # Klasifikasi
        logits = self.classifier(features)   # (B, 3)
        return logits


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def get_class_weights(train_meta, device):
    labels  = train_meta['aclDiagnosis']
    counts  = Counter(labels)
    total   = len(labels)
    weights = torch.tensor(
        [total / (3 * counts[i]) for i in range(3)],
        dtype=torch.float32
    ).to(device)
    return weights


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

    return (total_loss / total,
            correct / total,
            f1_score(all_labels, all_preds, average='macro', zero_division=0),
            all_preds, all_labels)


def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training Curves — CNN Only (Baseline)',
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Kurva tersimpan: {save_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BASELINE TRAINING — CNN Only (EfficientNet-B0)")
    print("  Ablation study: tanpa Vision Transformer")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # Data
    print("\nMemuat dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=CONFIG['batch_size']
    )
    metadata         = load_metadata()
    train_meta, _, _ = split_dataset(metadata)
    class_weights    = get_class_weights(train_meta, device)

    print("\nClass weights:")
    for i, (name, w) in enumerate(zip(LABEL_NAMES.values(), class_weights)):
        print(f"  {name:<20}: {w.item():.4f}")

    # Model
    print("\nMembangun CNN Only model...")
    model = CNNOnly(
        num_classes = CONFIG['num_classes'],
        num_slices  = CONFIG['num_slices'],
        dropout     = CONFIG['dropout']
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter: {trainable:,} trainable / {total:,} total")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(),
                      lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=CONFIG['num_epochs'],
                                  eta_min=1e-6)

    history = {k: [] for k in
               ['train_loss','val_loss','train_acc','val_acc','train_f1','val_f1']}

    best_val_f1    = 0.0
    patience_count = 0
    best_epoch     = 0

    print(f"\nMulai training ({CONFIG['num_epochs']} epoch)...")
    print("-" * 60)

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(
            ['train_loss','val_loss','train_acc','val_acc','train_f1','val_f1'],
            [tr_loss, vl_loss, tr_acc, vl_acc, tr_f1, vl_f1]
        ):
            history[k].append(v)

        print(f"Epoch {epoch:>3}/{CONFIG['num_epochs']} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
              f"F1 {tr_f1:.4f}/{vl_f1:.4f} | "
              f"{time.time()-t0:.1f}s")

        if vl_f1 > best_val_f1:
            best_val_f1    = vl_f1
            best_epoch     = epoch
            patience_count = 0
            ckpt_path = os.path.join(
                CONFIG['checkpoint_dir'], 'best_model_baseline.pth')
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_f1'     : vl_f1,
                'val_acc'    : vl_acc,
            }, ckpt_path)
            print(f"  ★ Model terbaik tersimpan (val F1: {best_val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= CONFIG['patience']:
                print(f"\nEarly stopping di epoch {epoch} "
                      f"(best: epoch {best_epoch})")
                break

    # Plot
    plot_curves(history, os.path.join(
        CONFIG['figures_dir'], 'training_curves_baseline.png'))

    # Evaluasi final
    print("\n" + "=" * 60)
    print("  EVALUASI FINAL — CNN Only Baseline")
    print("=" * 60)

    ckpt = torch.load(os.path.join(
        CONFIG['checkpoint_dir'], 'best_model_baseline.pth'))
    model.load_state_dict(ckpt['model_state'])
    print(f"Load model terbaik dari epoch {ckpt['epoch']}")

    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=list(LABEL_NAMES.values()),
        digits=4
    ))

    # Perbandingan langsung
    print("=" * 60)
    print("  ABLATION STUDY — Perbandingan")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1 Macro':>10}")
    print(f"  {'-'*47}")
    print(f"  {'CNN Only (baseline)':<25} {test_acc:>10.4f} {test_f1:>10.4f}")
    print(f"  {'Hybrid CNN-ViT (v1)':<25} {'0.8014':>10} {'0.5975':>10}")
    delta_acc = test_acc - 0.8014
    delta_f1  = test_f1  - 0.5975
    print(f"\n  Delta (Hybrid - CNN):")
    print(f"  {'Accuracy':<25} {delta_acc:>+10.4f}")
    print(f"  {'F1 Macro':<25} {delta_f1:>+10.4f}")

    if delta_f1 > 0:
        print(f"\n  ✅ Hybrid CNN-ViT LEBIH BAIK dari CNN Only")
        print(f"     (F1 naik {delta_f1:.4f} = {delta_f1*100:.2f}%)")
    else:
        print(f"\n  ⚠️  CNN Only lebih baik — perlu investigasi lebih lanjut")


if __name__ == '__main__':
    main()