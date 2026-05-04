"""
    train_attention_roi.py
    ----------------------
    Training script untuk AttentionROI model.
    Input: full slice (tanpa ROI manual)
    Output: klasifikasi + attention map otomatis
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_v2 import (get_dataloaders_v2, load_metadata,
                         split_dataset, LABEL_NAMES)
from model_attention_roi import AttentionROIModel, count_parameters

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CONFIG = {
    'num_epochs'    : 50,
    'batch_size'    : 8,
    'learning_rate' : 1e-4,
    'weight_decay'  : 1e-2,
    'patience'      : 10,
    'num_classes'   : 3,
    'num_slices'    : 9,
    'cnn_name'      : 'efficientnet_b0',
    'embed_dim'     : 256,
    'num_heads'     : 8,
    'num_layers'    : 4,
    'dropout'       : 0.2,
    'label_smoothing': 0.1,
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
    return device


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


# ─── TRAIN ONE EPOCH ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        # Model returns (logits, attn_map)
        logits, _ = model(imgs)
        loss      = criterion(logits, labels)
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
            f1_score(all_labels, all_preds,
                     average='macro', zero_division=0))


# ─── EVALUASI ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(imgs)
            loss      = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (total_loss / total,
            correct / total,
            f1_score(all_labels, all_preds,
                     average='macro', zero_division=0),
            all_preds, all_labels)


# ─── PLOT ────────────────────────────────────────────────────────────────────
def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        'Training Curves — AttentionROI (No Manual ROI)',
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


# ─── VISUALISASI ATTENTION MAP ───────────────────────────────────────────────
def visualize_attention(model, loader, device, save_dir, num_samples=3):
    """
    Simpan visualisasi attention map vs full slice.
    Ini bukti visual bahwa model menemukan area ACL sendiri.
    """
    import cv2
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs_dev = imgs.to(device)
            logits, attn_maps = model(imgs_dev)
            preds = logits.argmax(dim=1)

            for b in range(imgs.shape[0]):
                if count >= num_samples:
                    return

                label = labels[b].item()
                pred  = preds[b].item()
                img_slices  = imgs[b].cpu().numpy()    # (9, 224, 224)
                attn_slices = attn_maps[b].cpu().numpy() # (9, 7, 7)

                # Plot 9 slice: original + attention overlay
                fig, axes = plt.subplots(2, 9, figsize=(18, 5))
                correct     = label == pred
                title_color = 'green' if correct else 'red'
                status      = '✓ BENAR' if correct else '✗ SALAH'

                fig.suptitle(
                    f"Attention-guided ROI — "
                    f"Aktual: {LABEL_NAMES[label]} | "
                    f"Prediksi: {LABEL_NAMES[pred]} {status}",
                    fontsize=10, fontweight='bold',
                    color=title_color
                )

                for s in range(9):
                    img  = img_slices[s]   # (224, 224)
                    attn = attn_slices[s]  # (7, 7)

                    # Resize attention ke 224x224
                    attn_r = cv2.resize(
                        attn.astype(np.float32), (224, 224))

                    # Overlay
                    img_rgb = np.stack([img, img, img], axis=-1)
                    heatmap = plt.cm.jet(attn_r)[:, :, :3]
                    overlay = np.clip(
                        0.55 * img_rgb + 0.45 * heatmap, 0, 1)

                    axes[0, s].imshow(img, cmap='gray')
                    axes[0, s].axis('off')
                    axes[0, s].set_title(f'S{s+1}', fontsize=8)

                    axes[1, s].imshow(overlay)
                    axes[1, s].axis('off')

                axes[0, 0].set_ylabel('Original', fontsize=8)
                axes[1, 0].set_ylabel('Attention\n(ROI otomatis)',
                                      fontsize=8)

                plt.tight_layout()
                fname = (f'attn_roi_{LABEL_NAMES[label].replace(" ","_")}'
                         f'_{"correct" if correct else "wrong"}'
                         f'_{count}.png')
                plt.savefig(os.path.join(save_dir, fname),
                           dpi=100, bbox_inches='tight')
                plt.close()
                count += 1
                print(f"  Attention map {count}: {fname}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TRAINING — AttentionROI Model")
    print("  Novelty: Tidak butuh ROI manual!")
    print("=" * 60)

    device = setup()

    # Data
    print("\nMemuat dataset v2 (full slice, no ROI crop)...")
    train_loader, val_loader, test_loader, train_meta = \
        get_dataloaders_v2(batch_size=CONFIG['batch_size'])

    class_weights = get_class_weights(train_meta, device)

    # Model
    print("\nMembangun AttentionROI model...")
    model = AttentionROIModel(
        num_classes = CONFIG['num_classes'],
        num_slices  = CONFIG['num_slices'],
        cnn_name    = CONFIG['cnn_name'],
        embed_dim   = CONFIG['embed_dim'],
        num_heads   = CONFIG['num_heads'],
        num_layers  = CONFIG['num_layers'],
        dropout     = CONFIG['dropout'],
        pretrained  = True
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"Parameter: {trainable:,} trainable / {total:,} total")

    criterion = nn.CrossEntropyLoss(
        weight          = class_weights,
        label_smoothing = CONFIG['label_smoothing']
    )
    optimizer = AdamW(model.parameters(),
                      lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)

    history = {k: [] for k in
               ['train_loss','val_loss','train_acc',
                'val_acc','train_f1','val_f1']}

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
            ['train_loss','val_loss','train_acc',
             'val_acc','train_f1','val_f1'],
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
                CONFIG['checkpoint_dir'],
                'best_model_attention_roi.pth')
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_f1'     : vl_f1,
                'val_acc'    : vl_acc,
                'config'     : CONFIG
            }, ckpt_path)
            print(f"  ★ Model terbaik (val F1: {best_val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= CONFIG['patience']:
                print(f"\nEarly stopping (best: epoch {best_epoch})")
                break

    # Plot
    plot_curves(history, os.path.join(
        CONFIG['figures_dir'],
        'training_curves_attention_roi.png'))

    # Evaluasi final
    print("\n" + "=" * 60)
    print("  EVALUASI FINAL — Test Set")
    print("=" * 60)

    ckpt = torch.load(os.path.join(
        CONFIG['checkpoint_dir'],
        'best_model_attention_roi.pth'))
    model.load_state_dict(ckpt['model_state'])
    print(f"Load model terbaik dari epoch {ckpt['epoch']}")

    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print(classification_report(
        test_labels, test_preds,
        target_names=list(LABEL_NAMES.values()), digits=4))

    # Visualisasi attention map
    print("\nGenerating attention map visualizations...")
    attn_dir = os.path.join(CONFIG['figures_dir'], 'attention_roi')
    visualize_attention(model, test_loader, device,
                       attn_dir, num_samples=6)

    # Perbandingan final
    print("\n" + "=" * 60)
    print("  PERBANDINGAN SEMUA MODEL")
    print("=" * 60)
    print(f"\n  {'Model':<30} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*48}")
    print(f"  {'CNN Only':<30} {'0.8014':>8} {'0.5903':>8}")
    print(f"  {'Hybrid CNN-ViT (ROI manual)':<30} {'0.8014':>8} {'0.5975':>8}")
    print(f"  {'AttentionROI (no ROI manual)':<30} {test_acc:>8.4f} {test_f1:>8.4f}")
    print(f"\n  ★ Novelty: AttentionROI tidak butuh anotasi ROI manual")


if __name__ == '__main__':
    main()
