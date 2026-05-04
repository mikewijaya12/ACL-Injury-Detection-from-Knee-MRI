"""
    gradcam.py (fixed)
    ------------------
    Visualisasi Grad-CAM untuk Hybrid CNN-ViT.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import (load_metadata, split_dataset,
                     KneeMRIDataset, LABEL_NAMES)
from model import HybridCNNViT

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CHECKPOINT = r'D:\ACL\outputs\checkpoints\best_model.pth'
FIGURES    = r'D:\ACL\outputs\figures\gradcam'
os.makedirs(FIGURES, exist_ok=True)
CLASSES    = list(LABEL_NAMES.values())


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
def load_model(device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt  = torch.load(CHECKPOINT, map_location=device)
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
    print(f"Model loaded dari epoch {ckpt['epoch']}")
    return model, cfg['num_slices']


# ─── GRAD-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook pada conv_head EfficientNet (layer terakhir CNN)
        target = model.cnn.conv_head

        def fwd_hook(module, input, output):
            self.activations = output.detach().clone()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach().clone()

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def generate(self, imgs, class_idx, num_slices):
        """
        imgs: (1, S, H, W) tensor
        return: list of S numpy arrays (H, W) normalized 0-1
        """
        self.model.eval()
        self.gradients   = None
        self.activations = None

        imgs = imgs.clone().requires_grad_(True)

        # Forward
        logits = self.model(imgs)
        probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        # Backward untuk kelas target
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # gradients & activations shape: (B*S, C, h, w)
        grads = self.gradients    # (num_slices, C, h, w)
        acts  = self.activations  # (num_slices, C, h, w)

        # GAP weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (S, C, 1, 1)
        cam     = (weights * acts).sum(dim=1)            # (S, h, w)
        cam     = F.relu(cam).cpu().numpy()              # (S, h, w)

        # Normalize per slice ke [0, 1]
        cam_normalized = []
        for s in range(cam.shape[0]):
            c = cam[s]
            mn, mx = c.min(), c.max()
            if mx > mn:
                c = (c - mn) / (mx - mn)
            else:
                c = np.zeros_like(c)
            cam_normalized.append(c.astype(np.float32))

        return cam_normalized, probs


# ─── VISUALISASI ─────────────────────────────────────────────────────────────
def overlay_cam(img_np, cam_np, img_size):
    """
    img_np: (H, W) float [0,1]
    cam_np: (h, w) float [0,1]
    return: (H, W, 3) float [0,1] overlay
    """
    cam_r = cv2.resize(cam_np, (img_size, img_size))
    img_rgb = np.stack([img_np, img_np, img_np], axis=-1)
    heatmap = plt.cm.jet(cam_r)[:, :, :3]
    overlay = np.clip(0.55 * img_rgb + 0.45 * heatmap, 0, 1)
    return overlay


def visualize_sample(imgs_tensor, cam_list, label, pred, probs,
                     sample_idx, save_path):
    num_slices = imgs_tensor.shape[0]
    img_size   = imgs_tensor.shape[1]

    correct     = (label == pred)
    title_color = 'green' if correct else 'red'
    status      = '✓ BENAR' if correct else '✗ SALAH'

    fig, axes = plt.subplots(2, num_slices, figsize=(18, 5))
    fig.suptitle(
        f"Grad-CAM #{sample_idx} | Aktual: {CLASSES[label]} | "
        f"Prediksi: {CLASSES[pred]} {status}\n"
        f"P(Healthy)={probs[0]:.3f}  P(Partial)={probs[1]:.3f}  "
        f"P(Complete)={probs[2]:.3f}",
        fontsize=10, fontweight='bold', color=title_color
    )

    for i in range(num_slices):
        img = imgs_tensor[i].cpu().numpy()  # (H, W)
        cam = cam_list[i]                   # (h, w)
        ovl = overlay_cam(img, cam, img_size)

        # Baris atas: gambar asli
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'S{i+1}', fontsize=8)
        axes[0, i].axis('off')

        # Baris bawah: overlay Grad-CAM
        axes[1, i].imshow(ovl)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=8)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GRAD-CAM VISUALIZATION — Hybrid CNN-ViT")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model, num_slices = load_model(device)
    gradcam = GradCAM(model)

    # Load test dataset
    metadata          = load_metadata()
    _, _, test_meta   = split_dataset(metadata)
    print("Caching test dataset...")
    test_dataset = KneeMRIDataset(test_meta, augment=False, cache=True)

    print(f"\nMencari sampel per kelas...")

    # Kumpulkan 1 benar + 1 salah per kelas
    correct_per_class   = {0: None, 1: None, 2: None}
    incorrect_per_class = {0: None, 1: None, 2: None}

    model.eval()
    for idx in range(len(test_dataset)):
        imgs, label = test_dataset[idx]
        imgs_t = imgs.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(imgs_t)
            pred   = logits.argmax(dim=1).item()
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        if pred == label and correct_per_class[label] is None:
            correct_per_class[label] = (idx, imgs, label, pred, probs)
        elif pred != label and incorrect_per_class[label] is None:
            incorrect_per_class[label] = (idx, imgs, label, pred, probs)

        done = (all(v is not None for v in correct_per_class.values()) and
                all(v is not None for v in incorrect_per_class.values()))
        if done:
            break

    # Generate & simpan visualisasi individual
    print("\nGenerating Grad-CAM visualizations...")
    for cls_idx, cls_name in LABEL_NAMES.items():
        for status, pool in [('correct', correct_per_class),
                              ('wrong',   incorrect_per_class)]:
            entry = pool[cls_idx]
            if entry is None:
                print(f"  [{cls_name}] {status} — tidak ditemukan di test set")
                continue

            idx, imgs, label, pred, probs = entry
            imgs_t   = imgs.unsqueeze(0).to(device)
            cam_list, _ = gradcam.generate(imgs_t, pred, num_slices)

            fname = f'gradcam_{cls_name.replace(" ","_")}_{status}.png'
            save_path = os.path.join(FIGURES, fname)
            visualize_sample(imgs, cam_list, label, pred, probs, idx, save_path)
            print(f"  [{cls_name}] {status} → {fname}")

    # Summary grid: 1 correct per kelas
    print("\nMembuat summary grid...")
    fig, axes = plt.subplots(3, num_slices, figsize=(18, 7))
    fig.suptitle(
        'Grad-CAM Summary — Satu Contoh Per Kelas (Prediksi Benar)',
        fontsize=12, fontweight='bold'
    )

    for cls_idx, cls_name in LABEL_NAMES.items():
        entry = correct_per_class[cls_idx]
        if entry is None:
            continue
        idx, imgs, label, pred, probs = entry
        imgs_t   = imgs.unsqueeze(0).to(device)
        cam_list, _ = gradcam.generate(imgs_t, pred, num_slices)

        for s in range(num_slices):
            ax  = axes[cls_idx, s]
            img = imgs[s].cpu().numpy()
            ovl = overlay_cam(img, cam_list[s], img.shape[0])
            ax.imshow(ovl)
            ax.axis('off')
            if s == 0:
                ax.set_ylabel(f'{cls_name}\n(idx={idx})',
                             fontsize=8, fontweight='bold')
            if cls_idx == 0:
                ax.set_title(f'Slice {s+1}', fontsize=8)

    plt.tight_layout()
    summary_path = os.path.join(FIGURES, 'gradcam_summary.png')
    plt.savefig(summary_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Summary tersimpan: {summary_path}")
    print(f"\nSemua Grad-CAM tersimpan di: {FIGURES}")
    print("Grad-CAM selesai! Lanjut ke baseline.py")


if __name__ == '__main__':
    main()
