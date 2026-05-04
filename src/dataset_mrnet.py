"""
    dataset_mrnet.py
    ----------------
    Dataset class untuk MRNet (Stanford).
    Digunakan untuk external validation model yang ditraining di KneeMRI.

    Struktur MRNet:
        MRNet-v1.0/
        ├── train/
        │   ├── axial/      ← file .npy (D, H, W)
        │   ├── coronal/
        │   └── sagittal/
        ├── valid/
        │   ├── axial/
        │   ├── coronal/
        │   └── sagittal/
        ├── train-acl.csv   ← format: 0000,0 (id, label)
        └── valid-acl.csv

    Label: 0 = intact ACL, 1 = ACL tear
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from collections import Counter

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MRNET_DIR  = r'D:\ACL\MRNet-v1.0'
NUM_SLICES = 9
IMG_SIZE   = 224
PLANES     = ['sagittal', 'coronal', 'axial']
LABEL_NAMES_MRNET = {0: 'Intact', 1: 'ACL Tear'}


# ─── LOAD LABELS ─────────────────────────────────────────────────────────────
def load_mrnet_labels(split='train'):
    """
    Load label dari CSV.
    Return: dict {case_id: label}
    """
    csv_path = os.path.join(MRNET_DIR, f'{split}-acl.csv')
    labels   = {}
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case_id, label = line.split(',')
            labels[case_id] = int(label)
    return labels


# ─── PREPROCESS SATU VOLUME ──────────────────────────────────────────────────
def preprocess_mrnet_volume(volume, num_slices=NUM_SLICES, img_size=IMG_SIZE):
    """
    volume: numpy array (D, H, W)
    return: numpy array (num_slices, img_size, img_size) float32
    """
    D = volume.shape[0]

    # Ambil num_slices dari tengah secara merata
    indices  = np.linspace(0, D - 1, num_slices, dtype=int)
    selected = volume[indices]  # (num_slices, H, W)

    processed = []
    for s in selected:
        resized = cv2.resize(s.astype(np.float32), (img_size, img_size))
        processed.append(resized)

    stacked = np.stack(processed, axis=0)  # (num_slices, H, W)

    # Normalisasi per volume
    mn, mx = stacked.min(), stacked.max()
    if mx > mn:
        stacked = (stacked - mn) / (mx - mn)
    else:
        stacked = np.zeros_like(stacked)

    return stacked.astype(np.float32)


# ─── DATASET CLASS ───────────────────────────────────────────────────────────
class MRNetDataset(Dataset):
    def __init__(self, split='train', plane='sagittal',
                 num_slices=NUM_SLICES, img_size=IMG_SIZE,
                 augment=False, cache=True):
        """
        Args:
            split  : 'train' atau 'valid'
            plane  : 'sagittal', 'coronal', atau 'axial'
            cache  : load semua ke RAM
        """
        self.split       = split
        self.plane       = plane
        self.augment     = augment
        self.cached_data = []

        # Load labels
        labels_dict = load_mrnet_labels(split)
        self.case_ids = sorted(labels_dict.keys())
        self.labels   = [labels_dict[cid] for cid in self.case_ids]

        data_dir = os.path.join(MRNET_DIR, split, plane)

        if cache:
            print(f"  Caching {len(self.case_ids)} volume "
                  f"[{split}/{plane}] ke RAM...", flush=True)
            for i, case_id in enumerate(self.case_ids):
                path   = os.path.join(data_dir, f'{case_id}.npy')
                volume = np.load(path)
                slices = preprocess_mrnet_volume(volume, num_slices, img_size)
                label  = labels_dict[case_id]
                self.cached_data.append((slices, label))

                if (i + 1) % 200 == 0:
                    print(f"    {i+1}/{len(self.case_ids)} ter-cache...",
                          flush=True)
            print(f"  Cache selesai!", flush=True)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        slices, label = self.cached_data[idx]
        tensor = torch.from_numpy(slices.copy())  # (9, 224, 224)

        if self.augment:
            if np.random.rand() > 0.5:
                tensor = torch.flip(tensor, dims=[2])
            factor = np.random.uniform(0.85, 1.15)
            tensor = torch.clamp(tensor * factor, 0.0, 1.0)

        return tensor, label


# ─── DATALOADER ──────────────────────────────────────────────────────────────
def get_mrnet_dataloaders(plane='sagittal', batch_size=8):
    """
    Return train_loader, val_loader untuk MRNet.
    """
    print(f"\nMRNet DataLoader — plane: {plane}")

    print(f"  [Train]")
    train_ds = MRNetDataset(split='train', plane=plane,
                            augment=True, cache=True)
    print(f"  [Valid]")
    val_ds   = MRNetDataset(split='valid', plane=plane,
                            augment=False, cache=True)

    # Distribusi label
    train_counts = Counter(train_ds.labels)
    val_counts   = Counter(val_ds.labels)
    print(f"\n  Train: {len(train_ds)} sampel "
          f"(Intact={train_counts[0]}, Tear={train_counts[1]})")
    print(f"  Valid: {len(val_ds)} sampel "
          f"(Intact={val_counts[0]}, Tear={val_counts[1]})")

    # WeightedSampler untuk imbalance
    total         = len(train_ds.labels)
    class_weights = {c: total / (2 * train_counts[c])
                     for c in train_counts}
    sample_weights = [class_weights[l] for l in train_ds.labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=True)

    return train_loader, val_loader


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  TEST dataset_mrnet.py")
    print("=" * 55)

    train_loader, val_loader = get_mrnet_dataloaders(
        plane='sagittal', batch_size=4)

    imgs, labels = next(iter(train_loader))
    print(f"\nBatch shape : {imgs.shape}")
    print(f"Min / Max   : {imgs.min():.3f} / {imgs.max():.3f}")
    print(f"Labels      : {[LABEL_NAMES_MRNET[l.item()] for l in labels]}")
    print("\nMRNet dataset siap!")
