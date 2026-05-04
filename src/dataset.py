"""
    dataset.py (v2 - dengan caching)
    ---------------------------------
    Semua volume di-load ke RAM saat inisialisasi.
    Ini mempercepat training dan menghilangkan I/O bottleneck.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from collections import Counter

# ─── KONSTANTA ────────────────────────────────────────────────────────────────
VOLUMETRIC_DIR = r'D:\ACL\volumetric_data'
METADATA_PATH  = r'D:\ACL\metadata.csv'
NUM_SLICES     = 9
IMG_SIZE       = 224
LABEL_NAMES    = {0: 'Healthy', 1: 'Partial Tear', 2: 'Complete Rupture'}


# ─── LOAD METADATA ────────────────────────────────────────────────────────────
def load_metadata(metadata_path=METADATA_PATH):
    return np.genfromtxt(
        metadata_path, delimiter=',', names=True,
        dtype='i4,i4,i4,i4,i4,i4,i4,i4,i4,i4,U20'
    )


# ─── PREPROCESS SATU VOLUME ──────────────────────────────────────────────────
def preprocess_volume(volume, exam_row, num_slices=NUM_SLICES, img_size=IMG_SIZE):
    x       = exam_row['roiX']
    y       = exam_row['roiY']
    w       = exam_row['roiWidth']
    h       = exam_row['roiHeight']
    z_start = exam_row['roiZ']
    depth   = exam_row['roiDepth']

    roi_slices = volume[z_start: z_start + depth, :, :]
    total      = roi_slices.shape[0]
    indices    = np.linspace(0, total - 1, num_slices, dtype=int)
    selected   = roi_slices[indices]

    processed = []
    for s in selected:
        y1 = max(0, y); y2 = min(s.shape[0], y + h)
        x1 = max(0, x); x2 = min(s.shape[1], x + w)
        crop    = s[y1:y2, x1:x2]
        resized = cv2.resize(crop.astype(np.float32), (img_size, img_size))
        processed.append(resized)

    stacked = np.stack(processed, axis=0)
    mn, mx  = stacked.min(), stacked.max()
    if mx > mn:
        stacked = (stacked - mn) / (mx - mn)
    else:
        stacked = np.zeros_like(stacked)

    return stacked.astype(np.float32)


# ─── DATASET CLASS (dengan cache) ─────────────────────────────────────────────
class KneeMRIDataset(Dataset):
    def __init__(self, metadata, volumetric_dir=VOLUMETRIC_DIR,
                 num_slices=NUM_SLICES, img_size=IMG_SIZE,
                 augment=False, cache=True):
        self.metadata    = metadata
        self.augment     = augment
        self.cached_data = []

        if cache:
            print(f"  Caching {len(metadata)} volume ke RAM...", flush=True)
            import warnings
            for i, exam in enumerate(metadata):
                path = os.path.join(volumetric_dir, exam['volumeFilename'])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(path, 'rb') as f:
                        volume = pickle.load(f)
                processed = preprocess_volume(volume, exam, num_slices, img_size)
                label     = int(exam['aclDiagnosis'])
                self.cached_data.append((processed, label))
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(metadata)} ter-cache...", flush=True)
            print(f"  Cache selesai!", flush=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        slices, label = self.cached_data[idx]
        tensor = torch.from_numpy(slices.copy())

        if self.augment:
            if np.random.rand() > 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if np.random.rand() > 0.5:
                angle     = np.random.uniform(-10, 10)
                slices_aug = []
                for i in range(tensor.shape[0]):
                    s    = tensor[i].numpy()
                    h_s, w_s = s.shape
                    M    = cv2.getRotationMatrix2D((w_s/2, h_s/2), angle, 1.0)
                    rot  = cv2.warpAffine(s, M, (w_s, h_s))
                    slices_aug.append(rot)
                tensor = torch.from_numpy(np.stack(slices_aug).astype(np.float32))
            factor = np.random.uniform(0.85, 1.15)
            tensor = torch.clamp(tensor * factor, 0.0, 1.0)

        return tensor, label


# ─── SPLIT DATA ───────────────────────────────────────────────────────────────
def split_dataset(metadata, train_ratio=0.70, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in [0, 1, 2]:
        idx = np.where(metadata['aclDiagnosis'] == label)[0]
        np.random.shuffle(idx)
        n       = len(idx)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return metadata[train_idx], metadata[val_idx], metadata[test_idx]


# ─── DATALOADER ───────────────────────────────────────────────────────────────
def get_dataloaders(batch_size=8):
    metadata = load_metadata()
    train_meta, val_meta, test_meta = split_dataset(metadata)

    print(f"\nSplit dataset:")
    print(f"  Train : {len(train_meta)} samples")
    print(f"  Val   : {len(val_meta)} samples")
    print(f"  Test  : {len(test_meta)} samples")

    print(f"\nLoading + caching data ke RAM:")
    print(f"  [Train]")
    train_ds = KneeMRIDataset(train_meta, augment=True,  cache=True)
    print(f"  [Val]")
    val_ds   = KneeMRIDataset(val_meta,   augment=False, cache=True)
    print(f"  [Test]")
    test_ds  = KneeMRIDataset(test_meta,  augment=False, cache=True)

    labels         = train_meta['aclDiagnosis']
    counts         = Counter(labels)
    class_weights  = {c: 1.0 / counts[c] for c in counts}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  TEST dataset.py v2 (with cache)")
    print("=" * 50)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    imgs, labels = next(iter(train_loader))
    print(f"\nBatch shape : {imgs.shape}")
    print(f"Min / Max   : {imgs.min():.3f} / {imgs.max():.3f}")
    print(f"Labels      : {[LABEL_NAMES[l.item()] for l in labels]}")
    print("\nDataset v2 siap!")