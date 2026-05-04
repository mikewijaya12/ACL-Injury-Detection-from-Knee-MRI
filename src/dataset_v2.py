"""
    dataset_v2.py
    -------------
    Dataset tanpa ROI crop — pakai full slice.
    Model sendiri yang belajar area ACL via attention.
    Ini yang membedakan dari dataset.py versi pertama.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from collections import Counter
import warnings

# ─── KONSTANTA ────────────────────────────────────────────────────────────────
VOLUMETRIC_DIR = r'D:\ACL\volumetric_data'
METADATA_PATH  = r'D:\ACL\metadata.csv'
NUM_SLICES     = 9
IMG_SIZE       = 224
LABEL_NAMES    = {0: 'Healthy', 1: 'Partial Tear', 2: 'Complete Rupture'}


def load_metadata(metadata_path=METADATA_PATH):
    return np.genfromtxt(
        metadata_path, delimiter=',', names=True,
        dtype='i4,i4,i4,i4,i4,i4,i4,i4,i4,i4,U20'
    )


def preprocess_volume_full(volume, exam_row,
                            num_slices=NUM_SLICES,
                            img_size=IMG_SIZE):
    """
    BERBEDA dari v1: tidak crop ROI.
    Ambil full slice dari seluruh volume,
    biarkan model yang temukan area ACL sendiri.
    """
    z_start = exam_row['roiZ']
    depth   = exam_row['roiDepth']

    # Ambil slice dari range ROI (tapi tidak crop x,y)
    roi_slices = volume[z_start: z_start + depth, :, :]
    total      = roi_slices.shape[0]
    indices    = np.linspace(0, total - 1, num_slices, dtype=int)
    selected   = roi_slices[indices]  # (9, H, W) — FULL slice

    processed = []
    for s in selected:
        # Resize full slice ke img_size (tanpa crop)
        resized = cv2.resize(s.astype(np.float32),
                             (img_size, img_size))
        processed.append(resized)

    stacked = np.stack(processed, axis=0)  # (9, 224, 224)

    # Normalisasi
    mn, mx = stacked.min(), stacked.max()
    if mx > mn:
        stacked = (stacked - mn) / (mx - mn)
    else:
        stacked = np.zeros_like(stacked)

    return stacked.astype(np.float32)


class KneeMRIDatasetV2(Dataset):
    """Dataset v2 — full slice, no ROI crop."""

    def __init__(self, metadata, volumetric_dir=VOLUMETRIC_DIR,
                 num_slices=NUM_SLICES, img_size=IMG_SIZE,
                 augment=False, cache=True):
        self.metadata    = metadata
        self.augment     = augment
        self.cached_data = []

        if cache:
            print(f"  Caching {len(metadata)} volume (full slice)...",
                  flush=True)
            for i, exam in enumerate(metadata):
                path = os.path.join(volumetric_dir, exam['volumeFilename'])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(path, 'rb') as f:
                        volume = pickle.load(f)
                processed = preprocess_volume_full(
                    volume, exam, num_slices, img_size)
                label = int(exam['aclDiagnosis'])
                self.cached_data.append((processed, label))
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(metadata)} ter-cache...",
                          flush=True)
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
                angle     = np.random.uniform(-15, 15)
                slices_aug = []
                for i in range(tensor.shape[0]):
                    s    = tensor[i].numpy()
                    h_s, w_s = s.shape
                    M    = cv2.getRotationMatrix2D(
                        (w_s/2, h_s/2), angle, 1.0)
                    rot  = cv2.warpAffine(s, M, (w_s, h_s))
                    slices_aug.append(rot)
                tensor = torch.from_numpy(
                    np.stack(slices_aug).astype(np.float32))
            factor = np.random.uniform(0.8, 1.2)
            tensor = torch.clamp(tensor * factor, 0.0, 1.0)

        return tensor, label


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


def get_dataloaders_v2(batch_size=8):
    metadata = load_metadata()
    train_meta, val_meta, test_meta = split_dataset(metadata)

    print(f"\nSplit dataset (v2 — full slice):")
    print(f"  Train : {len(train_meta)} samples")
    print(f"  Val   : {len(val_meta)} samples")
    print(f"  Test  : {len(test_meta)} samples")

    print(f"\nLoading + caching:")
    print(f"  [Train]")
    train_ds = KneeMRIDatasetV2(train_meta, augment=True,  cache=True)
    print(f"  [Val]")
    val_ds   = KneeMRIDatasetV2(val_meta,   augment=False, cache=True)
    print(f"  [Test]")
    test_ds  = KneeMRIDatasetV2(test_meta,  augment=False, cache=True)

    labels         = train_meta['aclDiagnosis']
    counts         = Counter(labels)
    class_weights  = {c: 1.0 / counts[c] for c in counts}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=True)

    return train_loader, val_loader, test_loader, train_meta