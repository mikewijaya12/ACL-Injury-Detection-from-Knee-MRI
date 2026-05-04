"""
    dataset.py
    ----------
    PyTorch Dataset class untuk KneeMRI.
    - Load volume .pkl
    - Crop ROI area ACL
    - Ambil 9 slice tengah
    - Resize ke 224x224
    - Normalisasi ke [0, 1]
    - Augmentasi untuk training set
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import cv2
from collections import Counter

# ─── KONSTANTA ────────────────────────────────────────────────────────────────
VOLUMETRIC_DIR = r'D:\ACL\volumetric_data'
METADATA_PATH  = r'D:\ACL\metadata.csv'
NUM_SLICES     = 9       # jumlah slice yang diambil per volume
IMG_SIZE       = 224     # ukuran resize (224x224)
LABEL_NAMES    = {0: 'Healthy', 1: 'Partial Tear', 2: 'Complete Rupture'}


# ─── FUNGSI LOAD METADATA ─────────────────────────────────────────────────────
def load_metadata(metadata_path=METADATA_PATH):
    metadata = np.genfromtxt(
        metadata_path,
        delimiter=',',
        names=True,
        dtype='i4,i4,i4,i4,i4,i4,i4,i4,i4,i4,U20'
    )
    return metadata


# ─── FUNGSI PREPROCESS SATU VOLUME ───────────────────────────────────────────
def preprocess_volume(volume, exam_row, num_slices=NUM_SLICES, img_size=IMG_SIZE):
    """
    Input  : volume (D, H, W) dan satu baris metadata
    Output : tensor (num_slices, img_size, img_size) float32
    """
    # 1. Ambil koordinat ROI
    x = exam_row['roiX']
    y = exam_row['roiY']
    w = exam_row['roiWidth']
    h = exam_row['roiHeight']
    z_start = exam_row['roiZ']
    depth   = exam_row['roiDepth']

    # 2. Ambil slice-slice dalam range ROI
    roi_slices = volume[z_start : z_start + depth, :, :]  # (depth, H, W)

    # 3. Pilih num_slices slice dari tengah secara merata
    total = roi_slices.shape[0]
    if total >= num_slices:
        # ambil indeks merata dari tengah
        indices = np.linspace(0, total - 1, num_slices, dtype=int)
    else:
        # kalau slice-nya kurang, repeat yang ada
        indices = np.linspace(0, total - 1, num_slices, dtype=int)
    
    selected_slices = roi_slices[indices]  # (num_slices, H, W)

    # 4. Crop ROI dan resize setiap slice
    processed = []
    for s in selected_slices:
        # Crop area ACL
        roi_crop = s[y:y+h, x:x+w]
        
        # Resize ke img_size x img_size
        resized = cv2.resize(roi_crop.astype(np.float32), (img_size, img_size))
        processed.append(resized)

    # 5. Stack jadi array (num_slices, H, W)
    stacked = np.stack(processed, axis=0)  # (9, 224, 224)

    # 6. Normalisasi ke [0, 1]
    min_val = stacked.min()
    max_val = stacked.max()
    if max_val > min_val:
        stacked = (stacked - min_val) / (max_val - min_val)
    else:
        stacked = np.zeros_like(stacked)

    return stacked.astype(np.float32)


# ─── PYTORCH DATASET CLASS ────────────────────────────────────────────────────
class KneeMRIDataset(Dataset):
    def __init__(self, metadata, volumetric_dir=VOLUMETRIC_DIR,
                 num_slices=NUM_SLICES, img_size=IMG_SIZE, augment=False):
        """
        Args:
            metadata      : array hasil load_metadata()
            volumetric_dir: folder berisi file .pkl
            num_slices    : jumlah slice per volume
            img_size      : ukuran resize
            augment       : True untuk training (aktifkan augmentasi)
        """
        self.metadata       = metadata
        self.volumetric_dir = volumetric_dir
        self.num_slices     = num_slices
        self.img_size       = img_size
        self.augment        = augment

        # Augmentasi hanya untuk training
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        exam = self.metadata[idx]
        
        # Load volume
        vol_path = os.path.join(self.volumetric_dir, exam['volumeFilename'])
        with open(vol_path, 'rb') as f:
            volume = pickle.load(f)

        # Preprocess
        slices = preprocess_volume(volume, exam, self.num_slices, self.img_size)
        # slices shape: (9, 224, 224)

        # Convert ke tensor
        tensor = torch.from_numpy(slices)  # (9, 224, 224)

        # Augmentasi (hanya saat training)
        if self.augment:
            # Augmentasi per slice
            augmented = []
            for i in range(tensor.shape[0]):
                s = tensor[i].unsqueeze(0)  # (1, 224, 224)
                # Convert ke PIL untuk transforms
                from torchvision.transforms.functional import to_pil_image, to_tensor
                pil_img = to_pil_image(s)
                aug_img = self.aug_transforms(pil_img)
                augmented.append(to_tensor(aug_img))
            tensor = torch.cat(augmented, dim=0)  # (9, 224, 224)

        label = int(exam['aclDiagnosis'])
        return tensor, label


# ─── FUNGSI SPLIT DATA ────────────────────────────────────────────────────────
def split_dataset(metadata, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split metadata menjadi train / val / test
    dengan stratified split (proporsi kelas tetap seimbang)
    """
    np.random.seed(seed)

    # Pisahkan per kelas dulu (stratified)
    indices_per_class = {}
    for label in [0, 1, 2]:
        idx = np.where(metadata['aclDiagnosis'] == label)[0]
        np.random.shuffle(idx)
        indices_per_class[label] = idx

    train_idx, val_idx, test_idx = [], [], []

    for label, idx in indices_per_class.items():
        n       = len(idx)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    return (
        metadata[train_idx],
        metadata[val_idx],
        metadata[test_idx]
    )


# ─── FUNGSI BUAT DATALOADER ──────────────────────────────────────────────────
def get_dataloaders(batch_size=8):
    """
    Return train_loader, val_loader, test_loader
    """
    metadata = load_metadata()

    # Split
    train_meta, val_meta, test_meta = split_dataset(metadata)

    print(f"Split dataset:")
    print(f"  Train : {len(train_meta)} samples")
    print(f"  Val   : {len(val_meta)} samples")
    print(f"  Test  : {len(test_meta)} samples")

    # Dataset
    train_dataset = KneeMRIDataset(train_meta, augment=True)
    val_dataset   = KneeMRIDataset(val_meta,   augment=False)
    test_dataset  = KneeMRIDataset(test_meta,  augment=False)

    # WeightedRandomSampler untuk mengatasi class imbalance di training
    labels      = train_meta['aclDiagnosis']
    counts      = Counter(labels)
    class_weights = {c: 1.0 / counts[c] for c in counts}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = 0,    # Windows: pakai 0 dulu
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True
    )

    return train_loader, val_loader, test_loader


# ─── TEST CEPAT ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  TEST dataset.py")
    print("=" * 50)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)

    print("\nAmbil 1 batch dari train_loader...")
    batch_imgs, batch_labels = next(iter(train_loader))

    print(f"  Shape batch  : {batch_imgs.shape}")   # (4, 9, 224, 224)
    print(f"  Dtype        : {batch_imgs.dtype}")
    print(f"  Min / Max    : {batch_imgs.min():.3f} / {batch_imgs.max():.3f}")
    print(f"  Labels       : {[LABEL_NAMES[l.item()] for l in batch_labels]}")

    print("\nDataset siap! Lanjut ke model.py")