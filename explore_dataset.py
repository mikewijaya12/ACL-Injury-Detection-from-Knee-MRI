"""
    Step 1: Eksplorasi Dataset KneeMRI
    Jalankan file ini untuk memahami isi dataset sebelum training
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from collections import Counter

# ─── PATH KONFIGURASI ────────────────────────────────────────────────────────
VOLUMETRIC_DIR = r'D:\ACL\volumetric_data'
METADATA_PATH  = r'D:\ACL\metadata.csv'

# ─── 1. LOAD METADATA ────────────────────────────────────────────────────────
print("=" * 55)
print("  EKSPLORASI DATASET KneeMRI")
print("=" * 55)

metadata = np.genfromtxt(
    METADATA_PATH,
    delimiter=',',
    names=True,
    dtype='i4,i4,i4,i4,i4,i4,i4,i4,i4,i4,U20'
)

print(f"\n[1] Kolom yang tersedia:")
for name in metadata.dtype.names:
    print(f"    - {name}")

print(f"\n[2] Total data (rows): {len(metadata)}")

# ─── 2. DISTRIBUSI LABEL ─────────────────────────────────────────────────────
# aclDiagnosis: 0 = healthy, 1 = partial tear, 2 = complete rupture
labels      = metadata['aclDiagnosis']
label_names = {0: 'Healthy', 1: 'Partial Tear', 2: 'Complete Rupture'}
counts      = Counter(labels)

print(f"\n[3] Distribusi kelas ACL:")
for k, name in label_names.items():
    n   = counts[k]
    pct = n / len(labels) * 100
    bar = '█' * int(pct / 2)
    print(f"    {name:<20} : {n:>4} sampel  ({pct:.1f}%)  {bar}")

# ─── 3. CEK SATU SAMPEL ──────────────────────────────────────────────────────
print(f"\n[4] Contoh satu baris metadata (index 0):")
row = metadata[0]
for name in metadata.dtype.names:
    print(f"    {name:<20} = {row[name]}")

# ─── 4. LOAD SATU VOLUME MRI & TAMPILKAN ─────────────────────────────────────
print(f"\n[5] Load satu volume MRI...")

sample      = metadata[0]
vol_path    = os.path.join(VOLUMETRIC_DIR, sample['volumeFilename'])

with open(vol_path, 'rb') as f:
    volume = pickle.load(f)

print(f"    File     : {sample['volumeFilename']}")
print(f"    Shape    : {volume.shape}  (depth x height x width)")
print(f"    Dtype    : {volume.dtype}")
print(f"    Min/Max  : {volume.min():.2f} / {volume.max():.2f}")
print(f"    Label    : {label_names[sample['aclDiagnosis']]}")

# ─── 5. VISUALISASI ──────────────────────────────────────────────────────────
z_start = sample['roiZ']
depth   = sample['roiDepth']
x, y, w, h = sample['roiX'], sample['roiY'], sample['roiWidth'], sample['roiHeight']

# Pilih slice tengah dari ROI
z_mid  = z_start + depth // 2
slice_ = volume[z_mid, :, :]
roi    = slice_[y:y+h, x:x+w]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    f"KneeMRI — {label_names[sample['aclDiagnosis']]} "
    f"(examId: {sample['examId']})",
    fontsize=13, fontweight='bold'
)

# Slice penuh
axes[0].imshow(slice_, cmap='gray')
axes[0].add_patch(patch.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2))
axes[0].set_title('Full Slice (ROI = kotak merah)')
axes[0].axis('off')

# ROI crop
axes[1].imshow(roi, cmap='gray')
axes[1].set_title('ROI — Area ACL')
axes[1].axis('off')

# Histogram intensitas
axes[2].hist(slice_.flatten(), bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
axes[2].set_title('Histogram Intensitas Pixel')
axes[2].set_xlabel('Nilai Pixel')
axes[2].set_ylabel('Frekuensi')

plt.tight_layout()
plt.savefig(r'D:\ACL\sample_visualization.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"\n[6] Gambar tersimpan di: D:\\ACL\\sample_visualization.png")

# ─── 6. CEK SEMUA VOLUME (statistik ukuran) ──────────────────────────────────
print(f"\n[7] Mengecek ukuran semua volume (harap tunggu...)")

shapes = []
for i, row in enumerate(metadata):
    path = os.path.join(VOLUMETRIC_DIR, row['volumeFilename'])
    with open(path, 'rb') as f:
        vol = pickle.load(f)
    shapes.append(vol.shape)
    if (i + 1) % 100 == 0:
        print(f"    Sudah cek {i+1}/{len(metadata)} volume...")

depths   = [s[0] for s in shapes]
heights  = [s[1] for s in shapes]
widths   = [s[2] for s in shapes]

print(f"\n[8] Statistik ukuran volume:")
print(f"    Depth  (slices) — min: {min(depths)}, max: {max(depths)}, rata2: {np.mean(depths):.1f}")
print(f"    Height          — min: {min(heights)}, max: {max(heights)}, rata2: {np.mean(heights):.1f}")
print(f"    Width           — min: {min(widths)}, max: {max(widths)}, rata2: {np.mean(widths):.1f}")

print(f"\n{'=' * 55}")
print(f"  Eksplorasi selesai! Dataset siap diproses.")
print(f"{'=' * 55}\n")