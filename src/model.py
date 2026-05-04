"""
    model.py
    --------
    Arsitektur Hybrid CNN - Vision Transformer untuk deteksi ACL injury.

    Alur:
        Input (B, 9, 224, 224)
            ↓
        CNN Backbone (EfficientNet-B0)   ← ekstrak fitur lokal tiap slice
            ↓
        Feature Map per slice (B, 9, 1280)
            ↓
        Vision Transformer Encoder       ← tangkap relasi global antar slice
            ↓
        Classification Head              ← output 3 kelas
"""

import torch
import torch.nn as nn
import timm


# ─── HYBRID CNN-VIT ───────────────────────────────────────────────────────────
class HybridCNNViT(nn.Module):
    def __init__(
        self,
        num_classes  = 3,       # Healthy / Partial / Complete
        num_slices   = 9,       # jumlah slice per volume
        cnn_name     = 'efficientnet_b0',
        embed_dim    = 256,     # dimensi embedding untuk ViT
        num_heads    = 8,       # jumlah attention heads
        num_layers   = 4,       # jumlah transformer encoder layers
        dropout      = 0.1,
        pretrained   = True
    ):
        super(HybridCNNViT, self).__init__()

        self.num_slices = num_slices
        self.embed_dim  = embed_dim

        # ── 1. CNN BACKBONE ───────────────────────────────────────────────────
        # EfficientNet-B0 pretrained ImageNet
        # in_chans=1 karena MRI grayscale (1 channel per slice)
        self.cnn = timm.create_model(
            cnn_name,
            pretrained   = pretrained,
            in_chans     = 1,
            num_classes  = 0,       # hapus classification head bawaan
            global_pool  = 'avg'    # output: (B, 1280)
        )
        cnn_out_dim = self.cnn.num_features  # 1280 untuk EfficientNet-B0

        # ── 2. PROJECTION LAYER ───────────────────────────────────────────────
        # Projeksikan fitur CNN ke dimensi embed_dim untuk ViT
        self.projection = nn.Sequential(
            nn.Linear(cnn_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # ── 3. POSITIONAL ENCODING ────────────────────────────────────────────
        # Learnable positional encoding untuk tiap slice
        # +1 untuk [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── 4. VISION TRANSFORMER ENCODER ────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,     # input: (B, seq, dim)
            norm_first      = True      # Pre-LN lebih stabil
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers
        )

        # ── 5. CLASSIFICATION HEAD ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # ── 6. DROPOUT ────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: (B, num_slices, H, W)  → misal (8, 9, 224, 224)
        """
        B, S, H, W = x.shape  # B=batch, S=slices, H=height, W=width

        # ── Step 1: CNN per slice ─────────────────────────────────────────────
        # Reshape: gabung batch & slice jadi satu dimensi
        x = x.view(B * S, 1, H, W)          # (B*S, 1, 224, 224)

        # Forward CNN
        features = self.cnn(x)               # (B*S, 1280)

        # Project ke embed_dim
        features = self.projection(features) # (B*S, 256)

        # Reshape balik: pisahkan batch & slice
        features = features.view(B, S, self.embed_dim)  # (B, 9, 256)

        # ── Step 2: Tambahkan CLS token ───────────────────────────────────────
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, 256)
        features   = torch.cat([cls_tokens, features], dim=1)  # (B, 10, 256)

        # ── Step 3: Tambahkan positional encoding ─────────────────────────────
        features = features + self.pos_embed             # (B, 10, 256)
        features = self.dropout(features)

        # ── Step 4: Transformer Encoder ───────────────────────────────────────
        features = self.transformer(features)            # (B, 10, 256)

        # ── Step 5: Ambil CLS token untuk klasifikasi ─────────────────────────
        cls_output = features[:, 0, :]                  # (B, 256)

        # ── Step 6: Classification ────────────────────────────────────────────
        logits = self.classifier(cls_output)             # (B, 3)

        return logits


# ─── FUNGSI HITUNG PARAMETER ─────────────────────────────────────────────────
def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─── TEST CEPAT ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  TEST model.py — Hybrid CNN-ViT")
    print("=" * 55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Buat model
    model = HybridCNNViT(
        num_classes = 3,
        num_slices  = 9,
        embed_dim   = 256,
        num_heads   = 8,
        num_layers  = 4,
        pretrained  = True
    ).to(device)

    # Hitung parameter
    total, trainable = count_parameters(model)
    print(f"\nJumlah parameter:")
    print(f"  Total     : {total:,}")
    print(f"  Trainable : {trainable:,}")

    # Test forward pass dengan dummy input
    print(f"\nTest forward pass...")
    dummy = torch.randn(4, 9, 224, 224).to(device)  # batch=4
    with torch.no_grad():
        output = model(dummy)

    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {output.shape}")   # harus (4, 3)
    print(f"  Output sample: {output[0].cpu().numpy()}")

    # Cek softmax probabilities
    probs = torch.softmax(output, dim=1)
    print(f"\nContoh probabilitas (batch[0]):")
    labels = ['Healthy', 'Partial Tear', 'Complete Rupture']
    for i, (label, prob) in enumerate(zip(labels, probs[0])):
        print(f"  {label:<20}: {prob.item():.4f}")

    print(f"\nModel siap! Lanjut ke train.py")