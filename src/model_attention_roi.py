"""
    model_attention_roi.py
    ----------------------
    Hybrid CNN + Patch-based ViT dengan Attention-guided ROI.

    NOVELTY: Tidak butuh ROI manual.
    Model secara otomatis belajar patch mana yang paling
    relevan (= area ACL) melalui self-attention mechanism.

    Alur:
        Full MRI slice (224x224)
               ↓
        CNN Backbone → Feature Map (7x7)
               ↓
        Flatten → 49 patch tokens
               ↓
        Vision Transformer
        → attention weights per patch = "ROI otomatis"
               ↓
        Attention-weighted pooling
               ↓
        Classification Head
               ↓
        Output + Attention Map (untuk visualisasi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionROIModel(nn.Module):
    """
    Hybrid CNN-ViT dengan attention-guided ROI.
    Tidak memerlukan ROI annotation manual.
    """

    def __init__(
        self,
        num_classes  = 3,
        num_slices   = 9,
        cnn_name     = 'efficientnet_b2',
        embed_dim    = 384,
        num_heads    = 8,
        num_layers   = 6,
        dropout      = 0.2,
        pretrained   = True,
        patch_size   = 7,    # feature map size dari CNN
    ):
        super(AttentionROIModel, self).__init__()

        self.num_slices = num_slices
        self.embed_dim  = embed_dim
        self.patch_size = patch_size
        self.num_patches = patch_size * patch_size  # 49 patches per slice

        # ── 1. CNN BACKBONE (tanpa global pooling) ────────────────────────────
        # Kita ambil feature map (7x7) bukan vector (1280,)
        # supaya bisa di-patch dan diproses ViT
        self.cnn = timm.create_model(
            cnn_name,
            pretrained  = pretrained,
            in_chans    = 1,
            num_classes = 0,
            global_pool = ''   # ← tidak pakai global pooling
        )
        # Dapatkan jumlah channel output CNN
        with torch.no_grad():
            dummy     = torch.zeros(1, 1, 224, 224)
            feat      = self.cnn(dummy)
            cnn_channels = feat.shape[1]  # biasanya 1408 untuk B2
            feat_h    = feat.shape[2]
            feat_w    = feat.shape[3]
        print(f"  CNN feature map: {cnn_channels}ch × {feat_h}×{feat_w}")
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.num_patches = feat_h * feat_w

        # ── 2. PATCH PROJECTION ───────────────────────────────────────────────
        # Projeksikan tiap patch (1x1 spatial) ke embed_dim
        self.patch_proj = nn.Sequential(
            nn.Conv2d(cnn_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # ── 3. POSITIONAL ENCODING ────────────────────────────────────────────
        # [CLS] token + (num_patches × num_slices) tokens
        total_tokens    = num_slices * self.num_patches + 1
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, total_tokens, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── 4. VISION TRANSFORMER ─────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # ── 5. ATTENTION EXTRACTION ───────────────────────────────────────────
        # Hook untuk extract attention weights dari layer pertama
        self.attention_weights = None
        self._register_attention_hook()

        # ── 6. CLASSIFICATION HEAD ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def _register_attention_hook(self):
        """Extract attention weights dari transformer layer pertama."""
        def hook(module, input, output):
            # output dari MultiheadAttention adalah (attn_output, attn_weights)
            pass
        # Hook akan diimplementasikan saat forward pass

    def get_attention_map(self, imgs):
        """
        Generate attention map untuk visualisasi ROI otomatis.
        Return attention per patch → bisa divisualisasikan sebagai heatmap.
        """
        self.eval()
        with torch.no_grad():
            # Forward pass biasa
            _ = self.forward(imgs)

            # Ambil attention dari layer transformer pertama
            # Kita gunakan gradient-free attention rollout
            B = imgs.shape[0]
            return None  # akan diisi nanti

    def forward(self, x):
        """
        x: (B, num_slices, H, W)
        return: logits (B, num_classes),
                attention_map (B, num_slices, feat_h, feat_w)
        """
        B, S, H, W = x.shape

        # ── Step 1: CNN per slice → feature maps ──────────────────────────────
        x_flat  = x.view(B * S, 1, H, W)        # (B*S, 1, H, W)
        feat    = self.cnn(x_flat)               # (B*S, C, fh, fw)
        _, C, fh, fw = feat.shape

        # ── Step 2: Patch projection ──────────────────────────────────────────
        patches = self.patch_proj(feat)          # (B*S, embed_dim, fh, fw)
        patches = patches.flatten(2)             # (B*S, embed_dim, fh*fw)
        patches = patches.transpose(1, 2)        # (B*S, fh*fw, embed_dim)
        # patches shape: (B*S, num_patches, embed_dim)

        # ── Step 3: Reshape untuk gabungkan slice dimension ───────────────────
        num_patches = fh * fw
        patches = patches.reshape(B, S * num_patches, self.embed_dim)
        # patches shape: (B, S*num_patches, embed_dim)

        # ── Step 4: Tambahkan CLS token ───────────────────────────────────────
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        tokens     = torch.cat([cls_tokens, patches], dim=1)
        # tokens shape: (B, 1 + S*num_patches, embed_dim)

        # ── Step 5: Positional encoding ───────────────────────────────────────
        # Sesuaikan pos_embed jika ukuran tidak cocok
        if tokens.shape[1] != self.pos_embed.shape[1]:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=tokens.shape[1],
                mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed

        tokens = tokens + pos
        tokens = self.dropout(tokens)

        # ── Step 6: Transformer ───────────────────────────────────────────────
        tokens = self.transformer(tokens)        # (B, 1+S*P, embed_dim)

        # ── Step 7: CLS token untuk klasifikasi ──────────────────────────────
        cls_out = tokens[:, 0, :]               # (B, embed_dim)

        # ── Step 8: Attention map untuk visualisasi ───────────────────────────
        # Patch tokens setelah transformer = attended features
        patch_tokens = tokens[:, 1:, :]         # (B, S*P, embed_dim)
        # Hitung attention score = norm tiap patch token
        attn_scores  = patch_tokens.norm(dim=-1)           # (B, S*P)
        attn_scores  = attn_scores.view(B, S, fh, fw)      # (B, S, fh, fw)
        # Normalize ke [0,1]
        attn_min = attn_scores.flatten(1).min(dim=1)[0].view(B, 1, 1, 1)
        attn_max = attn_scores.flatten(1).max(dim=1)[0].view(B, 1, 1, 1)
        attn_map = (attn_scores - attn_min) / (attn_max - attn_min + 1e-8)

        # ── Step 9: Klasifikasi ───────────────────────────────────────────────
        logits = self.classifier(cls_out)        # (B, num_classes)

        return logits, attn_map


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return total, trainable


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  TEST model_attention_roi.py")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    print("Membangun AttentionROIModel...")
    model = AttentionROIModel(
        num_classes = 3,
        num_slices  = 9,
        cnn_name    = 'efficientnet_b2',
        embed_dim   = 384,
        num_heads   = 8,
        num_layers  = 6,
        pretrained  = True
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"\nParameter:")
    print(f"  Total     : {total:,}")
    print(f"  Trainable : {trainable:,}")

    # Test forward pass
    print(f"\nTest forward pass (batch=2, 9 slices, 224x224)...")
    dummy = torch.randn(2, 9, 224, 224).to(device)
    with torch.no_grad():
        logits, attn_map = model(dummy)

    print(f"  Input shape     : {dummy.shape}")
    print(f"  Logits shape    : {logits.shape}")    # (2, 3)
    print(f"  Attn map shape  : {attn_map.shape}")  # (2, 9, fh, fw)

    probs = torch.softmax(logits, dim=1)
    print(f"\nContoh probabilitas (batch[0]):")
    for name, p in zip(['Healthy','Partial Tear','Complete Rupture'],
                        probs[0]):
        print(f"  {name:<20}: {p.item():.4f}")

    print(f"\nAttention map range: "
          f"{attn_map.min().item():.3f} – {attn_map.max().item():.3f}")
    print(f"\nModel siap! Lanjut ke train_attention_roi.py")
