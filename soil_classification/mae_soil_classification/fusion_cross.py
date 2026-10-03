from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, num_modalities, d_model]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class FusionHeadV2(nn.Module):
    def __init__(
        self,
        in_dims: Dict[str, int],
        d_model: int = 64,
        out_dim: int = 1,
        dropout: float = 0.1,
        concat_order: Optional[List[str]] = None,
        use_presence_mask: bool = True,
        num_heads: int = 4,
        use_cross_attention: bool = True,
        use_rf: bool = False,
    ):
        super().__init__()

        self.modalities = concat_order or ["s2", "s1", "planet", "climate"]
        self.adapters = nn.ModuleDict()
        for m in self.modalities:
            in_d = in_dims[m] if m in in_dims else None
            if in_d is not None:
                self.adapters[m] = nn.Sequential(
                    nn.LayerNorm(in_d),
                    nn.Linear(in_d, d_model),
                    nn.GELU(),
                )

        self.d_model = d_model
        self.use_presence_mask = use_presence_mask
        self.use_cross_attention = use_cross_attention
        self.use_rf = use_rf

        if use_cross_attention:
            self.cross_attn = CrossAttentionBlock(d_model, num_heads=num_heads, dropout=dropout)

        self.max_concat_dim = d_model * len(self.modalities) + (
            len(self.modalities) if use_presence_mask else 0
        )

        if not self.use_rf:
            #  normal MLP head
            self.mlp = nn.Sequential(
                nn.LayerNorm(self.max_concat_dim),
                nn.Linear(self.max_concat_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, max(32, self.max_concat_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(32, self.max_concat_dim // 2), out_dim)
            )
        else:
            #  RF mode: only produce embeddings
            hidden_dim = max(32, self.max_concat_dim // 2)
            self.feature_proj = nn.Sequential(
                nn.LayerNorm(self.max_concat_dim),
                nn.Dropout(dropout),
                nn.Linear(self.max_concat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),  # keep embedding dimension same as hidden layer
            )

    def forward(self, batch: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        B = None
        presence = batch.get("presence", None)
        if presence is not None:
            B = presence.shape[0]

        per_mod_feats = []
        for j, m in enumerate(self.modalities):
            x = batch.get(m, None)
            if x is None:
                if B is None:
                    continue
                z = torch.zeros(B, self.d_model, device=presence.device if presence is not None else "cpu")
                per_mod_feats.append(z)
                continue

            adapter = self.adapters[m]
            z = adapter(x)
            if presence is not None:
                pm = presence[:, j].unsqueeze(1)
                z = z * pm
            per_mod_feats.append(z)

        if len(per_mod_feats) == 0:
            raise ValueError("No modalities found in batch")

        # [B, num_modalities, d_model]
        feat_seq = torch.stack(per_mod_feats, dim=1)

        if self.use_cross_attention:
            feat_seq = self.cross_attn(feat_seq)

        feat = feat_seq.reshape(feat_seq.shape[0], -1)  # flatten [B, num_modalities*d_model]

        if self.use_presence_mask:
            if presence is None:
                device = feat.device
                presence = torch.tensor(
                    [1.0 if batch[m] is not None else 0.0 for m in self.modalities],
                    dtype=feat.dtype,
                    device=device,
                ).unsqueeze(0).expand(B, len(self.modalities))
            feat = torch.cat([feat, presence], dim=1)

        if feat.shape[1] < self.max_concat_dim:
            feat = F.pad(feat, (0, self.max_concat_dim - feat.shape[1]))
        elif feat.shape[1] > self.max_concat_dim:
            feat = feat[:, :self.max_concat_dim]

        # return embedding for RF mode, else return mlp logits
        if self.use_rf:
            return self.feature_proj(feat)   # [B, hidden_dim] embedding for RF
        else:
            return self.mlp(feat)            # [B, out_dim] logits/prediction