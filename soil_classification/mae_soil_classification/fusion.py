# Fusion head with per-sample missing-modality support.
# Accepts optional key 'presence' in the batch: a [B, M] float tensor with 1=present, 0=missing
# in the order given by concat_order. If 'presence' is absent, falls back to dataset-level presence.
# fusion_head_v2.py

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionHeadV2(nn.Module):
    def __init__(
        self,
        in_dims: Dict[str, int],
        d_model: int = 64,
        out_dim: int = 1,
        dropout: float = 0.1,
        concat_order: Optional[List[str]] = None,
        use_presence_mask: bool = True,
        use_rf: bool = False
    ):
        super().__init__()

        self.use_rf = use_rf
        self.modalities = concat_order or ["s2", "s1", "planet", "climate"]
        self.adapters = nn.ModuleDict()
        for m in self.modalities:
            in_d = in_dims[m] if m in in_dims else None
            if in_d is not None:
                self.adapters[m] = nn.Sequential(
                    nn.LayerNorm(in_d),
                    nn.Linear(in_d, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )

        self.d_model = d_model
        self.use_presence_mask = use_presence_mask
        self.max_concat_dim = d_model * len(self.modalities) + (len(self.modalities) if use_presence_mask else 0)

        # self.mlp = nn.Sequential(
        #     nn.LayerNorm(self.max_concat_dim),
        #     nn.Linear(self.max_concat_dim, 128),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(128, max(32, self.max_concat_dim // 2)),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(max(32, self.max_concat_dim // 2), out_dim)
        # )
        
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
        presence = batch["presence"] if "presence" in batch else None
        if presence is not None:
            if presence.dim() != 2 or presence.shape[1] != len(self.modalities):
                raise ValueError(f"'presence' must be [B, {len(self.modalities)}]")
            B = presence.shape[0]

        per_mod_feats = []
        for j, m in enumerate(self.modalities):
            x = batch[m] if m in batch else None
            if x is None:
                if B is None:
                    continue
                z = torch.zeros(B, self.d_model, device=presence.device if presence is not None else "cpu")
                per_mod_feats.append(z)
                continue

            if not isinstance(x, torch.Tensor) or x.dim() != 2:
                raise TypeError(f"Modality '{m}' must be [B, D], got {type(x)} with dim {getattr(x, 'dim', lambda: None)()}")

            B = x.shape[0] if B is None else B
            if x.shape[0] != B:
                raise ValueError(f"Batch mismatch for '{m}': expected {B}, got {x.shape[0]}")

            if m not in self.adapters:
                raise KeyError(f"No adapter configured for modality '{m}'")
            adapter = self.adapters[m]

            z = adapter(x)
            if presence is not None:
                pm = presence[:, j].unsqueeze(1)
                z = z * pm
            per_mod_feats.append(z)

        if len(per_mod_feats) == 0:
            raise ValueError("No modalities found in batch")

        feat = torch.cat(per_mod_feats, dim=1)

        if self.use_presence_mask:
            if presence is None:
                device = feat.device
                presence = torch.tensor(
                    [1.0 if batch[m] is not None else 0.0 for m in self.modalities],
                    dtype=feat.dtype,
                    device=device
                ).unsqueeze(0).expand(B, len(self.modalities))
            feat = torch.cat([feat, presence], dim=1)

        if feat.shape[1] < self.max_concat_dim:
            feat = F.pad(feat, (0, self.max_concat_dim - feat.shape[1]))
        elif feat.shape[1] > self.max_concat_dim:
            feat = feat[:, :self.max_concat_dim]

        return self.feature_proj(feat) if self.use_rf else self.mlp(feat)