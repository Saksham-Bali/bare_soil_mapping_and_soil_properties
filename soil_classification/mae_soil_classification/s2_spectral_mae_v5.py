# spectral_mae_v5.py
# ======================================================================
# Pixel-wise Spectral MAE (λ-aware) — MAE-compliant masking
# - Encoder: masked bands replaced by learned token (no value leakage)
# - Decoder: nn.TransformerDecoder with cross-attn from masked-band queries
# - Wavelength-aware embeddings (λ center & FWHM) via Fourier+MLP
# - CLS: produced by encoder; optionally added to decoder memory
# - Utilities: dataset/loader, masked losses, CLS (no-mask) encoding,
#   reporting helpers (history plot, CLS t-SNE, text page)
# ======================================================================

from __future__ import annotations
from typing import Tuple, Optional, Dict, List
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Optional reporting
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.manifold import TSNE
    HAS_REPORT = True
except Exception:
    HAS_REPORT = False

# -----------------------------
# Dataset (REAL DATA ONLY)
# -----------------------------
class SpectralVectorDataset(Dataset):
    """Wraps a REAL (N,C) matrix of spectra."""
    def __init__(self, data: np.ndarray):
        assert data.ndim == 2, f"Expected (N,C), got {data.shape}"
        self.x = data.astype(np.float32, copy=False)

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return torch.from_numpy(self.x[i])  # (C,)

def make_loader(x_tensor: torch.Tensor | np.ndarray,
                batch_size: int,
                num_workers: int = 4,
                pin_memory: bool = True,
                shuffle: bool = True) -> DataLoader:
    if isinstance(x_tensor, np.ndarray):
        ds = SpectralVectorDataset(x_tensor)
    else:
        ds = torch.utils.data.TensorDataset(x_tensor)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

# -----------------------------
# Wavelength → embedding (Fourier + MLP)
# -----------------------------
class SpectralMetaEmbedding(nn.Module):
    """
    Compute D-dim embedding from continuous band metadata:
      - lambda_nm : center wavelength (nm)
      - fwhm_nm   : FWHM bandwidth (nm)
    Normalizes inputs; applies Fourier features on λ; MLP → D.
    """
    def __init__(self, embed_dim: int, fourier_features: int = 16,
                 lam_min_nm: float = 400.0, lam_max_nm: float = 2500.0):
        super().__init__()
        self.ff = fourier_features
        self.lam_min = lam_min_nm
        self.lam_max = lam_max_nm
        in_dim = 2 + 2 * fourier_features  # [lam_norm, fwhm_norm] + sin/cos(λ)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, lambda_nm: torch.Tensor, fwhm_nm: torch.Tensor) -> torch.Tensor:
        # lambda_nm, fwhm_nm: (B, C)
        lam = (lambda_nm - self.lam_min) / (self.lam_max - self.lam_min)
        fwhm = fwhm_nm / (self.lam_max - self.lam_min)
        k = torch.arange(1, self.ff + 1, device=lam.device, dtype=lam.dtype)[None, None, :]
        ang = 2 * math.pi * lam.unsqueeze(-1) * k  # (B,C,ff)
        phi = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B,C,2ff)
        feats = torch.cat([lam.unsqueeze(-1), fwhm.unsqueeze(-1), phi], dim=-1)  # (B,C,2+2ff)
        return self.mlp(feats)  # (B,C,D)

# -----------------------------
# Mask sampler (per-batch)
# -----------------------------
def sample_band_mask(B: int, C: int, mask_ratio: float, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      mask        : (B,C) bool, True=masked
      ids_keep    : (B,K) long, indices of kept (visible) bands
      ids_mask    : (B,M) long, indices of masked bands
    """
    K = int(round((1.0 - mask_ratio) * C))
    noise = torch.rand(B, C, device=device)            # (B,C)
    ids_shuffle = torch.argsort(noise, dim=1)          # low→keep
    ids_keep = ids_shuffle[:, :K]
    ids_mask = ids_shuffle[:, K:]
    mask = torch.zeros(B, C, dtype=torch.bool, device=device)
    mask.scatter_(1, ids_mask, True)
    return mask, ids_keep, ids_mask

# -----------------------------
# Encoder (with learned mask token replacement)
# -----------------------------
class SpectralMAEEncoder(nn.Module):
    """
    Per-pixel spectral encoder:
      - value_proj: scalar band value → D
      - add λ/FWHM meta embedding
      - replace masked bands with a learned encoder_mask_token (no leakage)
      - prepend CLS
      - TransformerEncoder over [CLS + (visible+masked-as-token)]
      - Return CLS, all band token encodings, and the mask/indices
    """
    def __init__(self, num_bands: int, embed_dim: int,
                 enc_depth: int, enc_heads: int,
                 lambda_nm: List[float], fwhm_nm: List[float],
                 fourier_features: int = 16,
                 lam_min_nm: float = 400.0, lam_max_nm: float = 2500.0):
        super().__init__()
        self.C = num_bands
        D = embed_dim

        lam = torch.tensor(lambda_nm, dtype=torch.float32)
        wid = torch.tensor(fwhm_nm,   dtype=torch.float32)
        self.register_buffer("lambda_nm_const", lam, persistent=False)  # (C,)
        self.register_buffer("fwhm_nm_const",   wid, persistent=False)  # (C,)

        self.value_proj = nn.Linear(1, D)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.meta = SpectralMetaEmbedding(D, fourier_features, lam_min_nm, lam_max_nm)

        # Learned encoder mask token (used to replace masked band values)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.enc_mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(d_model=D, nhead=enc_heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=enc_depth)

    def forward(self, x: torch.Tensor, mask_ratio: float) -> Dict[str, torch.Tensor]:
        """
        x: (B,C) spectrum
        Returns dict:
          cls: (B,D)
          band_enc: (B,C,D) encoder outputs per band (in original order)
          mask: (B,C) bool True=masked
          ids_keep: (B,K) long indices of visible
          ids_mask: (B,M) long indices of masked
        """
        B, C = x.shape
        device = x.device
        D = self.cls.shape[-1]
        assert C == self.C

        # scalar→D + λ/FWHM meta
        tok = self.value_proj(x.unsqueeze(-1))                # (B,C,1)->(B,C,D)
        lam = self.lambda_nm_const.unsqueeze(0).expand(B, C)
        wid = self.fwhm_nm_const .unsqueeze(0).expand(B, C)
        tok = tok + self.meta(lam, wid)                       # (B,C,D)

        # sample mask (canonical MAE: encoder sees ONLY visible bands)
        mask, ids_keep, ids_mask = sample_band_mask(B, C, mask_ratio, device)

        # gather visible tokens in ids_keep order
        tok_vis = torch.gather(tok, dim=1,
                            index=ids_keep.unsqueeze(-1).expand(-1, -1, D))  # (B,K,D)

        # prepend CLS and encode [CLS + visible]
        cls = self.cls.expand(B, 1, D)
        enc_in = torch.cat([cls, tok_vis], dim=1)  # (B,1+K,D)
        enc_out = self.encoder(enc_in)             # (B,1+K,D)

        # split CLS and visible-band encodings
        cls_out = enc_out[:, 0, :]                 # (B,D)
        band_vis_out = enc_out[:, 1:, :]           # (B,K,D)

        # scatter visible encodings back to (B,C,D), zeros for masked
        band_out = torch.zeros(B, C, D, device=device, dtype=band_vis_out.dtype)
        band_out.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D), band_vis_out)

        return {
            "cls": cls_out,              # (B,D)
            "band_enc": band_out,        # (B,C,D) in original band order (masked rows are 0)
            "mask": mask,                # (B,C) bool
            "ids_keep": ids_keep,        # (B,K)
            "ids_mask": ids_mask         # (B,M)
        }


# -----------------------------
# Decoder (TransformerDecoder with cross-attention)
# -----------------------------
class SpectralMAEDecoder(nn.Module):
    """
    TransformerDecoder for masked band reconstruction.
    Queries: masked band positions only (each starts from learned dec mask token + λ meta).
    Memory: encoder outputs for visible bands (optionally + CLS for global context).
    Output: scalar value per masked band; we reassemble to (B,C) with preds where masked,
            and (optionally) passthrough/aux for visible.
    """
    def __init__(self, num_bands: int, embed_dim: int, dec_dim: int,
                 dec_depth: int, dec_heads: int,
                 lambda_nm: List[float], fwhm_nm: List[float],
                 include_cls_in_memory: bool = True,
                 fourier_features: int = 16,
                 lam_min_nm: float = 400.0, lam_max_nm: float = 2500.0):
        super().__init__()
        self.C = num_bands
        self.include_cls = include_cls_in_memory

        # Wavelength constants
        lam = torch.tensor(lambda_nm, dtype=torch.float32)
        wid = torch.tensor(fwhm_nm,   dtype=torch.float32)
        self.register_buffer("lambda_nm_const", lam, persistent=False)  # (C,)
        self.register_buffer("fwhm_nm_const",   wid, persistent=False)  # (C,)

        # Projections & tokens
        self.mem_proj = nn.Linear(embed_dim, dec_dim)         # encoder → decoder dim
        self.dec_mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.dec_mask_token, std=0.02)

        # Meta for queries (masked positions) and for memory (optional)
        self.meta_q = SpectralMetaEmbedding(dec_dim, fourier_features, lam_min_nm, lam_max_nm)
        self.meta_m = SpectralMetaEmbedding(dec_dim, fourier_features, lam_min_nm, lam_max_nm)

        # TransformerDecoder
        dec_layer = nn.TransformerDecoderLayer(d_model=dec_dim, nhead=dec_heads, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_depth)

        # Head to scalar
        self.head = nn.Linear(dec_dim, 1)

    def forward(self,
                enc_band: torch.Tensor,    # (B,C,D_enc) encoder band outputs in original order
                enc_cls: torch.Tensor,     # (B,D_enc)
                mask: torch.Tensor,         # (B,C) True=masked
                ids_keep: torch.Tensor,     # (B,K) indices of visible
                ids_mask: torch.Tensor) -> torch.Tensor:  # (B,M) indices of masked
        B, C, D_enc = enc_band.shape
        device = enc_band.device

        # ----- Build memory (visible bands [+ CLS]) -----
        # gather visible band encodings
        mem_vis = torch.gather(enc_band, dim=1,
                               index=ids_keep.unsqueeze(-1).expand(-1, -1, D_enc))  # (B,K,D_enc)
        mem = mem_vis
        if self.include_cls:
            mem = torch.cat([enc_cls.unsqueeze(1), mem_vis], dim=1)  # (B,1+K,D_enc)

        # add λ/FWHM meta to memory (per-band only; CLS gets zero meta)
        dec_dim = self.head.in_features
        mem = self.mem_proj(mem)  # (B,1+K,dec_dim)
        if self.include_cls:
            # meta for visible bands only
            K = ids_keep.shape[1]
            lam = self.lambda_nm_const.unsqueeze(0).expand(B, C)  # (B,C)
            wid = self.fwhm_nm_const .unsqueeze(0).expand(B, C)
            lam_vis = torch.gather(lam, 1, ids_keep)              # (B,K)
            wid_vis = torch.gather(wid, 1, ids_keep)              # (B,K)
            meta_vis = self.meta_m(lam_vis, wid_vis)              # (B,K,dec_dim)
            meta = torch.cat([torch.zeros(B, 1, dec_dim, device=device), meta_vis], dim=1)
            mem = mem + meta
        else:
            lam = self.lambda_nm_const.unsqueeze(0).expand(B, C)
            wid = self.fwhm_nm_const .unsqueeze(0).expand(B, C)
            lam_vis = torch.gather(lam, 1, ids_keep)
            wid_vis = torch.gather(wid, 1, ids_keep)
            mem = mem + self.meta_m(lam_vis, wid_vis)

        # ----- Build queries (masked positions only) -----
        M = ids_mask.shape[1]
        q = self.dec_mask_token.expand(B, M, dec_dim)            # (B,M,dec_dim)
        lam = self.lambda_nm_const.unsqueeze(0).expand(B, C)
        wid = self.fwhm_nm_const .unsqueeze(0).expand(B, C)
        lam_m = torch.gather(lam, 1, ids_mask)                   # (B,M)
        wid_m = torch.gather(wid, 1, ids_mask)                   # (B,M)
        q = q + self.meta_q(lam_m, wid_m)                        # add meta to queries

        # ----- Decode -----
        dec_out = self.decoder(tgt=q, memory=mem)                # (B,M,dec_dim)
        pred_masked = self.head(dec_out).squeeze(-1)             # (B,M) scalar per masked band

        # ----- Reassemble to (B,C): preds only where masked -----
        out = torch.zeros(B, C, device=device, dtype=pred_masked.dtype)
        out.scatter_(1, ids_mask, pred_masked)
        return out  # preds for masked bands at masked positions; zeros elsewhere

# -----------------------------
# Model wrapper
# -----------------------------
class SpectralMAE(nn.Module):
    def __init__(self,
                 num_bands: int,
                 embed_dim: int, enc_depth: int, enc_heads: int,
                 dec_dim: int, dec_depth: int, dec_heads: int,
                 lambda_nm: List[float], fwhm_nm: List[float],
                 fourier_features: int = 16,
                 lam_min_nm: float = 400.0, lam_max_nm: float = 2500.0,
                 include_cls_in_memory: bool = True):
        super().__init__()
        self.encoder = SpectralMAEEncoder(
            num_bands, embed_dim, enc_depth, enc_heads,
            lambda_nm, fwhm_nm, fourier_features, lam_min_nm, lam_max_nm
        )
        self.decoder = SpectralMAEDecoder(
            num_bands, embed_dim, dec_dim, dec_depth, dec_heads,
            lambda_nm, fwhm_nm, include_cls_in_memory,
            fourier_features, lam_min_nm, lam_max_nm
        )

    def forward(self, x: torch.Tensor, mask_ratio: float) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x, mask_ratio=mask_ratio)
        # decoder predicts only masked positions; zeros elsewhere
        pred_masked_scatter = self.decoder(
            enc["band_enc"], enc["cls"], enc["mask"], enc["ids_keep"], enc["ids_mask"]
        )
        return {"cls": enc["cls"], "pred_masked": pred_masked_scatter, **enc}

# -----------------------------
# Masked losses
# -----------------------------
def masked_mse_loss(pred_masked_scatter: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    # Only masked positions contribute
    diff = (pred_masked_scatter - target)**2
    diff = diff[mask]
    return diff.mean() if diff.numel() > 0 else torch.tensor(0.0, device=target.device)

def masked_mae_loss(pred_masked_scatter: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    diff = (pred_masked_scatter - target).abs()
    diff = diff[mask]
    return diff.mean() if diff.numel() > 0 else torch.tensor(0.0, device=target.device)

# -----------------------------
# Deterministic CLS extractor (no-mask)
# -----------------------------
@torch.no_grad()
def encode_cls_no_mask(model: SpectralMAE, X: torch.Tensor,
                       batch: int = 4096, amp: bool = True) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i+batch].to(device, non_blocking=True)
        # use a tiny mask_ratio (0) to keep all tokens; replace path still applies but no mask
        with torch.amp.autocast(device_type="cuda" if device.type=="cuda" else "cpu", enabled=amp):
            enc = model.encoder(xb, mask_ratio=0.0)
        outs.append(enc["cls"].detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

# -----------------------------
# Report helpers
# -----------------------------
def plot_history_page(pdf: PdfPages, history: List[Dict[str, float]]):
    fig, ax = plt.subplots(figsize=(6, 4))
    tr = [h["train_loss"] for h in history]
    va = [h["val_loss"] for h in history]
    ax.plot(tr, label="train")
    ax.plot(va, label="val")
    ax.set_title("Masked Reconstruction Loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    pdf.savefig(fig); plt.close(fig)

@torch.no_grad()
def tsne_cls_page(pdf: PdfPages, model: SpectralMAE, loader: DataLoader,
                  points: int = 2000, amp: bool = True):
    device = next(model.parameters()).device
    model.eval()
    cls_all = []
    seen = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        with torch.amp.autocast(device_type="cuda" if device.type=="cuda" else "cpu", enabled=amp):
            enc = model.encoder(x, mask_ratio=0.0)
        cls_all.append(enc["cls"].detach().cpu().numpy())
        seen += enc["cls"].shape[0]
        if seen >= points: break
    Z = np.concatenate(cls_all, axis=0)[:points]
    Z2 = TSNE(n_components=2, perplexity=35, init="random", learning_rate="auto").fit_transform(Z)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Z2[:, 0], Z2[:, 1], s=6, alpha=0.8)
    ax.set_title("CLS t-SNE (subset)")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    pdf.savefig(fig); plt.close(fig)

def append_text_page(pdf: PdfPages, lines: List[str]):
    fig = plt.figure(figsize=(6, 4))
    plt.axis("off")
    text = "\n".join(lines)
    plt.text(0.02, 0.98, text, va="top", ha="left", fontsize=10)
    pdf.savefig(fig); plt.close(fig)





