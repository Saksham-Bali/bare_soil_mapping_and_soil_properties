# run_spectral_mae_v5.py
# ============================================================
# End-to-end pretraining for Spectral MAE (CLS) on REAL data.
# MAE-compliant: encoder mask token (no leakage), TransformerDecoder,
# λ/FWHM meta, loss selectable (MSE/MAE), early-stop, CSV log, PDF report,
# checkpoints, encoder export, scaler save. 
# ============================================================

import os, csv, time, joblib, torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from spectral_mae_v5 import (
    SpectralMAE, make_loader, masked_mse_loss, masked_mae_loss,
    encode_cls_no_mask, plot_history_page, tsne_cls_page, append_text_page, HAS_REPORT
)

# --------------------
# USER CONFIG
# --------------------
SEED              = 42
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# Data / IO
DATA_PATH    = 'training_data_balanced.parquet'
BAND_COLUMNS = ['B11','B12','B2','B3','B4','B5','B6','B7','B8','B8A']
SCALER_PATH  = "spectral_scaler.joblib"
CSV_LOG      = "spectral_mae_history.csv"
CKPT_PATH    = "spectral_mae_ckpt.pt"
ENCODER_EXPORT_PATH = "spectral_mae_encoder.pt"
REPORT_PDF   = "spectral_mae_report.pdf"
TSNE_POINTS  = 2000

# Sentinel-2 λ/FWHM (nm) aligned to BAND_COLUMNS:
LAMBDAS_NM = [1610, 2190, 490, 560, 665, 705, 740, 783, 842, 865]
FWHM_NM    = [  90,  180,  65,  35,  30,  15,  15,  20, 115,  20]
LAM_MIN_NM = 400.0
LAM_MAX_NM = 2500.0
FOURIER_FEATURES = 16

# Model
NUM_BANDS   = len(BAND_COLUMNS)
EMBED_DIM   = 32
ENC_DEPTH   = 4
ENC_HEADS   = 4
DEC_DIM     = 32
DEC_DEPTH   = 2
DEC_HEADS   = 4
INCLUDE_CLS_IN_MEMORY = True

# SSL objective / masking
MASK_RATIO  = 0.6              # per-pixel band masking
LOSS_TYPE   = "mae"            # "mse" or "mae"

# Optimization
BATCH_SIZE  = 1024
EPOCHS      = 15
LR          = 1e-3
WEIGHT_DECAY= 0.05
AMP         = True

# Early stopping
VAL_FRACTION= 0.10
EARLY_PATIENCE = 10
EARLY_DELTA    = 1e-6

# Dataloader workers
NUM_WORKERS = 4
PIN_MEMORY  = True

# --------------------
# Reproducibility
# --------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------
# IO helpers
# --------------------
def load_parquet_bands(path, band_columns):
    print(f"[IO] Reading Parquet: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in band_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing band columns: {missing}")
    X = (df[band_columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(np.float32)
            .values)
    print(f"[IO] Loaded X shape: {X.shape}")
    return X

def write_csv_header(csv_path):
    new = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["epoch","train_loss","val_loss","time_s"])
    return f, w

# --------------------
# Main
# --------------------
def main():
    set_seed(SEED)
    torch.set_float32_matmul_precision("high")

    # 1) Load data
    X = load_parquet_bands(DATA_PATH, BAND_COLUMNS)

    # 2) Split & scale
    idx = np.arange(X.shape[0])
    tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRACTION, random_state=SEED)
    Xtr_raw, Xva_raw = X[tr_idx], X[va_idx]

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    print("[Scale] Fitting RobustScaler on train split...")
    Xtr = scaler.fit_transform(Xtr_raw).astype(np.float32)
    Xva = scaler.transform(Xva_raw).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Scale] Saved scaler -> {SCALER_PATH}")
    print(f"[Data] Train: {Xtr.shape}, Val: {Xva.shape}")

    # 3) DataLoaders
    tr_tensor = torch.from_numpy(Xtr)
    va_tensor = torch.from_numpy(Xva)
    tr_loader = make_loader(tr_tensor, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, shuffle=True)
    va_loader = make_loader(va_tensor, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, shuffle=False)

    # 4) Model/opt
    model = SpectralMAE(
        num_bands=NUM_BANDS,
        embed_dim=EMBED_DIM, enc_depth=ENC_DEPTH, enc_heads=ENC_HEADS,
        dec_dim=DEC_DIM, dec_depth=DEC_DEPTH, dec_heads=DEC_HEADS,
        lambda_nm=LAMBDAS_NM, fwhm_nm=FWHM_NM,
        fourier_features=FOURIER_FEATURES,
        lam_min_nm=LAM_MIN_NM, lam_max_nm=LAM_MAX_NM,
        include_cls_in_memory=INCLUDE_CLS_IN_MEMORY
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=AMP)

    # 5) Train with early stopping; CSV + checkpointing
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    history = []

    csv_f, csv_w = write_csv_header(CSV_LOG)
    outer = tqdm(range(1, EPOCHS+1), desc="Epochs", position=0)

    for epoch in outer:
        t0 = time.time()
        # ----- train -----
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for batch in tr_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            device_type = "cuda" if DEVICE.startswith("cuda") else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=AMP):
                out = model(x, mask_ratio=MASK_RATIO)
                if LOSS_TYPE.lower() == "mae":
                    loss = masked_mae_loss(out["pred_masked"], x, out["mask"])
                else:
                    loss = masked_mse_loss(out["pred_masked"], x, out["mask"])

            if AMP:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            train_loss_sum += float(loss.detach().item())
            n_batches += 1

        train_loss = train_loss_sum / max(1, n_batches)

        # ----- val -----
        model.eval()
        val_loss_sum, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(DEVICE, non_blocking=True)
                device_type = "cuda" if DEVICE.startswith("cuda") else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=AMP):
                    out = model(x, mask_ratio=MASK_RATIO)
                    if LOSS_TYPE.lower() == "mae":
                        vloss = masked_mae_loss(out["pred_masked"], x, out["mask"])
                    else:
                        vloss = masked_mse_loss(out["pred_masked"], x, out["mask"])
                val_loss_sum += float(vloss.detach().item())
                n_batches += 1
        val_loss = val_loss_sum / max(1, n_batches)

        # ----- bookkeeping -----
        dur = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "time_s": dur})
        csv_w.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{dur:.2f}"])
        csv_f.flush()
        outer.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}")

        # early stopping + checkpoint
        improved = (best_val - val_loss) > EARLY_DELTA
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            patience += 1
            if patience >= EARLY_PATIENCE:
                print(f"[EarlyStop] No improvement for {EARLY_PATIENCE} epochs.")
                break

    csv_f.close()

    # 6) Reload best, export encoder only
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    torch.save(model.encoder.state_dict(), ENCODER_EXPORT_PATH)

    # 7) PDF report (curves + CLS t-SNE)
    if REPORT_PDF and HAS_REPORT:
        with PdfPages(REPORT_PDF) as pdf:
            # Data & SOP page
            append_text_page(pdf, [
                "Spectral MAE — REAL DATA (λ-aware, MAE-compliant)",
                f"Data path: {DATA_PATH}",
                f"Bands: {BAND_COLUMNS}",
                f"Train size: {Xtr.shape[0]}   Val size: {Xva.shape[0]}",
                "SOP:",
                " - Per-pixel spectrum → scalar→D + λ/FWHM meta",
                " - Random band mask; replace masked with encoder mask token",
                " - Prepend CLS; TransformerEncoder over all tokens",
                " - Decoder (TransformerDecoder): queries = masked bands (dec mask token + meta)",
                "   cross-attend to visible band memory (+CLS optional)",
                f" - Optimize masked {LOSS_TYPE.upper()} only on masked bands",
            ])
            plot_history_page(pdf, history)
            # CLS t-SNE on a subset of val
            tsne_cls_page(pdf, model, va_loader, points=TSNE_POINTS, amp=AMP)
            # Summary page
            append_text_page(pdf, [
                f"Best val loss: {best_val:.6f} at epoch {best_epoch}",
                f"Checkpoint:   {CKPT_PATH}",
                f"Encoder:      {ENCODER_EXPORT_PATH}",
                f"Scaler:       {SCALER_PATH}",
                "λ centers (nm): " + str(LAMBDAS_NM),
                "FWHM (nm):     " + str(FWHM_NM),
                f"include_cls_in_memory: {INCLUDE_CLS_IN_MEMORY}",
                f"mask_ratio: {MASK_RATIO}   loss: {LOSS_TYPE}",
                f"embed/dec dims: {EMBED_DIM}/{DEC_DIM}",
                f"enc/dec depth:  {ENC_DEPTH}/{DEC_DEPTH}",
                f"heads: enc {ENC_HEADS} / dec {DEC_HEADS}",
            ])

    # 8) Console summary
    print(f"[Done] Best val loss={best_val:.6f} @ epoch {best_epoch}")
    print(f"[Saved] Checkpoint -> {CKPT_PATH}")
    print(f"[Saved] Encoder    -> {ENCODER_EXPORT_PATH}")
    print(f"[Saved] Scaler     -> {SCALER_PATH}")
    if REPORT_PDF and HAS_REPORT:
        print(f"[Saved] Report PDF -> {REPORT_PDF}")

if __name__ == "__main__":
    main()
