import torch 
import torch.nn as nn
import pandas as pd
from PyEMD import CEEMDAN
import numpy as np


class EventWeightedMSE(nn.Module):
    """Event-Weighted Loss (used for ALL models in fair comparison)"""
    def __init__(self, alpha=3.0):
        super(EventWeightedMSE, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, is_event):
        loss = self.mse(pred, target)
        weights = torch.ones_like(loss)
        weights[is_event == 1] = self.alpha
        return (loss * weights).mean()
    
def apply_ceemd_decomposition(df, target_col="OT_log", n_imfs=3):
    """CEEMD decomposition"""
    print(f"\n🌊 [2/8] Applying CEEMD...")

    signal_full = df[target_col].values
    n_total = len(signal_full)
    MAX_SAMPLES = 2000

    if n_total > MAX_SAMPLES:
        print(f"   📊 Using last {MAX_SAMPLES}/{n_total} samples")
        signal = signal_full[-MAX_SAMPLES:]
        start_idx = n_total - MAX_SAMPLES
    else:
        signal = signal_full
        start_idx = 0

    try:
        ceemdan = CEEMDAN(trials=25, noise_scale=0.1, parallel=False)
        ceemdan.ceemdan(signal, max_imf=n_imfs)
        imfs, residue = ceemdan.get_imfs_and_residue()

        n_imfs_actual = min(n_imfs, len(imfs))

        for i in range(n_imfs_actual):
            if start_idx > 0:
                padded = np.concatenate([np.zeros(start_idx), imfs[i]])
            else:
                padded = imfs[i]
            df[f"IMF_{i}"] = padded

        if start_idx > 0:
            df["residue"] = np.concatenate([np.zeros(start_idx), residue])
        else:
            df["residue"] = residue

        print(f"   ✅ CEEMDAN success: {n_imfs_actual} IMFs")

    except Exception as e:
        print(f"   ⚠️ CEEMDAN failed, using fallback")
        for i in range(min(n_imfs, 3)):
            window = [10, 50, 200][i]
            ma = pd.Series(signal_full).rolling(window, center=True).mean()
            ma = ma.fillna(method='bfill').fillna(method='ffill')
            df[f"IMF_{i}"] = signal_full - ma.values

        residue_ma = pd.Series(signal_full).rolling(200, center=True).mean()
        df["residue"] = residue_ma.fillna(method='bfill').fillna(method='ffill').values
        n_imfs_actual = min(n_imfs, 3)

    return df, n_imfs_actual