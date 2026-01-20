import pandas as pd
import numpy as np
from src.Computing.compute import detect_events_from_threshold


def preprocess_dataframe(df, col_name="OT", event_threshold=None, compute_rolling=False):
    """Data preprocessing with proper NaN handling"""
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if col_name != "OT" and col_name in df.columns:
        df = df.rename(columns={col_name: "OT"})

    df["OT"] = pd.to_numeric(df["OT"], errors='coerce')

    if df["OT"].isna().sum() > 0:
        df["OT"] = df["OT"].fillna(method='ffill').fillna(method='bfill')
        if df["OT"].isna().sum() > 0:
            mean_val = df["OT"].mean()
            if pd.isna(mean_val): mean_val = 1.0
            df["OT"] = df["OT"].fillna(mean_val)

    df["OT"] = df["OT"].clip(lower=0.1)
    df["OT_log"] = np.log(df["OT"] + 1e-6)

    df["delta_x"] = df["OT_log"].diff().fillna(0)
    df["abs_delta"] = df["delta_x"].abs()

    if event_threshold is not None:
        df = detect_events_from_threshold(df, event_threshold)
    else:
        df["is_event"] = 0.0

    if compute_rolling:
        df["rolling_std"] = df["OT_log"].rolling(12).std().fillna(0)
        df["rolling_zscore"] = ((df["OT_log"] - df["OT_log"].rolling(24).mean()) / 
                                (df["OT_log"].rolling(24).std() + 1e-6)).fillna(0)
    else:
        df["rolling_std"] = 0.0
        df["rolling_zscore"] = 0.0

    df["relative_change"] = df["OT"].pct_change().fillna(0)
    df["ma20"] = df["relative_change"].rolling(20).mean().fillna(0)

    return df