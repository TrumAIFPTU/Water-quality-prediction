import numpy as np


def compute_event_threshold_from_train(df_train, percentile=95):
    """Compute event threshold from training data only"""
    threshold = np.percentile(df_train["abs_delta"], percentile)
    return threshold

def detect_events_from_threshold(df, threshold):
    """Event detection using raw log-diff from training distribution"""
    df["is_event"] = (df["abs_delta"] > threshold).astype(float)
    return df

def compute_rolling_features_post_split(df, train_end_idx):
    """Compute rolling features without leakage"""
    df_copy = df.copy()
    signal = df_copy["OT_log"].values

    rolling_std = np.zeros(len(signal))
    rolling_zscore = np.zeros(len(signal))

    for i in range(len(signal)):
        if i < 12:
            rolling_std[i] = 0
        else:
            if i <= train_end_idx:
                window = signal[max(0, i-12):i]
            else:
                window = signal[max(0, i-12):min(i, train_end_idx+1)]
            rolling_std[i] = np.std(window) if len(window) > 0 else 0

        if i < 24:
            rolling_zscore[i] = 0
        else:
            if i <= train_end_idx:
                window = signal[max(0, i-24):i]
            else:
                window = signal[max(0, i-24):min(i, train_end_idx+1)]

            if len(window) > 0:
                mean = np.mean(window)
                std = np.std(window) + 1e-6
                rolling_zscore[i] = (signal[i] - mean) / std
            else:
                rolling_zscore[i] = 0

    df_copy["rolling_std"] = rolling_std
    df_copy["rolling_zscore"] = rolling_zscore

    return df_copy
