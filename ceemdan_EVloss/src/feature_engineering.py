import pandas as pd
import numpy as np

def create_change_aware_features(df, target_col, window_size=24, percentile=0.95):
    df_feat = df[[target_col]].copy()
    df_feat.rename(columns={target_col: 'Target'}, inplace=True)

    df_feat['Delta_X'] = df_feat['Target'].diff()
    df_feat['Abs_Delta_X'] = df_feat['Delta_X'].abs()

    df_feat['Rolling_Std'] = df_feat['Target'].rolling(window=window_size).std()
    rolling_mean = df_feat['Target'].rolling(window=window_size).mean()
    
    df_feat['Rolling_Zscore'] = (df_feat['Target'] - rolling_mean) / (df_feat['Rolling_Std'] + 1e-6)

    threshold = df_feat['Abs_Delta_X'].quantile(percentile)
    df_feat['Event_Flag'] = (df_feat['Abs_Delta_X'] > threshold).astype(float)
    
    df_feat.dropna(inplace=True)
    cols = [c for c in df_feat.columns if c != 'Event_Flag'] + ['Event_Flag']

    return df_feat[cols]