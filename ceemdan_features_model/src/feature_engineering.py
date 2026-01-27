import pandas as pd
import numpy as np

def create_change_aware_feature(df,target_col,window_size=24, percentile = 0.95):
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
    
    print(f">>> Đã tạo Features. Ngưỡng thay đổi đột ngột (95%): {threshold:.4f}")
    
    return df_feat[['Target', 'Delta_X', 'Rolling_Std', 'Rolling_Zscore', 'Event_Flag']]