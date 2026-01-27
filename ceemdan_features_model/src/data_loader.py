import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset


class FeatureDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, flag='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        total_len = len(data)
        train_end = int(total_len * 0.6)
        val_end = int(total_len * 0.8)
        
        if flag == 'train':
            self.data = data[:train_end]
        elif flag == 'val':
            self.data = data[train_end - seq_len : val_end]
        else:
            self.data = data[val_end - seq_len:]
            
    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        
        seq_x = self.data[index:s_end] 
        seq_y = self.data[s_end:r_end, 0:1] 
        
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_target(self, pred, scaler):
        dummy = np.zeros((len(pred), scaler.n_features_in_))
        dummy[:, 0] = pred
        inv = scaler.inverse_transform(dummy)
        return inv[:, 0]