import torch 
import torch.nn as nn


class CALinear(nn.Module):
    """CALinear with multi-feature support"""
    def __init__(self, seq_len, pred_len, n_features=3):
        super(CALinear, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        self.linear_main = nn.Linear(seq_len, pred_len)
        self.linear_std = nn.Linear(seq_len, pred_len)
        self.linear_zscore = nn.Linear(seq_len, pred_len)
        self.fusion = nn.Linear(3 * pred_len, pred_len)

    def forward(self, x):
        if x.shape[-1] == 1:
            x_main = x[:, :, 0]
            x_std = torch.zeros_like(x_main)
            x_zscore = torch.zeros_like(x_main)
        elif x.shape[-1] >= 3:
            x_main = x[:, :, 0]
            x_std = x[:, :, 1]
            x_zscore = x[:, :, 2]
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        last_val = x_main[:, -1:]
        x_main_norm = x_main - last_val

        pred_main = self.linear_main(x_main_norm)
        pred_std = self.linear_std(x_std)
        pred_zscore = self.linear_zscore(x_zscore)

        concat = torch.cat([pred_main, pred_std, pred_zscore], dim=1)
        pred_fused = self.fusion(concat)
        pred = pred_fused + last_val

        return pred.unsqueeze(-1)
