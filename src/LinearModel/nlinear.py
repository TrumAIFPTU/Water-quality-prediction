import torch
import torch.nn as nn

class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, 1]
        x = x.squeeze(-1)           # [Batch, Seq_Len]
        last_value = x[:, -1:]      # Lấy giá trị cuối
        x_norm = x - last_value     # Normalize
        pred_norm = self.linear(x_norm) # Linear Projection
        pred = pred_norm + last_value # Denormalize
        return pred.unsqueeze(-1)
