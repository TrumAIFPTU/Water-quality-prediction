import torch 
import torch.nn as nn



class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        
        # Cấu hình Moving Average
        self.kernel_size = 25
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def decompose(self, x):
        # Padding replicate (giống logic bài mẫu)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Calculate Trend & Seasonal
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal

    def forward(self, x):
        trend, seasonal = self.decompose(x)
        
        # Linear Projection cho từng phần
        trend_pred = self.linear_trend(trend.squeeze(-1))
        seasonal_pred = self.linear_seasonal(seasonal.squeeze(-1))
        
        # Cộng gộp
        return (trend_pred + seasonal_pred).unsqueeze(-1)
