import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        return self.avg(x)

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Phân rã xu hướng
        self.decompsition = SeriesDecomp(25)

        self.Linear_Seasonal = nn.Linear(self.seq_len * in_channels, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len * in_channels, self.pred_len)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.reshape(seasonal_init.shape[0], -1)
        trend_init = trend_init.reshape(trend_init.shape[0], -1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.unsqueeze(-1) 

class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_channels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.Linear = nn.Linear(self.seq_len * in_channels, self.pred_len)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        x = x.reshape(x.shape[0], -1)
        x = self.Linear(x) 

        x = x + seq_last[:,:,0]
        
        return x.unsqueeze(-1) 