import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.Utils.path import DATA_DIR
from src.Data.data_preprocessing import preprocess_dataframe

def load_data_source_separate():
    """Load from separate files"""
    print("\n📂 [1/8] Loading separate files...")
    try:
        df_ec = pd.read_csv(DATA_DIR/"G_WTP-main/G_WTP-main/EC_origin.csv")
        df_ph = pd.read_csv(DATA_DIR/"G_WTP-main/G_WTP-main/pH_origin.csv")
        return preprocess_dataframe(df_ec, "OT"), preprocess_dataframe(df_ph, "OT")
    except FileNotFoundError:
        print("⚠️ Error: Missing EC_origin.csv or pH_origin.csv")
        return None, None

def load_data_source_api():
    """Load from API file"""
    print("\n📂 [1/8] Loading API file...")
    try:
        df = pd.read_csv(DATA_DIR/"water_data_api/water_data_api.csv")
        col_date = "DateTime" if "DateTime" in df.columns else "date"
        df["date"] = pd.to_datetime(df[col_date])
        df_ec = df[["date", "EC"]].copy()
        df_ph = df[["date", "pH"]].copy()
        return preprocess_dataframe(df_ec, "EC"), preprocess_dataframe(df_ph, "pH")
    except FileNotFoundError:
        print("⚠️ Error: Missing water_data_api.csv")
        return None, None

class WaterQualityDataset(Dataset):
    """Normalization computed externally from train data only"""
    def __init__(self, data, seq_len, pred_len, use_features=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_features = use_features

        self.series = torch.FloatTensor(data["OT_log"].values).unsqueeze(-1)
        self.events = torch.FloatTensor(data["is_event"].values).unsqueeze(-1)

        if use_features:
            self.std = torch.FloatTensor(data["rolling_std"].values).unsqueeze(-1)
            self.zscore = torch.FloatTensor(data["rolling_zscore"].values).unsqueeze(-1)

        self.mean = None
        self.std_val = None
        self.series_norm = None

    def set_normalization(self, mean, std):
        """Set normalization parameters from external source"""
        self.mean = mean
        self.std_val = std

        if torch.abs(self.std_val) < 1e-6:
            self.std_val = torch.tensor(1.0)

        self.series_norm = (self.series - self.mean) / self.std_val

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_main = self.series_norm[idx : idx + self.seq_len]
        y = self.series_norm[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        y_event = self.events[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        if self.use_features:
            x_std = self.std[idx : idx + self.seq_len]
            x_zscore = self.zscore[idx : idx + self.seq_len]
            x = torch.cat([x_main, x_std, x_zscore], dim=-1)
        else:
            x = x_main

        return x, y, y_event

    def denormalize(self, x):
        return x * self.std_val + self.mean

def create_dataloaders(df, seq_len, pred_len, batch_size=32, use_features=False):
    """Compute normalization ONLY from train split"""
    dataset = WaterQualityDataset(df, seq_len, pred_len, use_features=use_features)

    total_len = len(dataset)
    train_size = int(0.6 * total_len)
    val_size = int(0.2 * total_len)

    # Compute mean/std from train indices only
    train_indices = range(0, train_size)
    train_data = dataset.series[train_indices]
    mean = torch.mean(train_data)
    std = torch.std(train_data)

    if torch.abs(std) < 1e-6:
        std = torch.tensor(1.0)

    dataset.set_normalization(mean, std)

    train_set = torch.utils.data.Subset(dataset, range(0, train_size))
    val_set = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_set = torch.utils.data.Subset(dataset, range(train_size + val_size, total_len))

    loaders = (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )

    split_info = {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': total_len - train_size - val_size
    }

    return loaders, dataset, split_info