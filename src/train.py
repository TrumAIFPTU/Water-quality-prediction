import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

# --- CẤU HÌNH CHUNG ---
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# =================================================================
# 1. DATA LOADING & PREPROCESSING
# =================================================================
def preprocess_dataframe(df, col_name="OT"):
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    if col_name != "OT" and col_name in df.columns:
        df = df.rename(columns={col_name: "OT"})
        
    # Xử lý NaN và data bẩn
    df["OT"] = pd.to_numeric(df["OT"], errors='coerce')
    df["OT"] = df["OT"].fillna(method='ffill').fillna(method='bfill').fillna(0)
    df["OT"] = df["OT"].clip(lower=0)

    # Log-transform & Feature Engineering
    df["OT_log"] = np.log(df["OT"] + 1e-6)
    df["delta_x"] = df["OT_log"].diff().fillna(0)
    df["abs_delta"] = df["delta_x"].abs()
    threshold = np.percentile(df["abs_delta"], 95)
    df["is_event"] = (df["abs_delta"] > threshold).astype(float)
    
    # Features cho visualize
    df["relative_change"] = df["OT"].pct_change().fillna(0)
    df["ma20"] = df["relative_change"].rolling(20).mean().fillna(0)
    
    return df

def load_data_source_separate():
    print("\n[1/4] Loading separate files...")
    try:
        df_ec = pd.read_csv("datasets/G_WTP-main/G_WTP-main/EC_origin.csv")
        df_ph = pd.read_csv("datasets/G_WTP-main/G_WTP-main/pH_origin.csv")
        return preprocess_dataframe(df_ec, "OT"), preprocess_dataframe(df_ph, "OT")
    except FileNotFoundError:
        print("Error: Thiếu file EC_origin.csv hoặc pH_origin.csv")
        exit()

def load_data_source_api():
    print("\n[2/4] Loading API file...")
    try:
        df = pd.read_csv("datasets/water_data_api/water_data_api.csv")
        col_date = "DateTime" if "DateTime" in df.columns else "date"
        df["date"] = pd.to_datetime(df[col_date])
        df_ec = df[["date", "EC"]].copy()
        df_ph = df[["date", "pH"]].copy()
        return preprocess_dataframe(df_ec, "EC"), preprocess_dataframe(df_ph, "pH")
    except FileNotFoundError:
        print("Error: Thiếu file water_data_api.csv")
        exit()

# =================================================================
# 2. VISUALIZATION (PHẦN BẠN YÊU CẦU THÊM)
# =================================================================
def plot_comprehensive_analysis(df_ec, df_ph):
    print("\n[3/4] Đang vẽ và lưu biểu đồ phân tích...")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), gridspec_kw={"wspace": 0.3, "hspace": 0.7})
    fig.suptitle("Comprehensive Water Quality Analysis: EC vs pH Trends", fontsize=20, fontweight="bold", y=0.97)

    locator = mdates.MonthLocator(interval=3)
    formatter = mdates.DateFormatter("%b %Y")

    def format_subplot(ax, title, ylabel):
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
        ax.set_xlabel("Date", fontsize=11, labelpad=2) 
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', pad=2) 
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3)

    # --- Hàng 1: Raw Data ---
    axes[0, 0].plot(df_ec["date"], df_ec["OT"], color="royalblue", linewidth=1)
    format_subplot(axes[0, 0], "EC Over Time", "EC (µS/cm)")
    
    axes[0, 1].plot(df_ph["date"], df_ph["OT"], color="crimson", linewidth=1)
    format_subplot(axes[0, 1], "pH Over Time", "pH Level")

    # --- Hàng 2: Log Transformed + Events ---
    axes[1, 0].plot(df_ec["date"], df_ec["OT_log"], color="seagreen", label="Log(EC)", linewidth=1)
    events_ec = df_ec[df_ec["is_event"] == 1]
    axes[1, 0].scatter(events_ec["date"], events_ec["OT_log"], color="red", s=15, alpha=0.7, label="Sudden Fluctuation", zorder=5)
    format_subplot(axes[1, 0], "Log-transformed EC (with Events)", "Log(EC)")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].plot(df_ph["date"], df_ph["OT_log"], color="darkorange", linewidth=1)
    format_subplot(axes[1, 1], "Log-transformed pH", "Log(pH)")

    # --- Hàng 3: Relative Change ---
    axes[2, 0].plot(df_ec["date"], df_ec["relative_change"], color="orchid", alpha=0.5, linewidth=0.8)
    axes[2, 0].plot(df_ec["date"], df_ec["ma20"], color="gold", linestyle="--", linewidth=1.5)
    format_subplot(axes[2, 0], "EC Relative Change", "Rel Change")
    axes[2, 0].yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[2, 1].plot(df_ph["date"], df_ph["relative_change"], color="teal", alpha=0.5, linewidth=0.8)
    axes[2, 1].plot(df_ph["date"], df_ph["ma20"], color="darkblue", linestyle="--", linewidth=1.5)
    format_subplot(axes[2, 1], "pH Relative Change", "Rel Change")
    axes[2, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Lưu ảnh để không chặn chương trình
    plt.savefig("analysis_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(">> Đã lưu biểu đồ vào file 'analysis_chart.png'")

# =================================================================
# 3. MODELS (NLinear & DLinear - Cấu trúc mới)
# =================================================================

# --- NLINEAR ---
class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(NLinear, self).__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, 1]
        x = x.squeeze(-1)           # [Batch, Seq_Len]
        last_value = x[:, -1:]      # Lấy giá trị cuối
        x_norm = x - last_value     # Normalize
        pred_norm = self.linear(x_norm) # Linear Projection
        pred = pred_norm + last_value # Denormalize
        return pred.unsqueeze(-1)


# --- DLINEAR ---
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(DLinear, self).__init__()
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

# =================================================================
# 4. DATASET & TRAINING PIPELINE
# =================================================================
class WaterQualityDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_col="OT_log"):
        self.series = torch.FloatTensor(data[target_col].values).unsqueeze(-1)
        self.events = torch.FloatTensor(data["is_event"].values).unsqueeze(-1)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = torch.mean(self.series)
        self.std = torch.std(self.series)
        if self.std == 0: self.std = 1
        self.series_norm = (self.series - self.mean) / self.std

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.series_norm[idx : idx + self.seq_len]
        y = self.series_norm[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        y_event = self.events[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y, y_event
    
    def denormalize(self, x): return x * self.std + self.mean

def create_dataloaders(df, seq_len, pred_len, batch_size=32):
    dataset = WaterQualityDataset(df, seq_len, pred_len)
    total_len = len(dataset)
    train_size = int(0.6 * total_len)
    val_size = int(0.2 * total_len)
    
    train_set = torch.utils.data.Subset(dataset, range(0, train_size))
    val_set = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_set = torch.utils.data.Subset(dataset, range(train_size + val_size, total_len))
    
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=False),
            DataLoader(test_set, batch_size=batch_size, shuffle=False), dataset)

class EventWeightedMSE(nn.Module):
    def __init__(self, alpha=2.5):
        super(EventWeightedMSE, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')
    def forward(self, pred, target, is_event):
        loss = self.mse(pred, target)
        weights = torch.ones_like(loss)
        weights[is_event == 1] = self.alpha
        return (loss * weights).mean()

def evaluate_model(model, loader, dataset, device):
    model.eval()
    preds, actuals, events = [], [], []
    with torch.no_grad():
        for bx, by, bevent in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds.append(dataset.denormalize(out).cpu().numpy())
            actuals.append(dataset.denormalize(by).cpu().numpy())
            events.append(bevent.cpu().numpy())
    if len(preds) == 0: return {}
    preds = np.concatenate(preds).flatten()
    actuals = np.concatenate(actuals).flatten()
    events = np.concatenate(events).flatten()
    
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    event_indices = np.where(events == 1)
    sf_score = mean_absolute_error(actuals[event_indices], preds[event_indices]) if len(event_indices[0]) > 0 else 0.0
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Sudden_Fluct_MAE": sf_score}

def run_experiment(df, target_name, source_name, model_type="NLinear"):
    seq_len, pred_len = 96, 24
    train_loader, val_loader, test_loader, dataset = create_dataloaders(df, seq_len, pred_len)
    
    if model_type == "NLinear": 
        model = NLinear(seq_len, pred_len).to(device)
    elif model_type == "DLinear": 
        model = DLinear(seq_len, pred_len).to(device)
        
    criterion = EventWeightedMSE(alpha=2.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print(f"   Training {model_type} on {source_name} ({target_name})...")
    for epoch in range(5): 
        model.train()
        for bx, by, bevent in train_loader:
            bx, by, bevent = bx.to(device), by.to(device), bevent.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by, bevent)
            loss.backward()
            optimizer.step()
            
    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Target": target_name, "Source": source_name, "Model": model_type})
    return metrics

# =================================================================
# 5. MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    # B1: Load Data
    df_ec_sep, df_ph_sep = load_data_source_separate()
    df_ec_api, df_ph_api = load_data_source_api()

    # B2: Visualize (Sẽ lưu vào analysis_chart.png)
    # Hàm này sẽ lấy dữ liệu từ file gốc để vẽ
    plot_comprehensive_analysis(df_ec_sep, df_ph_sep)

    # B3: So sánh Model
    results = []
    print("\n[4/4] Bắt đầu chạy thực nghiệm so sánh NLinear vs DLinear...")
    configs = [
        (df_ec_sep, "EC", "Separate Files"),
        (df_ph_sep, "pH", "Separate Files"),
        (df_ec_api, "EC", "API Combined"),
        (df_ph_api, "pH", "API Combined")
    ]
    
    for df, target, source in configs:
        results.append(run_experiment(df, target, source, "NLinear"))
        results.append(run_experiment(df, target, source, "DLinear"))
    
    final_df = pd.DataFrame(results)
    final_df = final_df[["Model", "Source", "Target", "RMSE", "MAE", "R2", "Sudden_Fluct_MAE"]]
    
    print("\n" + "="*80)
    print("BẢNG SO SÁNH HIỆU NĂNG: NLINEAR vs DLINEAR")
    print("="*80)
    print(final_df)
    final_df.to_csv("comparison_results.csv", index=False)
    print("\nĐã hoàn tất! Kiểm tra 2 file output: 'analysis_chart.png' và 'comparison_results.csv'")