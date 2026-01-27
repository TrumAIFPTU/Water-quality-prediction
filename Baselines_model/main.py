import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from src.decomposition import run_ceemdan
from src.data_loader import load_raw_data, IMFDataset
from src.ltsf_linear import NLinear, DLinear
from src.metrics import metric
from src.path import DATA_DIR, SERIES_DIR, CHECKPOINTS_DIR, CACHE_DIR
from plot_visual import plot_all
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- CONFIGURATION ---
BASE_CONFIG = {
    'seq_len': 168,      
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 50,        
    'ceemd_trials': 50,
    'n_imfs': 12,
    'target': 'EC'       
}

PRED_LENS = [6, 12, 24, 48, 96, 168]
MODELS = ['DLinear', 'NLinear']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(model_name, pred_len, imfs):

    all_imf_preds = []
    print(f"\n>>> Thí nghiệm: {model_name} | Dự báo: {pred_len}h")
    
    # Huấn luyện từng IMF
    for i in range(imfs.shape[0]):
        print(f"\n>>> {model_name} | IMFs: {i}")
        imf_data = imfs[i]
        best_path = CHECKPOINTS_DIR / f"USGs/{BASE_CONFIG['target']}/{model_name}_P{pred_len}_IMF{i}.pth"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        
        train_set = IMFDataset(imf_data, BASE_CONFIG['seq_len'], pred_len, flag='train')
        val_set   = IMFDataset(imf_data, BASE_CONFIG['seq_len'], pred_len, flag='val')
        test_set  = IMFDataset(imf_data, BASE_CONFIG['seq_len'], pred_len, flag='test')
        
        train_loader = DataLoader(train_set, batch_size=BASE_CONFIG['batch_size'], shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_set, batch_size=BASE_CONFIG['batch_size'], shuffle=False)
        test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)
        
        model = DLinear(BASE_CONFIG['seq_len'], pred_len).to(device) if model_name == 'DLinear' else NLinear(BASE_CONFIG['seq_len'], pred_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=BASE_CONFIG['lr'])
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        if not best_path.exists():
            for epoch in range(BASE_CONFIG['epochs']):
                model.train()
                for bx, by in train_loader:
                    optimizer.zero_grad(); out = model(bx.to(device))
                    loss = criterion(out, by.to(device)); loss.backward(); optimizer.step()
                
                model.eval()
                v_loss = []
                with torch.no_grad():
                    for bx, by in val_loader:
                        v_loss.append(criterion(model(bx.to(device)), by.to(device)).item())
                
                avg_v = np.mean(v_loss)
                if avg_v < best_loss:           
                    best_loss = avg_v; torch.save(model.state_dict(), best_path)
        else:
            print(f'>>> Tồn tại {model_name} bắt đầu lấy kết quả dự đoán')
        # Predict
        model.load_state_dict(torch.load(best_path))
        model.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in test_loader:
                preds.append(test_set.inverse(model(bx.to(device)).detach().cpu().numpy().squeeze(0)))
        all_imf_preds.append(np.array(preds))

    # Tổng hợp kết quả: Cộng các IMF dự báo
    final_preds = np.sum(all_imf_preds, axis=0)
    
    # Lấy Ground Truth chuẩn
    final_trues = 0
    for i in range(imfs.shape[0]):
        t_set = IMFDataset(imfs[i], BASE_CONFIG['seq_len'], pred_len, flag='test')
        t_loader = DataLoader(t_set, batch_size=1, shuffle=False)
        imf_t = [t_set.inverse(by.numpy().squeeze(0)) for _, by in t_loader]
        if i == 0: final_trues = np.array(imf_t)
        else: 
            ml = min(len(final_trues), len(imf_t))
            final_trues = final_trues[:ml] + np.array(imf_t)[:ml]

    preds_flat = final_preds.flatten()
    trues_flat = final_trues.flatten()
    m = metric(preds_flat, trues_flat)

    if final_preds.ndim == 3:
        preds_visual = final_preds[:, -1, :].flatten()
        trues_visual = final_trues[:, -1, :].flatten()
    else:
        # Trường hợp dự phòng nếu mảng chỉ có 2 chiều
        preds_visual = final_preds[:, -1].flatten()
        trues_visual = final_trues[:, -1].flatten()
    
    return {
        'RMSE': m[2], 'MAE': m[0], 'MAPE': m[1], 'R2': m[3], 
        'preds_series': preds_visual,  
        'trues_series': trues_visual
    }

def main():
    # 1. Load data và CEEMDAN
    #df = load_raw_data(DATA_DIR/'G_WTP-main/EC_origin.csv', DATA_DIR/'G_WTP-main/pH_origin.csv')
    df = pd.read_csv(DATA_DIR/'USGs/water_data_2021_2025_clean.csv')
    df = df[df['site_no'] == 1463500]
    target_data = df[BASE_CONFIG['target']].values
    
    # Cache IMFs để không phải chạy lại CEEMDAN tốn thời gian
    imfs = run_ceemdan(target_data, trials=BASE_CONFIG['ceemd_trials'], max_imfs=BASE_CONFIG['n_imfs'])
    
    # Lưu IMFs ra CSV để kiểm tra nếu cần
    df_imfs = pd.DataFrame(imfs.T)
    df_imfs.columns = [f'IMF_{i}' for i in range(imfs.shape[0])]
    df_imfs.to_csv(CACHE_DIR/'imfs_data_usgs.csv', index=False)

    results_summary = []

    for m_name in MODELS:
        for p_len in PRED_LENS:
            res = train_and_evaluate(m_name, p_len, imfs)
            
            # Lưu metrics tổng quát
            results_summary.append({
                'Model': m_name, 'Horizon': p_len, 
                'RMSE': res['RMSE'], 'MAE': res['MAE'], 
                'MAPE': res['MAPE'], 'R2': res['R2']
            })
            
            # Lưu chuỗi thời gian Actual vs Predicted
            series_df = pd.DataFrame({
                'Actual': res['trues_series'],
                'Predicted': res['preds_series']
            })
            series_filename = SERIES_DIR / f"USGs/series_{m_name}_P{p_len}_{BASE_CONFIG['target']}.csv"
            series_df.to_csv(series_filename, index=False)
            
            print(f"--> Đã lưu chuỗi thời gian vào: {series_filename.name}")
            print(f"--> Done {m_name} {p_len}h: R2={res['R2']:.4f}")

    # Xuất file kết quả cuối cùng
    pd.DataFrame(results_summary).to_csv(DATA_DIR / f"USGs/final_results_{BASE_CONFIG['target']}.csv", index=False)
    print(">>> ĐANG TRỰC QUAN HÓA DỰ ĐOÁN")
    plot_all(BASE_CONFIG['target'])
    print("\n" + "="*50)
    print(">>> TẤT CẢ THÍ NGHIỆM ĐÃ HOÀN TẤT!")
    print("="*50)

if __name__ == "__main__":
    main()