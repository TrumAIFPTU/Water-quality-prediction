import numpy as np
import pandas as pd
import torch 
import time
import torch.nn as nn
from src.Parameters.parameter import CONFIG,RUNTIME_LOG,STABILITY_LOG
from src.Data.data_loading import create_dataloaders
from src.LinearModel import DLinear,CALinear,NLinear
from src.CEEMD.ceemd_filter import EventWeightedMSE
from src.evaluate.evaluate import evaluate_model
from src.Training.Model_training import train_model_simple,train_model
from src.Computing.compute import compute_event_threshold_from_train,compute_rolling_features_post_split,detect_events_from_threshold

def run_experiment(df,device ,target_name, source_name, model_type="NLinear"):
    """
    🔧 FIXED: Correct inference time calculation (per-sample, not per-batch)

    Run experiment with RUNTIME tracking for Training, Inference, and XAI.
    """
    print(f"\n🚀 Training {model_type} on {source_name} ({target_name})...")

    seq_len, pred_len = CONFIG['seq_len'], CONFIG['pred_len']
    use_features = (model_type == "CALinear")

    # Create temporary dataset for split info only
    loaders_tmp, dataset_tmp, split_info = create_dataloaders(
        df, seq_len, pred_len, CONFIG['batch_size'], use_features=False
    )

    train_size = split_info['train_size']
    train_end_idx = train_size + seq_len
    df_train = df.iloc[:train_end_idx].copy()

    event_threshold = compute_event_threshold_from_train(
        df_train, CONFIG['event_threshold_pct']
    )

    df = detect_events_from_threshold(df, event_threshold)

    if use_features:
        df = compute_rolling_features_post_split(df, train_size + seq_len)

    loaders, dataset, split_info = create_dataloaders(
        df, seq_len, pred_len, CONFIG['batch_size'], use_features=use_features
    )
    train_loader, val_loader, test_loader = loaders

    if model_type == "NLinear":
        model = NLinear(seq_len, pred_len).to(device)
    elif model_type == "DLinear":
        model = DLinear(seq_len, pred_len).to(device)
    elif model_type == "CALinear":
        model = CALinear(seq_len, pred_len, n_features=3).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = EventWeightedMSE(alpha=CONFIG['event_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                       CONFIG['epochs'], device, model_name=model_type)

    # ⏱️ FIXED: MEASURE INFERENCE TIME PER SAMPLE CORRECTLY
    inference_times_per_sample = []
    model.eval()
    with torch.no_grad():
        for bx, by, _ in test_loader:
            batch_size = bx.size(0)
            start = time.time()
            _ = model(bx.to(device))
            elapsed = time.time() - start
            # Divide by batch size to get per-sample time
            inference_times_per_sample.append(elapsed / batch_size)

    # Average inference time per sample in milliseconds
    avg_infer_ms = np.mean(inference_times_per_sample) * 1000

    RUNTIME_LOG.append({
        "Stage": "Inference",
        "Model": model_type,
        "Time_s": avg_infer_ms,
        "Unit": "ms/sample"
    })

    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({
        "Target": target_name,
        "Source": source_name,
        "Model": model_type
    })

    return metrics, model, dataset, test_loader, df

def run_multi_horizon_experiments(df, target_name, source_name, model_type="NLinear",
                                   horizons=[6, 12, 24, 48, 72]):
    """Multi-horizon with runtime tracking"""
    print(f"\n🔬 Multi-horizon: {model_type}...")

    results_by_horizon = []
    original_pred_len = CONFIG['pred_len']

    # ⏱️ START MULTI-HORIZON TIMER
    mh_start = time.time()

    for pred_len in horizons:
        print(f"   Horizon: {pred_len}h")

        try:
            CONFIG['pred_len'] = pred_len
            metrics, model, dataset, test_loader, df_updated = run_experiment(
                df, target_name, source_name, model_type
            )
            metrics['Horizon'] = pred_len
            results_by_horizon.append(metrics)
        except Exception as e:
            print(f"   ⚠️ Failed: {str(e)[:50]}")
        finally:
            CONFIG['pred_len'] = original_pred_len

    # ⏱️ LOG MULTI-HORIZON TIME
    mh_time = time.time() - mh_start
    RUNTIME_LOG.append({
        "Stage": "Multi-Horizon",
        "Model": model_type,
        "Time_s": mh_time,
        "Unit": "s/total"
    })

    return pd.DataFrame(results_by_horizon)

def run_ablation_study(df, device, target_name, source_name):
    """Ablation study"""
    print(f"\n🔬 Ablation study...")

    results = []
    seq_len, pred_len = CONFIG['seq_len'], CONFIG['pred_len']

    loaders_tmp, dataset_tmp, split_info = create_dataloaders(
        df, seq_len, pred_len, CONFIG['batch_size'], False
    )
    train_size = split_info['train_size']
    train_end_idx = train_size + seq_len
    df_train = df.iloc[:train_end_idx].copy()

    event_threshold = compute_event_threshold_from_train(
        df_train, CONFIG['event_threshold_pct']
    )
    df = detect_events_from_threshold(df, event_threshold)

    # 1. Baseline
    print("   [1/4] Baseline")
    loaders, dataset, _ = create_dataloaders(df, seq_len, pred_len, CONFIG['batch_size'], False)
    train_loader, val_loader, test_loader = loaders

    model = NLinear(seq_len, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model = train_model_simple(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, "Ablation-Baseline")
    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Variant": "Baseline", "Change_Features": "No", "Event_Loss": "No"})
    results.append(metrics)

    # 2. + Features
    print("   [2/4] + Features")
    df_with_features = compute_rolling_features_post_split(df, train_size + seq_len)
    loaders, dataset, _ = create_dataloaders(df_with_features, seq_len, pred_len, CONFIG['batch_size'], True)
    train_loader, val_loader, test_loader = loaders

    model = CALinear(seq_len, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model = train_model_simple(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, "Ablation-+Features")
    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Variant": "+ Features", "Change_Features": "Yes", "Event_Loss": "No"})
    results.append(metrics)

    # 3. + Event loss
    print("   [3/4] + Event loss")
    loaders, dataset, _ = create_dataloaders(df, seq_len, pred_len, CONFIG['batch_size'], False)
    train_loader, val_loader, test_loader = loaders

    model = NLinear(seq_len, pred_len).to(device)
    criterion = EventWeightedMSE(alpha=CONFIG['event_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model = train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, "Ablation-+EventLoss")
    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Variant": "+ Event loss", "Change_Features": "No", "Event_Loss": "Yes"})
    results.append(metrics)

    # 4. Full
    print("   [4/4] Full")
    loaders, dataset, _ = create_dataloaders(df_with_features, seq_len, pred_len, CONFIG['batch_size'], True)
    train_loader, val_loader, test_loader = loaders

    model = CALinear(seq_len, pred_len).to(device)
    criterion = EventWeightedMSE(alpha=CONFIG['event_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model = train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, "Ablation-Full")
    metrics = evaluate_model(model, test_loader, dataset, device)
    metrics.update({"Variant": "Full", "Change_Features": "Yes", "Event_Loss": "Yes"})
    results.append(metrics)

    return pd.DataFrame(results)

def run_alpha_sensitivity_analysis(df, device, target_name, source_name, 
                                    alphas=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0]):
    """Alpha sensitivity"""
    print(f"\n🔬 Alpha sensitivity...")

    results = []
    seq_len, pred_len = CONFIG['seq_len'], CONFIG['pred_len']

    loaders_tmp, dataset_tmp, split_info = create_dataloaders(
        df, seq_len, pred_len, CONFIG['batch_size'], False
    )
    train_size = split_info['train_size']
    train_end_idx = train_size + seq_len
    df_train = df.iloc[:train_end_idx].copy()

    event_threshold = compute_event_threshold_from_train(
        df_train, CONFIG['event_threshold_pct']
    )
    df = detect_events_from_threshold(df, event_threshold)
    df = compute_rolling_features_post_split(df, train_size + seq_len)

    for alpha in alphas:
        print(f"   Alpha={alpha}")

        loaders, dataset, _ = create_dataloaders(df, seq_len, pred_len, CONFIG['batch_size'], True)
        train_loader, val_loader, test_loader = loaders

        model = CALinear(seq_len, pred_len).to(device)
        criterion = EventWeightedMSE(alpha=alpha)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        model = train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'], device, f"Alpha-{alpha}")
        metrics = evaluate_model(model, test_loader, dataset, device)
        metrics['Alpha'] = alpha
        results.append(metrics)

    return pd.DataFrame(results)