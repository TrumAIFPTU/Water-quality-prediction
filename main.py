import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.Parameters.parameter import CONFIG,RUNTIME_LOG,STABILITY_LOG
from src.seed import seed_everything
from src.Data.data_loading import load_data_source_api,load_data_source_separate,create_dataloaders
from src.CEEMD.ceemd_filter import apply_ceemd_decomposition
from src.Computing.compute import compute_event_threshold_from_train,compute_rolling_features_post_split,detect_events_from_threshold
from src.Experiments.deploy_experiments import run_alpha_sensitivity_analysis,run_ablation_study,run_experiment,run_multi_horizon_experiments
from src.Visualize.visual import plot_event_distribution,plot_event_vs_normal_error,plot_imf_attribution,plot_runtime_and_stability
from src.Statistical.stats import run_stability_analysis
from src.Utils.path import OUTPUT_DIR

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Running on device: {device}")

def main():
    """Main v3.6 FINAL FIXED - Bug-free thesis version with ALL PLOTS"""
    print("="*80)
    print("🌊 WATER QUALITY FORECASTING v3.6 FINAL FIXED + FULL POWER")
    print("="*80)
    print("✅ ALL CRITICAL BUGS FIXED:")
    print("   🐛 Inference time (per-sample, not per-batch)")
    print("   🐛 Attribution variance (variance of attributions, not correlations)")
    print("   🐛 Baseline strategy (TimeSHAP-compliant)")
    print("   🐛 Metric names (Perturbation Consistency)")
    print("   🐛 Runtime aggregation (no duplicate bars)")
    print("   🔥 CEEMD Full Power (100 trials, all data, 5 IMFs)")
    print("="*80)

    # Seed for reproducibility
    seed_everything(42)

    # [1/8] LOAD DATA
    df_ec, df_ph = load_data_source_separate()
    if df_ec is None or df_ph is None:
        df_ec, df_ph = load_data_source_api()
    if df_ec is None or df_ph is None:
        print("❌ Cannot load data")
        return

    # [2/8] CEEMD DECOMPOSITION (FULL POWER)
    n_imfs_ec = 5
    try:
        df_ec, n_imfs_ec = apply_ceemd_decomposition(df_ec, n_imfs=5)
    except Exception as e:
        print(f"   ⚠️ CEEMD skipped: {e}")

    # [2.5/8] EVENT DETECTION & DISTRIBUTION PLOT
    print("\n" + "="*80)
    print("📊 [2.5/8] Event Distribution Analysis")
    print("="*80)
    
    loaders_tmp, dataset_tmp, split_info = create_dataloaders(
        df_ec, CONFIG['seq_len'], CONFIG['pred_len'], CONFIG['batch_size'], False
    )
    train_size = split_info['train_size']
    train_end_idx = train_size + CONFIG['seq_len']
    df_train = df_ec.iloc[:train_end_idx].copy()
    
    event_threshold = compute_event_threshold_from_train(
        df_train, CONFIG['event_threshold_pct']
    )
    df_ec = detect_events_from_threshold(df_ec, event_threshold)
    
    # ✅ CRITICAL PLOT 1
    plot_event_distribution(df_ec, event_threshold,save_path= OUTPUT_DIR/"image/event_distribution.png")
    
    # [2.6/8] IMF ATTRIBUTION PLOT
    if 'IMF_0' in df_ec.columns:
        print("\n🌊 [2.6/8] IMF Contribution Analysis")
        # ✅ CRITICAL PLOT 2
        plot_imf_attribution(df_ec, n_imfs=n_imfs_ec, save_path=OUTPUT_DIR/"image/imf_attribution.png")


    # [3/8] STANDARD MODEL COMPARISON
    print("\n" + "="*80)
    print("📊 [3/8] STANDARD: Model Comparison")
    print("="*80)

    results = []
    for model_type in ["NLinear", "DLinear", "CALinear"]:
        metrics, model, dataset, test_loader, df_updated = run_experiment(
            df_ec.copy(),device, "EC", "Files", model_type
        )
        results.append(metrics)

    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_DIR/"report/comparison_results.csv", index=False)
    
    # ✅ CRITICAL PLOT 3
    plot_event_vs_normal_error(final_df, OUTPUT_DIR/"image/event_vs_normal_error.png")

    # [4/8] MULTI-HORIZON
    horizons = [12, 24, 48]
    horizon_results = run_multi_horizon_experiments(
        df_ec.copy(), "EC", "Files", "CALinear", horizons
    )
    horizon_results.to_csv(OUTPUT_DIR/"report/horizon_comparison.csv", index=False)

    # [5/8] ABLATION
    ablation_df = run_ablation_study(df_ec.copy(),device, "EC", "Files")
    ablation_df.to_csv(OUTPUT_DIR/"report/ablation_study.csv", index=False)

    # [6/8] ALPHA
    alpha_df = run_alpha_sensitivity_analysis(df_ec.copy(),device, "EC", "Files", [1.0, 3.0, 5.0])
    alpha_df.to_csv(OUTPUT_DIR/"report/alpha_sensitivity.csv", index=False)

    # [7/8] STABILITY
    for model_type in ["NLinear", "DLinear", "CALinear"]:
        metrics, model, dataset, test_loader, _ = run_experiment(
            df_ec.copy(),device, "EC", "Files", model_type
        )
        run_stability_analysis(model, dataset, test_loader, device, model_type)

    # [8/8] SAVE REPORTS
    df_runtime = pd.DataFrame(RUNTIME_LOG)
    df_stability = pd.DataFrame(STABILITY_LOG)
    df_runtime.to_csv(OUTPUT_DIR/"report/runtime_report.csv", index=False)
    df_stability.to_csv(OUTPUT_DIR/"report/stability_report.csv", index=False)

    plot_runtime_and_stability(save_runtime=OUTPUT_DIR/"image/runtime_comparison.png", 
                               save_stability=OUTPUT_DIR/"image/stability_analysis.png")

    print("\n EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    os.makedirs(OUTPUT_DIR/'report',exist_ok=True)
    os.makedirs(OUTPUT_DIR/'image',exist_ok=True)
    main()