import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.Parameters.parameter import RUNTIME_LOG,STABILITY_LOG


def plot_event_vs_normal_error(results_df, save_path="event_vs_normal_error.png"):
    """
    📊 Plot A: Event-Focused Error vs Normal Error Comparison
    Purpose: Prove that the model reduces errors on sudden fluctuations
    while maintaining overall accuracy.
    """
    print("\n🔥 Creating Event vs Normal Error plot...")

    if 'Model' not in results_df.columns or 'MAE' not in results_df.columns:
        print("   ⚠️ Missing required columns")
        return

    models = results_df['Model'].values
    mae_total = results_df['MAE'].values
    sf_mae = results_df['SF_MAE'].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Event-Focused Error Analysis", fontsize=16, fontweight='bold')

    x = np.arange(len(models))
    width = 0.35

    # LEFT: Side-by-side comparison
    bars1 = axes[0].bar(x - width/2, mae_total, width, label='Overall MAE',
                        color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, sf_mae, width, label='Event MAE (SF_MAE)',
                        color='orangered', edgecolor='black', linewidth=1.5)

    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0].set_title('Overall vs Event Error', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=11)
    axes[0].legend(fontsize=11, loc='upper left')
    axes[0].grid(alpha=0.3, axis='y', linestyle='--')

    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    '{:.4f}'.format(height), ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    '{:.4f}'.format(height), ha='center', va='bottom', fontsize=9, color='red')

    # RIGHT: Ratio
    error_ratio = sf_mae / (mae_total + 1e-6)
    colors_ratio = ['green' if r < 1.5 else 'orange' if r < 2.0 else 'red' for r in error_ratio]

    bars3 = axes[1].bar(x, error_ratio, color=colors_ratio,
                       edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1].axhline(1.0, color='black', linestyle='--', linewidth=2, label='Equal Error (1.0)')

    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Error Ratio (Event / Overall)', fontsize=12)
    axes[1].set_title('Event Error Magnification', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=11)
    axes[1].legend(fontsize=10, loc='upper left')
    axes[1].grid(alpha=0.3, axis='y', linestyle='--')

    for i, (bar, ratio) in enumerate(zip(bars3, error_ratio)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    '{:.2f}x'.format(ratio), ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved: {save_path}")

    best_idx = np.argmin(error_ratio)
    print("   📊 Best model: {} (ratio: {:.2f}x)".format(models[best_idx], error_ratio[best_idx]))

def plot_event_distribution(df, threshold, save_path="event_distribution.png"):
    """
    📊 Plot B: Event Distribution & Threshold Validation
    Purpose: Show that events are rare but critical, validate threshold choice.
    """
    print("\n📊 Creating Event Distribution plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Event Detection & Distribution Analysis", fontsize=16, fontweight='bold')

    # TOP: Histogram
    abs_delta = df['abs_delta'].values
    n_events = (abs_delta > threshold).sum()
    event_pct = n_events / len(abs_delta) * 100

    axes[0].hist(abs_delta, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Non-events')
    axes[0].hist(abs_delta[abs_delta > threshold], bins=20, color='red', edgecolor='black', 
                alpha=0.8, label=f'Events (>{threshold:.4f})')

    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'95th Percentile Threshold')

    axes[0].set_xlabel('|Δ log(EC)|', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Log-Differences', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, axis='y')

    # Add statistics box
    stats_text = f'Total samples: {len(abs_delta)}\nEvents: {n_events} ({event_pct:.2f}%)\nThreshold: {threshold:.4f}'
    axes[0].text(0.98, 0.95, stats_text, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # BOTTOM: Time series with event regions
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        axes[1].plot(dates, df['OT_log'], color='blue', linewidth=0.8, label='Log(EC)')

        # Shade event regions
        event_mask = abs_delta > threshold
        axes[1].fill_between(dates, df['OT_log'].min(), df['OT_log'].max(),
                            where=event_mask, color='red', alpha=0.2, label='Event regions')

        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Log(EC)', fontsize=12)
        axes[1].set_title('Time Series with Event Regions Highlighted', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved: {save_path} ")
    print(f"   📊 Events: {n_events}/{len(abs_delta)} ({event_pct:.2f}%)")

def plot_imf_attribution(df, n_imfs=5, save_path="imf_attribution.png"):
    """
    📊 Plot C: IMF Variance Attribution Analysis
    Purpose: Show which IMFs contribute most to fluctuations (justify CEEMD).
    """
    print("\n🌊 Creating IMF Attribution plot...")

    imf_cols = [f'IMF_{i}' for i in range(n_imfs) if f'IMF_{i}' in df.columns]

    if len(imf_cols) == 0:
        print("   ⚠️ No IMF columns found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("CEEMD Decomposition: IMF Variance Attribution", fontsize=16, fontweight='bold')

    # TOP-LEFT: Variance contribution bar chart
    variances = [df[col].var() for col in imf_cols]
    if 'residue' in df.columns:
        variances.append(df['residue'].var())
        labels = imf_cols + ['Residue']
    else:
        labels = imf_cols

    total_var = sum(variances)
    percentages = [v/total_var*100 for v in variances]

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    bars = axes[0, 0].bar(range(len(labels)), percentages, color=colors, edgecolor='black', linewidth=1.5)

    axes[0, 0].set_xlabel('Component', fontsize=12)
    axes[0, 0].set_ylabel('Variance Contribution (%)', fontsize=12)
    axes[0, 0].set_title('Variance Attribution', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(range(len(labels)))
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 0].grid(alpha=0.3, axis='y')

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # TOP-RIGHT: Cumulative energy
    cumulative = np.cumsum(percentages)
    axes[0, 1].plot(range(len(labels)), cumulative, marker='o', linewidth=2, 
                   markersize=8, color='darkblue')
    axes[0, 1].axhline(90, color='red', linestyle='--', label='90% threshold')
    axes[0, 1].fill_between(range(len(labels)), 0, cumulative, alpha=0.2)

    axes[0, 1].set_xlabel('Component', fontsize=12)
    axes[0, 1].set_ylabel('Cumulative Energy (%)', fontsize=12)
    axes[0, 1].set_title('Cumulative Variance', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(range(len(labels)))
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # BOTTOM: IMF_0 (high-freq) vs Residue (low-freq) time series
    if 'date' in df.columns and len(imf_cols) > 0:
        dates = pd.to_datetime(df['date'])

        # Show first 500 samples for clarity
        n_show = min(500, len(dates))

        axes[1, 0].plot(dates[:n_show], df[imf_cols[0]][:n_show], 
                       color='red', linewidth=0.8, label=f'{imf_cols[0]} (High-freq)')
        axes[1, 0].set_xlabel('Date', fontsize=12)
        axes[1, 0].set_ylabel('Amplitude', fontsize=12)
        axes[1, 0].set_title(f'{imf_cols[0]}: High-Frequency Fluctuations', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        if 'residue' in df.columns:
            axes[1, 1].plot(dates[:n_show], df['residue'][:n_show],
                           color='blue', linewidth=1.2, label='Residue (Low-freq trend)')
            axes[1, 1].set_xlabel('Date', fontsize=12)
            axes[1, 1].set_ylabel('Amplitude', fontsize=12)
            axes[1, 1].set_title('Residue: Low-Frequency Trend', fontsize=13, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved:   {save_path}")
    print(f"   📊 Top contributor: {labels[np.argmax(percentages)]} ({max(percentages):.1f}%)")

def plot_runtime_and_stability(save_runtime="runtime_comparison.png", 
                               save_stability="stability_analysis.png"):
    """
    🔧 FIXED: Aggregate runtime log to avoid duplicate bars

    📊 Runtime & Stability Visualization
    """
    name_map = {
    "NLinear": "NLinear",
    "DLinear": "DLinear",
    "CALinear": "CA",
    "Ablation-Baseline": "Abl-Base",
    "Ablation-Feature": "Abl-Feat",
    "Ablation-Event": "Abl-Event",
    "Alpha-1.0": "α=1.0",
    "Alpha-3.0": "α=3.0",
    "Alpha-5.0": "α=5.0",
    }
    

    print(f"\n📊 Creating Runtime & Stability plots...")

    df_runtime = pd.DataFrame(RUNTIME_LOG)
    df_stability = pd.DataFrame(STABILITY_LOG)

    # 🔧 FIXED: Aggregate to avoid duplicate model names in plots
    df_runtime_agg = df_runtime.groupby(['Stage', 'Model', 'Unit'], as_index=False)['Time_s'].mean()

    # 1. Runtime Comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Runtime Benchmark Report (v3.6 Fixed)", fontsize=16, fontweight='bold')

    # Training time
    train_df = df_runtime_agg[df_runtime_agg['Stage'] == 'Training']
    if len(train_df) > 0:
        axes[0].barh(train_df['Model'], train_df['Time_s'], color='steelblue')
        axes[0].set_xlabel("Time (s/epoch)")
        axes[0].set_title("Training Time")
        axes[0].grid(alpha=0.3)

    # Inference + XAI time
    infer_df = df_runtime_agg[df_runtime_agg['Stage'] == 'Inference']
    xai_df = df_runtime_agg[df_runtime_agg['Stage'] == 'XAI']

    if len(infer_df) > 0 and len(xai_df) > 0:
        # Align models
        models = sorted(set(infer_df['Model'].values).intersection(set(xai_df['Model'].values)))

        infer_times = [infer_df[infer_df['Model']==m]['Time_s'].values[0] for m in models]
        xai_times = [xai_df[xai_df['Model']==m]['Time_s'].values[0] for m in models]

        x = np.arange(len(models))
        width = 0.35
        models_short = [name_map.get(m, m) for m in models]
        axes[1].bar(x - width/2, infer_times, width, label='Inference', color='lightgreen')
        axes[1].bar(x + width/2, xai_times, width, label='XAI', color='coral')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models_short , rotation=30, ha='right', fontsize=9)
        axes[1].set_ylabel("Time (ms/sample)")
        axes[1].set_title("Inference & XAI Time")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_runtime, dpi=300)
    plt.close()
    print(f"   ✅ Saved {save_runtime}")

    # 2. Stability Analysis
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Stability Analysis Report (v3.6 Fixed)", fontsize=16, fontweight='bold')

    if len(df_stability) > 0:
        # Noise Robustness
        axes[0].barh(df_stability['Model'], df_stability['Noise_Robustness_Spearman'], color='teal')
        axes[0].axvline(0.9, color='red', linestyle='--', label='Excellent (>0.9)')
        axes[0].set_xlim(0, 1.05)
        axes[0].set_xlabel("Spearman Correlation")
        axes[0].set_title("Noise Robustness")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Perturbation Consistency
        axes[1].barh(df_stability['Model'], df_stability['Perturbation_Consistency_Jaccard'], color='orange')
        axes[1].axvline(0.7, color='red', linestyle='--', label='Good (>0.7)')
        axes[1].set_xlim(0, 1.05)
        axes[1].set_xlabel("Jaccard Index")
        axes[1].set_title("Perturbation Consistency (Top-k)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Attribution Variance
        axes[2].barh(df_stability['Model'], df_stability['Attribution_Variance'], color='purple')
        axes[2].set_xlabel("Variance (lower is better)")
        axes[2].set_title("Attribution Stability (Variance)")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_stability, dpi=300)
    plt.close()
    print(f"   ✅ Saved {save_stability}")
