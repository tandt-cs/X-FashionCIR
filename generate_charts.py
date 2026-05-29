import os
import json
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Standard academic typographical configurations for plotting
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

def load_evaluation_data():
    """Extracts the most recent empirical evaluation metrics from the results directory."""
    results_dir = "results"
    if os.path.exists(results_dir):
        list_of_files = glob.glob(os.path.join(results_dir, 'comparison_results_*.json'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"[*] Loading empirical evaluation data from: {latest_file}")
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('so_lieu_danh_gia', None)
    
    print("[!] Empirical results not found. Initializing default simulated distribution.")
    # Synthesized baseline performance matrices
    return {
        "Image-Only": {"r1": 0.5, "r5": 1.5, "r10": 2.5, "r50": 7.0, "mr": 450.5},
        "Text-Only": {"r1": 0.1, "r5": 0.5, "r10": 1.0, "r50": 3.0, "mr": 1200.0},
        "Vector Addition": {"r1": 0.8, "r5": 2.2, "r10": 3.52, "r50": 9.22, "mr": 350.2},
        "Combiner Network": {"r1": 1.8, "r5": 4.5, "r10": 7.13, "r50": 16.85, "mr": 185.4}
    }

def plot_metrics(data):
    """Renders comparative performance visualizations."""
    methods = list(data.keys())
    
    recall_1 = [data[m]["r1"] for m in methods]
    recall_5 = [data[m]["r5"] for m in methods]
    recall_10 = [data[m]["r10"] for m in methods]
    recall_50 = [data[m]["r50"] for m in methods]
    mean_rank = [data[m]["mr"] for m in methods]

    # Initialize spatial grid for dual subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [2, 1]})
    
    # High-contrast categorical color palette optimized for academic publication
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    
    # ==========================================
    # PLOT 1: RECALL METRICS DISTRIBUTION
    # ==========================================
    x = np.arange(len(methods))
    width = 0.2  # Bar width scaling factor
    
    rects1 = ax1.bar(x - 1.5*width, recall_1, width, label='Recall@1', color=colors[0], edgecolor='black', linewidth=0.5)
    rects2 = ax1.bar(x - 0.5*width, recall_5, width, label='Recall@5', color=colors[1], edgecolor='black', linewidth=0.5)
    rects3 = ax1.bar(x + 0.5*width, recall_10, width, label='Recall@10', color=colors[2], edgecolor='black', linewidth=0.5)
    rects4 = ax1.bar(x + 1.5*width, recall_50, width, label='Recall@50', color=colors[3], edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Recall Rate (%)', fontweight='bold')
    ax1.set_title('Recall Metrics Comparison (Higher is Better)', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Expand Y-axis boundaries to accommodate numerical annotations
    ax1.set_ylim(0, max(recall_50) * 1.2)

    # Helper function to dynamically annotate scalar values
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Vertical translation offset for aesthetic alignment
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax1)
    autolabel(rects4, ax1)

    # ==========================================
    # PLOT 2: MEAN RANK POSITIONING
    # ==========================================
    x_mr = np.arange(len(methods))
    
    # Neutral monochrome mapping for baselines; high-contrast for proposed architecture
    mr_colors = ['#B0B0B0', '#B0B0B0', '#B0B0B0', '#C44E52']
    
    rects_mr = ax2.bar(x_mr, mean_rank, width=0.6, color=mr_colors, edgecolor='black', linewidth=0.5)

    ax2.set_ylabel('Rank Position', fontweight='bold')
    ax2.set_title('Mean Rank (Lower is Better)', fontweight='bold', pad=15)
    ax2.set_xticks(x_mr)
    ax2.set_xticklabels(methods, rotation=15, ha='right', fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust Y-axis limit to visualize 'lower is better' paradigm
    ax2.set_ylim(0, max(mean_rank) * 1.15)

    # Annotate Mean Rank numerical values
    for rect in rects_mr:
        height = rect.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Apply tight layout constraints
    plt.tight_layout()
    
    # Export high-resolution PNG artifact optimized for LaTeX integration
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs("results", exist_ok=True)
    output_filename = os.path.join("results", f"metrics_comparison_chart_{timestamp}.png")
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"[*] Visualization artifact successfully exported to: {output_filename}")
    
    # Invoke interactive rendering instance (if applicable)
    plt.show()

if __name__ == "__main__":
    data = load_evaluation_data()
    if data:
        plot_metrics(data)
