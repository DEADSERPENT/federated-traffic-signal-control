import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# --- 1. Apply Professional Style (Same as your other plots) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.titlesize': 14, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Your Project Colors
COLORS = {
    'fl': '#2E86AB',     # Blue
    'local': '#A23B72',  # Magenta
}

def plot_generalization():
    # --- 2. Data from your logs (Seed 9999) ---
    methods = ['Local-ML', 'FL (Ours)']
    mae_values = [2.2930, 1.9494]  # From your log output
    improvement = 14.98            # Your calculated improvement
    
    # Create Figure
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot Bars
    bars = ax.bar(methods, mae_values, color=[COLORS['local'], COLORS['fl']], 
                  edgecolor='black', width=0.5)
    
    # --- 3. Styling & Annotations ---
    ax.set_ylabel('Mean Absolute Error (Lower is Better)')
    ax.set_title('Generalization Performance on Unseen Traffic\n(Seed 9999 - Unknown Scenario)')
    ax.set_ylim(0, 2.8)  # Give headroom for text
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add Value Labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # Add "Improvement" Bracket/Arrow
    # Coordinates for the arrow
    x_local = bars[0].get_x() + bars[0].get_width()/2
    x_fl = bars[1].get_x() + bars[1].get_width()/2
    y_max = max(mae_values)
    
    # Draw a curved arrow from Local to FL indicating drop
    ax.annotate(f'', 
                xy=(x_fl, mae_values[1] + 0.1), 
                xytext=(x_local, mae_values[0] + 0.1),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", 
                              color='green', lw=2))
    
    # Text label for the improvement
    mid_x = (x_local + x_fl) / 2
    mid_y = (mae_values[0] + mae_values[1]) / 2 + 0.25
    
    ax.text(mid_x, mid_y, 
            f'-{improvement}%\nError Reduction', 
            ha='center', va='center', 
            color='green', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    # --- 4. Save ---
    save_path = Path('results/ieee/ieee_generalization.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Generalization plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_generalization()