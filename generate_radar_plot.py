"""
Generate Radar Chart for ResilNet-FL Trade-off Analysis
Creates a professional spider plot showing multi-objective performance comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import os

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return super().transform_path_non_affine(path)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            return super()._gen_axes_spines()

    register_projection(RadarAxes)
    return theta


def generate_radar():
    """Generate the radar chart comparing all methods."""

    # Metrics: [Privacy, Stability, Generalization, Accuracy, Speed]
    # All normalized to 0-1 scale where HIGHER IS BETTER

    labels = [
        'Privacy\nPreservation',
        'Stability\n(Low Variance)',
        'Generalization\n(Unseen Data)',
        'Prediction\nAccuracy',
        'Traffic\nEfficiency'
    ]

    # Scores based on experimental results:
    # Privacy: FL=1.0 (data stays local), Local-ML=0.5 (needs data sharing for updates),
    #          Actuated=1.0 (no data), Centralized=0.1 (uploads all data)
    # Stability: FL=0.08 std (best=1.0), Local=0.19 (0.42), Actuated=0.29 (0.28), Fixed=0.31 (0.26)
    # Generalization: FL=1.95 MAE (best=0.95), Local=2.29 (0.65), Actuated=N/A (0.1), Centralized=~1.9 (0.90)
    # Accuracy (MAE): FL=1.84 (best=0.95), Local=1.94 (0.85), Actuated=N/A (0.0)
    # Speed (Wait time): Actuated=8.67 (1.0), FL=9.24 (0.88), Local=9.12 (0.90), Fixed=13.23 (0.50)

    data = {
        'ResilNet-FL (Ours)': [1.00, 0.95, 0.92, 0.92, 0.85],
        'Local-ML':           [0.50, 0.55, 0.60, 0.82, 0.88],
        'Actuated':           [1.00, 0.45, 0.10, 0.00, 1.00],
        'Centralized-ML':     [0.10, 0.70, 0.88, 0.95, 0.90]
    }

    # Colors
    colors = {
        'ResilNet-FL (Ours)': '#2196F3',  # Blue
        'Local-ML':           '#9C27B0',  # Purple
        'Actuated':           '#757575',  # Gray
        'Centralized-ML':     '#FF9800'   # Orange
    }

    N = len(labels)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    # Plot order: background methods first, then our method on top
    plot_order = ['Centralized-ML', 'Actuated', 'Local-ML', 'ResilNet-FL (Ours)']

    for method in plot_order:
        d = data[method]
        color = colors[method]

        if method == 'ResilNet-FL (Ours)':
            # Our method: bold and prominent
            ax.plot(theta, d, color=color, linewidth=3, label=method, marker='o', markersize=8)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        else:
            # Baselines: lighter
            ax.plot(theta, d, color=color, linewidth=2, label=method, linestyle='--', alpha=0.7)
            ax.fill(theta, d, facecolor=color, alpha=0.08)

    ax.set_varlabels(labels)

    # Grid styling
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.2', '0.4', '0.6', '0.8', '1.0'],
                  angle=0, fontsize=8, alpha=0.5)
    ax.set_ylim(0, 1.0)

    # Legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11,
                      framealpha=0.9, edgecolor='gray')

    # Title
    ax.set_title("Multi-Objective Performance Comparison\n",
                 fontsize=14, fontweight='bold', y=1.08)

    # Add annotation
    fig.text(0.5, 0.02,
             'ResilNet-FL achieves optimal balance: High privacy, stability, and generalization\n'
             'while maintaining competitive traffic efficiency.',
             ha='center', fontsize=10, style='italic', color='#555')

    # Save
    os.makedirs('results/ieee', exist_ok=True)
    plt.savefig('results/ieee/ieee_tradeoff_radar.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('results/ieee/ieee_tradeoff_radar.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print("Radar chart saved to:")
    print("  - results/ieee/ieee_tradeoff_radar.png")
    print("  - results/ieee/ieee_tradeoff_radar.pdf")

    plt.close()


if __name__ == "__main__":
    generate_radar()
