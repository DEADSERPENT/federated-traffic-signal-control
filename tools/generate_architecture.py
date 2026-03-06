"""
Generate Professional System Architecture Diagram for ResilNet-FL IEEE Paper
Creates a detailed block diagram showing the three-layer architecture with data flows
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np
import os


def draw_server_icon(ax, x, y, width=0.6, height=0.4, color='#42A5F5'):
    """Draw a server rack icon."""
    for i in range(3):
        rect = FancyBboxPatch((x - width/2, y - height/2 + i*height/3),
                               width, height/3.5,
                               boxstyle="round,pad=0.02,rounding_size=0.02",
                               facecolor=color, edgecolor='#1565C0', linewidth=1)
        ax.add_patch(rect)
        # LED indicators
        ax.plot(x + width/2 - 0.08, y - height/2 + i*height/3 + height/7, 'go', markersize=2)


def draw_intersection_icon(ax, x, y, size=0.4):
    """Draw a traffic intersection icon."""
    # Roads
    ax.add_patch(Rectangle((x - size/2, y - size/6), size, size/3, facecolor='#424242', edgecolor='none'))
    ax.add_patch(Rectangle((x - size/6, y - size/2), size/3, size, facecolor='#424242', edgecolor='none'))
    # Center
    ax.add_patch(Rectangle((x - size/6, y - size/6), size/3, size/3, facecolor='#616161', edgecolor='none'))
    # Traffic lights
    ax.plot(x - size/4, y + size/4, 'ro', markersize=4)
    ax.plot(x + size/4, y - size/4, 'go', markersize=4)


def draw_cloud_icon(ax, x, y, width=1.5, height=0.8, color='#E3F2FD'):
    """Draw a cloud icon."""
    from matplotlib.patches import Ellipse
    # Main cloud body
    ax.add_patch(Ellipse((x, y), width, height, facecolor=color, edgecolor='#1976D2', linewidth=2))
    ax.add_patch(Ellipse((x - width/3, y - height/6), width/2, height/1.5, facecolor=color, edgecolor='#1976D2', linewidth=2))
    ax.add_patch(Ellipse((x + width/3, y - height/6), width/2, height/1.5, facecolor=color, edgecolor='#1976D2', linewidth=2))
    # Cover edges
    ax.add_patch(Ellipse((x, y - height/6), width*0.8, height/1.2, facecolor=color, edgecolor='none'))


def create_system_architecture():
    """Create the ResilNet-FL system architecture diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    layer1_color = '#E8F5E9'  # Light green - Traffic
    layer2_color = '#FFF3E0'  # Light orange - Edge
    layer3_color = '#E3F2FD'  # Light blue - Cloud/FL

    # =========================================================================
    # LAYER 1: TRAFFIC INFRASTRUCTURE (Bottom)
    # =========================================================================
    layer1_box = FancyBboxPatch((0.5, 0.3), 13, 2.5,
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor=layer1_color, edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(layer1_box)
    ax.text(0.8, 2.5, 'Layer 1: Traffic Infrastructure', fontsize=11, fontweight='bold',
            color='#1B5E20', va='top')

    # 4 Intersections
    intersection_x = [2, 5, 9, 12]
    for i, x in enumerate(intersection_x):
        # Intersection icon
        draw_intersection_icon(ax, x, 1.3, size=0.8)
        ax.text(x, 0.6, f'Intersection {i+1}', fontsize=8, ha='center', fontweight='bold')

        # Sensors
        ax.text(x, 2.0, 'Sensors', fontsize=7, ha='center', color='#555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#888', linewidth=0.5))

    # Coordination arrows between intersections
    for i in range(len(intersection_x) - 1):
        ax.annotate('', xy=(intersection_x[i+1] - 0.6, 1.3),
                   xytext=(intersection_x[i] + 0.6, 1.3),
                   arrowprops=dict(arrowstyle='<->', color='#666', lw=1.5, connectionstyle='arc3,rad=0'))

    # =========================================================================
    # LAYER 2: EDGE COMPUTING + NS-3 NETWORK (Middle)
    # =========================================================================
    layer2_box = FancyBboxPatch((0.5, 3.1), 13, 2.8,
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor=layer2_color, edgecolor='#E65100', linewidth=2)
    ax.add_patch(layer2_box)
    ax.text(0.8, 5.7, 'Layer 2: Edge Computing (CloudSim) + NS-3 Network', fontsize=11,
            fontweight='bold', color='#E65100', va='top')

    # Edge servers
    edge_x = [2, 5, 9, 12]
    for i, x in enumerate(edge_x):
        # Edge server box
        server_box = FancyBboxPatch((x - 0.7, 3.8), 1.4, 1.2,
                                     boxstyle="round,pad=0.03,rounding_size=0.1",
                                     facecolor='white', edgecolor='#FF6F00', linewidth=1.5)
        ax.add_patch(server_box)

        # Server icon inside
        for j in range(3):
            rect = Rectangle((x - 0.45, 4.0 + j*0.25), 0.9, 0.18,
                            facecolor='#FFB74D', edgecolor='#F57C00', linewidth=0.5)
            ax.add_patch(rect)

        ax.text(x, 3.55, f'Edge Server {i+1}', fontsize=7, ha='center', fontweight='bold')
        ax.text(x, 4.75, 'Local\nTraining', fontsize=6, ha='center', color='#555')

    # NS-3 Network visualization (wavy lines representing wireless)
    ns3_box = FancyBboxPatch((3.5, 5.0), 7, 0.6,
                              boxstyle="round,pad=0.03,rounding_size=0.1",
                              facecolor='#FFECB3', edgecolor='#FFA000', linewidth=1.5, linestyle='--')
    ax.add_patch(ns3_box)
    ax.text(7, 5.3, 'NS-3: IEEE 802.11p DSRC Channel', fontsize=9, ha='center',
            fontweight='bold', color='#E65100')

    # Arrows from intersections to edge servers
    for x in edge_x:
        ax.annotate('', xy=(x, 3.75), xytext=(x, 2.85),
                   arrowprops=dict(arrowstyle='->', color='#388E3C', lw=2))

    # =========================================================================
    # LAYER 3: FEDERATED LEARNING (Top)
    # =========================================================================
    layer3_box = FancyBboxPatch((0.5, 6.2), 13, 3.3,
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor=layer3_color, edgecolor='#1565C0', linewidth=2)
    ax.add_patch(layer3_box)
    ax.text(0.8, 9.3, 'Layer 3: Federated Learning (FedProx)', fontsize=11,
            fontweight='bold', color='#0D47A1', va='top')

    # Central Cloud Server
    cloud_box = FancyBboxPatch((5, 7.8), 4, 1.4,
                                boxstyle="round,pad=0.05,rounding_size=0.3",
                                facecolor='#BBDEFB', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(cloud_box)

    # Server rack inside cloud
    for j in range(3):
        rect = Rectangle((6.2, 8.0 + j*0.3), 1.6, 0.22,
                        facecolor='#64B5F6', edgecolor='#1976D2', linewidth=0.5)
        ax.add_patch(rect)

    ax.text(7, 8.95, 'Cloud Aggregation Server', fontsize=10, ha='center', fontweight='bold', color='#0D47A1')
    ax.text(7, 7.65, 'FedProx Weighted Averaging', fontsize=8, ha='center', color='#1565C0')

    # Local model boxes at edges (in FL layer)
    local_model_x = [2, 5, 9, 12]
    for i, x in enumerate(local_model_x):
        model_box = FancyBboxPatch((x - 0.5, 6.6), 1.0, 0.7,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor='white', edgecolor='#1976D2', linewidth=1.5)
        ax.add_patch(model_box)
        ax.text(x, 7.05, r'$w_k$', fontsize=10, ha='center', fontweight='bold', color='#1565C0')
        ax.text(x, 6.75, f'Model {i+1}', fontsize=7, ha='center')

    # Arrows: Upload model updates (red)
    for x in local_model_x:
        if x < 7:
            end_x = 5.1
        else:
            end_x = 8.9
        ax.annotate('', xy=(end_x, 7.9), xytext=(x, 7.35),
                   arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5,
                                  connectionstyle='arc3,rad=0.2'))

    # Arrows: Download global model (blue)
    for x in local_model_x:
        if x < 7:
            start_x = 5.1
        else:
            start_x = 8.9
        ax.annotate('', xy=(x, 7.35), xytext=(start_x, 8.1),
                   arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5,
                                  connectionstyle='arc3,rad=-0.2'))

    # Arrows from edge servers to local models
    for x in edge_x:
        ax.annotate('', xy=(x, 6.55), xytext=(x, 5.65),
                   arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5))

    # =========================================================================
    # LEGEND
    # =========================================================================
    legend_y = 0.1
    legend_items = [
        ('Upload Gradients', '#C62828', '-'),
        ('Download Global Model', '#1565C0', '-'),
        ('Data Flow', '#388E3C', '-'),
    ]

    for i, (label, color, style) in enumerate(legend_items):
        ax.annotate('', xy=(1.2 + i*4, legend_y), xytext=(0.5 + i*4, legend_y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(1.4 + i*4, legend_y, label, fontsize=8, va='center')

    # Key equations/annotations
    ax.text(11.5, 8.5, r'$w_{t+1} = \frac{\sum \alpha_k w_k}{\sum \alpha_k}$',
            fontsize=10, ha='center', color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1976D2'))

    ax.text(11.5, 7.1, r'$\mathcal{L} = \mathcal{L}_{task} + \frac{\mu}{2}\|w-w_g\|^2$',
            fontsize=9, ha='center', color='#E65100',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#FF6F00'))

    # Title
    ax.text(7, 9.7, 'ResilNet-FL: System Architecture', fontsize=16, fontweight='bold',
            ha='center', color='#263238')

    plt.tight_layout()

    # Save
    os.makedirs('results/ieee', exist_ok=True)
    plt.savefig('results/ieee/system_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('results/ieee/system_architecture.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print("System architecture diagram saved to:")
    print("  - results/ieee/system_architecture.png")
    print("  - results/ieee/system_architecture.pdf")

    plt.close()


if __name__ == "__main__":
    create_system_architecture()
