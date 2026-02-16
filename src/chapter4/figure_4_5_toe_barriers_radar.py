"""
Figure 4.5: Integrated Organisational Barriers to Big Data Adoption (TOE Framework)

Radar chart visualising barriers across Technology, Organisation, and Environment
dimensions of the TOE framework.

Data Sources:
    ECR Retail Loss (2019), Gartner (2023), Accenture (2024),
    BearingPoint (2024), IoD (2025), World Bank (2024)

Author: Jonathan Duque González
Version: 4.0
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate to repository root (two levels up from src/chapter4/)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Set output and data directories relative to repo root
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs')
DATA_DIR = os.path.join(REPO_ROOT, 'data')


def generate_figure(output_path=None):
    """Generate the TOE barriers radar chart."""
    
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection='polar')

    # Barrier labels and values
    labels = [
        'Business\nPessimism\n(80%)',
        'Inventory\nInaccuracies\n(60%)',
        'Data\nInaccuracy\n(62%)',
        'Staff\nTurnover\n(57%)',
        'Unclear AI\nUse Cases\n(52%)',
        'Skills /\nAnalytics Gap\n(48%)',
        'Legacy IT\nSystems\n(29%)',
        'Investment\nGap\n(18%)'
    ]

    values = [80, 60, 62, 57, 52, 48, 29, 18]

    # TOE dimension colours
    toe_colors = [
        '#14A76C',  # Environment
        '#E8505B',  # Technology
        '#E8505B',  # Technology
        '#F9C74F',  # Organisation
        '#F9C74F',  # Organisation
        '#F9C74F',  # Organisation
        '#E8505B',  # Technology
        '#14A76C'   # Environment
    ]

    # Radar setup
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    values_plot = values + values[:1]

    # Plot
    ax.fill(angles, values_plot, color='#2E86AB', alpha=0.25)
    ax.plot(angles, values_plot, color='#2E86AB', linewidth=2)

    # Points
    for angle, val, color in zip(angles[:-1], values, toe_colors):
        ax.scatter(angle, val, s=130, c=color,
                   edgecolors='white', linewidths=2, zorder=5)

    # Axes formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20%', '40%', '60%', '80%'], fontsize=8, color='grey')
    ax.yaxis.grid(True, color='#E0E0E0', alpha=0.6)

    # Title
    ax.set_title(
        'Figure 4.5: Integrated Organisational Barriers to Big Data Adoption (TOE Framework)',
        fontsize=13, fontweight='bold', pad=25
    )

    # Legend
    legend_elements = [
        Patch(facecolor='#E8505B', label='Technology'),
        Patch(facecolor='#F9C74F', label='Organisation'),
        Patch(facecolor='#14A76C', label='Environment')
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=9,
              frameon=False, title='TOE Dimension')

    # Footnote
    plt.figtext(
        0.5, -0.1,
        'Source: Secondary Data Extraction Matrix v2.3; ECR Retail Loss (2019); '
        'Gartner (2023); Accenture (2024); BearingPoint (2024); IoD (2025).\n'
        'Note: Higher values indicate greater organisational constraint.',
        ha='center', fontsize=8, style='italic'
    )

    plt.tight_layout()

    if output_path:
        for fmt in ['png', 'pdf']:
            filepath = f'{output_path}.{fmt}'
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filepath}")

    return fig


def main():
    """Generate Figure 4.5."""
    print("\n" + "="*60)
    print("FIGURE 4.5: TOE Barriers Radar Chart")
    print("="*60)

    output_path = os.path.join(OUTPUT_DIR, 'figure_4_5_toe_barriers_radar')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_figure(output_path)

    print("\n✓ Figure generation complete.\n")


if __name__ == "__main__":
    main()
