"""
Figure 2.1: Corporate Strategists' Use of Analytics (Percentage of Respondents)

Recreated from Gartner (2023) survey data to match dissertation style.

Data Source:
    Gartner, Inc. (2023) Survey of 200 corporate strategy leaders (Oct 2022 – Apr 2023).
    n = 209; excludes "discontinued or planning to discontinue use".

Author: Jonathan Duque González
Version: 1.0
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate to repository root (two levels up from src/chapter2/)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Set output and data directories relative to repo root
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs')
DATA_DIR = os.path.join(REPO_ROOT, 'data')


def generate_figure(output_path=None):
    """Generate the analytics adoption stacked bar chart."""
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.facecolor': 'white',
    })

    # Data from Gartner Survey (2023)
    analytics_types = [
        'Digital twins',
        'Geospatial analytics',
        'Text analytics / NLP',
        'Machine learning',
        'Prescriptive analytics',
        'Graph/network analytics',
        'Predictive analytics',
        'Social/Multimedia analytics',
        'Diagnostic analytics',
        'Descriptive analytics'
    ]

    # Percentages: [Deployed, Piloting, Exploring, No plans]
    deployed = [8, 16, 23, 20, 26, 36, 41, 40, 62, 72]
    piloting = [20, 12, 22, 32, 28, 16, 28, 18, 16, 10]
    exploring = [16, 17, 13, 19, 19, 15, 17, 15, 10, 10]
    no_plans = [54, 52, 41, 28, 27, 31, 13, 27, 11, 8]

    # Colour palette
    colors = {
        'deployed': '#1a5276',
        'piloting': '#2980b9',
        'exploring': '#85c1e9',
        'no_plans': '#d5d8dc'
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(analytics_types))
    bar_height = 0.7

    # Create stacked bars
    bars_deployed = ax.barh(y_pos, deployed, bar_height,
                            label='Deployed', color=colors['deployed'],
                            edgecolor='white', linewidth=0.5)
    bars_piloting = ax.barh(y_pos, piloting, bar_height, left=deployed,
                            label='Piloting', color=colors['piloting'],
                            edgecolor='white', linewidth=0.5)
    bars_exploring = ax.barh(y_pos, exploring, bar_height,
                             left=np.array(deployed) + np.array(piloting),
                             label='Exploring / Knowledge gathering',
                             color=colors['exploring'], edgecolor='white', linewidth=0.5)
    bars_no_plans = ax.barh(y_pos, no_plans, bar_height,
                            left=np.array(deployed) + np.array(piloting) + np.array(exploring),
                            label='No plans to deploy currently',
                            color=colors['no_plans'], edgecolor='white', linewidth=0.5)

    # Add percentage labels
    def add_labels(bars, values, cumulative_left, min_width=8):
        for bar, val, left in zip(bars, values, cumulative_left):
            if val >= min_width:
                x_pos = left + val / 2
                y_pos = bar.get_y() + bar.get_height() / 2
                ax.text(x_pos, y_pos, f'{val}%', ha='center', va='center',
                       color='white', fontsize=9, fontweight='bold')

    add_labels(bars_deployed, deployed, [0]*len(deployed))
    add_labels(bars_piloting, piloting, deployed)
    add_labels(bars_exploring, exploring, np.array(deployed) + np.array(piloting))
    add_labels(bars_no_plans, no_plans,
               np.array(deployed) + np.array(piloting) + np.array(exploring))

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(analytics_types, fontsize=10)
    ax.set_xlabel('Percentage of Respondents (%)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # Title
    fig.suptitle("Corporate Strategists' Use of Analytics",
                 fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.93, '(Percentage of Respondents)',
             ha='center', fontsize=11, color='#555555', style='italic')

    # Legend
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                       ncol=4, frameon=True, fancybox=True, shadow=False,
                       edgecolor='#cccccc', fontsize=9)
    legend.get_frame().set_facecolor('white')

    # Source citation
    fig.text(0.5, 0.02,
             'Source: Gartner, Inc. (2023). Survey of 200 corporate strategy leaders '
             '(Oct 2022 – Apr 2023).\n'
             'Note: n = 209; excludes "discontinued or planning to discontinue use". '
             'Numbers may not add to 100% due to rounding.',
             ha='center', fontsize=8, style='italic', color='#666666')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.90)

    if output_path:
        for fmt in ['png', 'pdf']:
            filepath = f'{output_path}.{fmt}'
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filepath}")

    return fig


def main():
    """Generate Figure 2.1."""
    print("\n" + "="*60)
    print("FIGURE 2.1: Corporate Strategists' Use of Analytics")
    print("="*60)

    output_path = os.path.join(OUTPUT_DIR, 'figure_2_1_analytics_adoption')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_figure(output_path)

    print("\n✓ Figure generation complete.\n")


if __name__ == "__main__":
    main()
