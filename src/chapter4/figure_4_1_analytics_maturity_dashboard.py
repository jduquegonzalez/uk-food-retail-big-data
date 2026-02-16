"""
Figure 4.1: Analytics Maturity Gap and Adoption Patterns in UK Food Retail

Multi-panel dashboard synthesising analytics maturity landscape:
- Panel A: Global analytics maturity gap (Gartner, 2023)
- Panel B: UK retailer maturity levels (Corporate disclosures FY 2024/25)
- Panel C: UK retail technology adoption butterfly chart

Data Sources:
    Gartner (2023) Survey of 209 corporate strategists
    BearingPoint (2024) UK Retail AI/ML adoption survey
    Retail Economics/NatWest (2025) Automation adoption survey (n=100)
    Corporate annual reports FY 2024/25

Author: Jonathan Duque González
Version: 3.1
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
    """Generate the three-panel analytics maturity dashboard."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], wspace=0.35, hspace=0.4)

    # =========================================================================
    # PANEL A: Global Analytics Maturity (Gartner 2023)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    maturity_levels = ['Descriptive', 'Diagnostic', 'Predictive', 'Prescriptive']
    gartner_global = [72, 62, 41, 26]
    colors_maturity = ['#2E86AB', '#14A76C', '#F9C74F', '#E8505B']

    bars = ax1.bar(maturity_levels, gartner_global, color=colors_maturity,
                   edgecolor='white', linewidth=1.5, width=0.7)

    # Value labels
    for bar, val in zip(bars, gartner_global):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val}%',
                 ha='center', fontsize=11, fontweight='bold')

    # Maturity gap line
    ax1.plot([0, 3], [85, 40], color='#C73E1D', linestyle='--', linewidth=1.5,
             marker='o', markersize=7, zorder=5)
    ax1.annotate('46pp\nMaturity Gap', xy=(1.5, 70), fontsize=10, fontweight='bold',
                 color='#C73E1D', ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#C73E1D', alpha=0.9))

    ax1.set_ylabel('Adoption Rate (%)', fontsize=10, fontweight='bold')
    ax1.set_title('A: Global Analytics Maturity\n(Gartner, 2023; n=209)',
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 95)
    ax1.spines[['top', 'right']].set_visible(False)

    # =========================================================================
    # PANEL B: UK Retailer Maturity — Horizontal Bars
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    # Updated Retailers and Scores based on Table 4.1
    # 4.0 = Prescriptive
    # 3.0 = Predictive
    # 2.5 = Diagnostic / Predictive
    # 2.0 = Diagnostic
    # 1.5 = Diagnostic (Transitioning)
    
    retailers = [
        'Ocado', 'Tesco',           # Prescriptive
        "Sainsbury's", 'M&S',       # Predictive
        'Aldi', 'Asda', 'Iceland',  # Predictive
        'Morrisons',                # Diagnostic / Predictive
        'Waitrose',                 # Diagnostic
        'Lidl'                      # Diagnostic (Transitioning)
    ]
    
    maturity_scores = [4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.5, 2.0, 1.5]

    # Sort data for plotting
    sorted_idx = np.argsort(maturity_scores)[::-1] # Sort descending
    retailers_sorted = [retailers[i] for i in sorted_idx]
    maturity_sorted = [maturity_scores[i] for i in sorted_idx]

    def get_color(score):
        if score >= 4.0:
            return '#E8505B'  # Prescriptive (Red)
        elif score >= 3.0:
            return '#F9C74F'  # Predictive (Yellow)
        else:
            return '#14A76C'  # Diagnostic (Green)

    colors_bars = [get_color(s) for s in maturity_sorted]
    y_pos = np.arange(len(retailers_sorted))

    ax2.barh(y_pos, maturity_sorted, color=colors_bars, edgecolor='white',
             linewidth=1.5, height=0.7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(retailers_sorted, fontsize=10)
    ax2.set_xlabel('Analytics Maturity Level', fontsize=10, fontweight='bold')
    ax2.set_title('B: UK Food Retailer Analytics Maturity\n(Corporate Disclosures, FY 2024/25)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 4.5)
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels(['Descriptive', 'Diagnostic', 'Predictive', 'Prescriptive'], fontsize=9)
    ax2.invert_yaxis()
    ax2.spines[['top', 'right']].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor='#E8505B', label='Prescriptive'),
        mpatches.Patch(facecolor='#F9C74F', label='Predictive'),
        mpatches.Patch(facecolor='#14A76C', label='Diagnostic')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

    # =========================================================================
    # PANEL C: AI & Automation Adoption — Butterfly Chart
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    categories = ['Currently\nUsing', 'Planning to\nInvest',
                  'Waiting\nto See', 'Other/Not\nDisclosed']

    # AI/ML values from BearingPoint (2024)
    ai_values = [22, 17, 24, 37]

    # Automation values from Retail Economics (2025) - Large firms >£300m
    auto_values = [18.7, 46.7, 20.0, 14.6]

    y_pos = np.arange(len(categories))

    ax3.barh(y_pos, [-v for v in ai_values], height=0.6, color='#B0B0B0',
             edgecolor='white', linewidth=1, label='AI/ML Analytics')
    ax3.barh(y_pos, auto_values, height=0.6, color='#4D908E',
             edgecolor='white', linewidth=1, label='Automation')

    for i, (ai_val, auto_val) in enumerate(zip(ai_values, auto_values)):
        ax3.text(-ai_val - 2, i, f'{ai_val}%', ha='right', va='center',
                 fontsize=9, fontweight='bold', color='#2E86AB')
        ax3.text(auto_val + 2, i, f'{auto_val}%', ha='left', va='center',
                 fontsize=9, fontweight='bold', color='#E8505B')

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(categories, fontsize=9)
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.set_xlim(-55, 60)
    ax3.set_xticks([-40, -20, 0, 20, 40])
    ax3.set_xticklabels(['40%', '20%', '0', '20%', '40%'])
    ax3.set_xlabel('← AI/ML Analytics          Adoption Rate          Automation →',
                   fontsize=9, fontweight='bold')
    ax3.set_title('C: UK Retail Technology Adoption\n(AI Analytics & Automation)',
                  fontsize=11, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.spines[['top', 'right', 'left']].set_visible(False)
    ax3.invert_yaxis()

    # =========================================================================
    # MAIN TITLE & FOOTNOTES
    # =========================================================================
    plt.suptitle('Analytics Maturity Gap and Adoption Patterns in UK Food Retail',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        for fmt in ['png', 'pdf']:
            filepath = f'{output_path}.{fmt}'
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filepath}")

    return fig


def main():
    """Generate Figure 4.1."""
    print("\n" + "="*60)
    print("FIGURE 4.1: Analytics Maturity Dashboard")
    print("="*60)

    output_path = os.path.join(OUTPUT_DIR, 'figure_4_1_analytics_maturity_dashboard')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_figure(output_path)

    print("\n✓ Figure generation complete.\n")


if __name__ == "__main__":
    main()