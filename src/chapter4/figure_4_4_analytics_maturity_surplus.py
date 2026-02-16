"""
Figure 4.4: Analytics Maturity vs Shelf Availability and Surplus Performance
            + Adjusted Operating Profit Margin

Panel A: Bubble chart with retailer logos (original style preserved)
Panel B: Bar chart of adjusted operating profit margins (all 10 retailers)

Bubble size = Surplus % Handled (Directly Proportional)
Retailers represented by logos embedded inside bubbles
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) if os.path.exists(os.path.join(SCRIPT_DIR, '..', '..', 'data')) else SCRIPT_DIR
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs') if os.path.exists(os.path.join(REPO_ROOT, 'outputs')) else '/mnt/user-data/outputs'
DATA_DIR = os.path.join(REPO_ROOT, 'data') if os.path.exists(os.path.join(REPO_ROOT, 'data')) else '/mnt/user-data/uploads'

MATRIX_FILE = os.path.join(DATA_DIR, 'UK_Food_Waste_Matrix.xlsx')
LOGO_DIR = os.path.join(REPO_ROOT, 'data', 'logos') if os.path.exists(os.path.join(REPO_ROOT, 'data', 'logos')) else '/mnt/user-data/logos'

# ═══════════════════════════════════════════════════════════════════════
# COLOUR MAPPING (original)
# ═══════════════════════════════════════════════════════════════════════

etailer_map = {
    "Sainsbury's": "#EC8A00",  # Orange
    "Morrisons":   "#00563F",  # Green
    "Iceland":     "#D2212E",  # Red
    "Tesco":       "#00539F",  # Blue
    "Lidl GB":     "#FFF200",  # Yellow
    "Aldi UK":     "#00B4DC",  # Cyan
}

# Extended map for Panel B (includes retailers not in bubble chart)
extended_color_map = {
    "Sainsbury's": "#EC8A00",
    "Morrisons":   "#00563F",
    "Iceland":     "#D2212E",
    "Tesco":       "#00539F",
    "Lidl":        "#FFF200",
    "Aldi":        "#00B4DC",
    "Asda":        "#78BE20",
    "Waitrose":    "#5C8A3C",
    "M&S (Food)":  "#000000",
    "Ocado":       "#6B2D8B",
}

def get_retailer_color(short_name):
    if short_name in etailer_map:
        return etailer_map[short_name]
    mapping = {'Aldi': 'Aldi UK', 'Lidl': 'Lidl GB'}
    key = mapping.get(short_name)
    return etailer_map.get(key, '#808080')

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING (original)
# ═══════════════════════════════════════════════════════════════════════

def load_surplus_data(matrix_path):
    try:
        df = pd.read_excel(matrix_path, sheet_name='Analysis_Ready', header=2)
        df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]

        surplus_data = {}
        for _, row in df.iterrows():
            retailer = row.get('Retailer', row.iloc[0])
            surplus_pct = row.get('Surplus % Handled', row.iloc[3])
            if pd.notna(retailer) and pd.notna(surplus_pct):
                surplus_data[str(retailer).strip()] = float(surplus_pct)
        return surplus_data
    except Exception as e:
        print(f"⚠ Using fallback data: {e}")
        return {
            'Aldi UK': 0.47, 'Lidl GB': 1.81, 'Iceland': 0.78,
            'Morrisons': 0.44, "Sainsbury's": 0.85, 'Tesco': 1.01,
        }

# ═══════════════════════════════════════════════════════════════════════
# PANEL A DATA (Updated coordinates)
# ═══════════════════════════════════════════════════════════════════════

retailers = {
    'Tesco':       (4.0, 96.4),  # Prescriptive
    "Sainsbury's": (3.5, 95.6),  # Predictive
    'Morrisons':   (2.55, 94.0), # Diagnostic/Predictive
    'Iceland':     (3.0, 89.3),  # Predictive
    'Lidl':        (1.5, 97.9),  # Diagnostic (Transitioning)
    'Aldi':        (3.0, 97.7),  # Predictive
}

name_map = {'Aldi UK': 'Aldi', 'Lidl GB': 'Lidl'}

# ═══════════════════════════════════════════════════════════════════════
# PANEL B DATA — Operating margins (FY 2024/25)
# Maturity x-positions consistent with Table 4.1
# ═══════════════════════════════════════════════════════════════════════

margins_data = {
    'Lidl':        (1.5,  2.68), # Diagnostic (Transitioning)
    'Waitrose':    (2.0,  3.03), # Diagnostic
    'Morrisons':   (2.55, 1.34), # Diagnostic/Predictive
    'Aldi':        (3.0,  2.40), # Predictive
    'Asda':        (3.0,  3.09), # Predictive
    'Iceland':     (3.0,  2.00), # Predictive
    'M&S (Food)':  (3.0,  5.37), # Predictive
    "Sainsbury's": (3.5,  3.28), # Predictive
    'Tesco':       (4.0,  5.08), # Prescriptive
    'Ocado':       (4.5, -0.36), # Prescriptive
}

# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def generate_figure_4_4():
    surplus_raw = load_surplus_data(MATRIX_FILE)
    surplus = {name_map.get(k, k): v for k, v in surplus_raw.items()}

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(9, 9.5),
        gridspec_kw={'height_ratios': [3, 1.6], 'hspace': 0.06},
        sharex=True
    )

    # ═══════════════════════════════════════════════════════════════════
    # PANEL A: BUBBLE CHART WITH LOGOS (original style)
    # ═══════════════════════════════════════════════════════════════════

    SCALING_FACTOR = 1600

    for r, (x, y) in retailers.items():
        if r not in surplus:
            continue

        val = surplus[r]
        s_val = max(val * SCALING_FACTOR, 100)
        color_hex = get_retailer_color(r)

        # Plot Bubble (original style: alpha=0.2, linewidth=0)
        ax1.scatter(
            x, y,
            s=s_val,
            facecolor=color_hex,
            linewidth=0,
            alpha=0.2,
            zorder=2
        )

        # ── LOGO HANDLING (original) ──
        logo_filename = f"{r}.png"
        logo_path = os.path.join(LOGO_DIR, logo_filename)

        if not os.path.exists(logo_path) and "'" in r:
            safe_name = r.replace("'", "")
            logo_path = os.path.join(LOGO_DIR, f"{safe_name}.png")

        if os.path.exists(logo_path):
            image = plt.imread(logo_path)
            h, w = image.shape[:2]

            # Dynamic Zoom (original)
            dynamic_zoom = (np.sqrt(s_val) / max(h, w)) * 0.55

            imagebox = OffsetImage(image, zoom=dynamic_zoom)
            ab = AnnotationBbox(
                imagebox, (x, y),
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=3
            )
            ax1.add_artist(ab)
        else:
            # Fallback: text label if logo not found
            ax1.annotate(
                f'{r}\n({val:.2f}%)',
                (x, y),
                fontsize=7.5, fontweight='bold', color=color_hex,
                ha='center', va='center', zorder=4
            )

    # Panel A formatting (original style)
    ax1.set_xlim(1.2, 4.8) # Extended slightly for Ocado
    ax1.set_ylim(88, 100)
    ax1.set_ylabel('Shelf Availability (%)', fontsize=10, fontweight='bold')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.yaxis.grid(False)
    ax1.tick_params(axis='x', which='both', length=0, labelbottom=False)

    # Panel A label
    ax1.text(0.015, 0.97, 'A', transform=ax1.transAxes,
             fontsize=13, fontweight='bold', va='top')

    # ═══════════════════════════════════════════════════════════════════
    # PANEL B: OPERATING MARGIN BARS
    # ═══════════════════════════════════════════════════════════════════

    BAR_WIDTH = 0.15
    GAP = 0.025

    # Group by maturity x-position, sort within group by margin desc
    groups = defaultdict(list)
    for name, (mat_x, margin) in margins_data.items():
        groups[mat_x].append((name, margin))
    
    # Sort keys to ensure bars are drawn left-to-right
    sorted_mat_x = sorted(groups.keys())

    for mat_x in sorted_mat_x:
        members = groups[mat_x]
        members.sort(key=lambda t: -t[1]) # Sort descending by margin
        
        n = len(members)
        total_w = n * BAR_WIDTH + (n - 1) * GAP
        start_x = mat_x - total_w / 2 + BAR_WIDTH / 2

        for i, (name, margin) in enumerate(members):
            bx = start_x + i * (BAR_WIDTH + GAP)
            color = extended_color_map.get(name, '#808080')
            bar_col = color if margin >= 0 else '#D32F2F'

            ax2.bar(bx, margin, width=BAR_WIDTH,
                    color=bar_col, alpha=0.78, edgecolor='white',
                    linewidth=0.5, zorder=2)

            # Value label above/below bar
            if margin >= 0:
                ax2.text(bx, margin + 0.18, f'{margin:.1f}%',
                         ha='center', va='bottom', fontsize=6,
                         fontweight='bold', color=color)
            else:
                ax2.text(bx, margin - 0.18, f'{margin:.1f}%',
                         ha='center', va='top', fontsize=6,
                         fontweight='bold', color='#D32F2F')

            # Name label below baseline
            ax2.text(bx, -1.6, name,
                     ha='center', va='top', fontsize=5.8, rotation=50,
                     color='#555555')

    ax2.set_ylim(-3.5, 7.0)
    ax2.set_ylabel('Adj. Operating\nMargin (%)', fontsize=9, fontweight='bold')
    ax2.axhline(0, color='#333333', linewidth=0.7, zorder=1)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.yaxis.grid(False)

    # Shared x-axis (original ticks)
    ax2.set_xlim(1.2, 4.8)
    ax2.set_xticks([1.5, 2.5, 3.5, 4.5])
    ax2.set_xticklabels(['Descriptive', 'Diagnostic', 'Predictive', 'Prescriptive'],
                         fontsize=9)
    ax2.set_xlabel('Analytics Maturity Level', fontsize=10, fontweight='bold',
                   labelpad=10)
    ax2.tick_params(axis='x', pad=28)

    # Panel B label
    ax2.text(0.015, 0.95, 'B', transform=ax2.transAxes,
             fontsize=13, fontweight='bold', va='top')

    # ═══════════════════════════════════════════════════════════════════
    # TITLE AND SOURCE
    # ═══════════════════════════════════════════════════════════════════

    fig.suptitle(
        'Analytics Maturity vs Shelf Availability, Surplus Performance\n'
        'and Adjusted Operating Profit Margin',
        fontsize=12, fontweight='bold', y=0.98
    )

    '''
    # Source (uncomment to include)
    fig.text(0.5, 0.01,
             'Sources: Analytics maturity and surplus from corporate disclosures (FY 2024/25); '
             'Availability from The Grocer Week 28 (Jan 2026) Avg YTD;\n'
             'Operating margins compiled from annual reports (FY 2024/25). '
             'Note: M&S availability from Week 24 (Dec 2025) due to N/A in Week 28.',
             ha='center', fontsize=7, style='italic', color='gray')
    '''

    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.13, hspace=0.06)

    output_path = os.path.join(OUTPUT_DIR, 'figure_4_4_revised.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure 4.4 saved to {output_path}")
    return fig


if __name__ == "__main__":
    generate_figure_4_4()