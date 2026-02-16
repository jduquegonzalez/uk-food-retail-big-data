"""
Figure 4.2: Complexity-Efficiency Trade-Off

Scatter plot with retailer logos illustrating the counterintuitive finding that
discounters achieve superior availability despite lower analytics maturity.

MATURITY X-POSITIONS aligned with Table 4.1:
    x=1.5  Diagnostic (Transitioning) (Lidl)
    x=2.0  Diagnostic           (Waitrose)
    x=2.75 Diagnostic/Predictive (Morrisons)
    x=3.0  Predictive           (Aldi, Asda, Iceland, M&S)
    x=3.5  Predictive           (Sainsbury's)
    x=4.0  Prescriptive         (Tesco)
    x=4.5  Prescriptive         (Ocado)

    Visual jitter applied to the Predictive cluster (x=3.0) to prevent
    logo overlap.

Data Sources:
    Availability: The Grocer Week 28 (26 Jan 2026) Avg YTD
                  M&S data from Week 24 (08 Dec 2025) due to N/A in Week 28
    Analytics maturity: Table 4.1 (Corporate disclosures FY 2024/25)

Author: Jonathan Duque González
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs') if os.path.exists(os.path.join(REPO_ROOT, 'outputs')) else '/mnt/user-data/outputs'
LOGO_DIR = os.path.join(REPO_ROOT, 'data', 'logos') if os.path.exists(os.path.join(REPO_ROOT, 'data', 'logos')) else '/mnt/user-data/uploads'

# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

COLOURS = {
    'discounter': '#C0392B',
    'big_four': '#2471A3',
    'premium': '#7D3C98',
    'convenience': '#1E8449',
    'online': '#E67E22',
    'highlight_zone': '#F9E79F',
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOGO FILE MAP
# ═══════════════════════════════════════════════════════════════════════════════

LOGOS = {
    'Aldi': 'aldi.png',
    'Asda': 'asda.png',
    'Iceland': 'iceland.png',
    'Lidl': 'lidl.png',
    'M&S': 'm&s.png',
    'Morrisons': 'morrisons.png',
    'Ocado': 'ocado.png',
    "Sainsbury's": 'sainsburys.png',
    'Tesco': 'tesco.png',
    'Waitrose': 'waitrose.png',
}

# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD LOGO WIDTH (pixels) - All logos resized to this width
# Adjust these values to change logo display size
# ═══════════════════════════════════════════════════════════════════════════════

STANDARD_WIDTH = 120  # pixels for regular retailers
DISCOUNTER_WIDTH = 140  # slightly larger for emphasis on discounters

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_resize_logo(logo_path, target_width):
    """
    Load a logo and resize to standard width, maintaining aspect ratio.
    
    Args:
        logo_path: Path to the logo image file
        target_width: Desired width in pixels
    
    Returns:
        numpy array of the resized image (RGBA)
    """
    img = Image.open(logo_path)
    
    # Convert to RGBA for transparency support
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Calculate new height to maintain aspect ratio
    aspect_ratio = img.height / img.width
    new_height = int(target_width * aspect_ratio)
    
    # Resize using high-quality Lanczos resampling
    img_resized = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)


def plot_logo(ax, x, y, logo_array, zoom=1.0, zorder=4):
    """
    Plot a pre-processed logo array at specified coordinates.
    
    Args:
        ax: Matplotlib axes object
        x, y: Plot coordinates
        logo_array: NumPy array of the logo image (from load_and_resize_logo)
        zoom: Additional zoom factor (1.0 = use standardised size as-is)
        zorder: Drawing order (higher = on top)
    """
    imagebox = OffsetImage(logo_array, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        (x, y),
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=zorder
    )
    ax.add_artist(ab)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_figure(output_path=None):

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ═══════════════════════════════════════════════════════════════════════════
    # DATA
    # ═══════════════════════════════════════════════════════════════════════════

    data = {
        #                  (maturity_x,  availability%, category)
        'Ocado':       (4.5,  97.0, 'online'),         # Prescriptive
        'Tesco':       (4.0,  96.4, 'big_four'),       # Prescriptive
        "Sainsbury's": (3.5,  95.6, 'big_four'),       # Predictive
        'M&S':         (3.25, 100.0, 'premium'),       # Predictive (Offset slightly right for jitter)
        'Aldi':        (3.0,  97.7, 'discounter'),     # Predictive (High availability)
        'Iceland':     (2.9,  89.3, 'convenience'),    # Predictive (Low availability, offset left)
        'Asda':        (3.1,  93.6, 'big_four'),       # Predictive (Mid availability, offset right)
        'Morrisons':   (2.75, 94.0, 'big_four'),       # Diagnostic/Predictive
        'Waitrose':    (2.0,  95.9, 'premium'),        # Diagnostic
        'Lidl':        (1.5,  97.9, 'discounter'),     # Diagnostic (Transitioning)
    }
    
    retailers = list(data.keys())
    # Extract x and y for plotting, using the updated dictionary values
    analytics = [data[r][0] for r in retailers] 
    availability = [data[r][1] for r in retailers]
    categories = [data[r][2] for r in retailers]

    # ═══════════════════════════════════════════════════════════════════════════
    # PRE-LOAD AND STANDARDISE ALL LOGOS
    # ═══════════════════════════════════════════════════════════════════════════

    logo_arrays = {}
    for retailer in retailers:
        # Construct path (assumes logo filename matches dict key + .png or similar)
        # Using LOGOS map defined globally
        if retailer in LOGOS:
             logo_path = os.path.join(LOGO_DIR, LOGOS[retailer])
        else:
             print(f"Warning: No logo mapping for {retailer}")
             logo_arrays[retailer] = None
             continue

        category = data[retailer][2]
        
        # Use larger width for discounters (for emphasis)
        target_width = DISCOUNTER_WIDTH if category == 'discounter' else STANDARD_WIDTH
        
        try:
            if os.path.exists(logo_path):
                logo_arrays[retailer] = load_and_resize_logo(logo_path, target_width)
                print(f"  ✓ Loaded: {retailer} → {logo_arrays[retailer].shape[1]}×{logo_arrays[retailer].shape[0]}px")
            else:
                 print(f"  ✗ File not found: {logo_path}")
                 logo_arrays[retailer] = None
        except Exception as e:
            print(f"  ✗ Failed to load {retailer}: {e}")
            logo_arrays[retailer] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # HIGHLIGHT ZONE
    # ═══════════════════════════════════════════════════════════════════════════

    # Adjust highlight zone to cover the high-availability discounters/value retailers
    # Spanning from Lidl (1.5) to Aldi (3.0) at high availability
    ax.axhspan(96.5, 99, xmin=0.08, xmax=0.58, # Covers roughly x=1.5 to x=3.5 in plot coordinates
               alpha=0.15,
               color=COLOURS['highlight_zone'],
               zorder=1)

    ax.annotate(
        'DISCOUNTER\nEXCELLENCE\nZONE',
        xy=(2.25, 97.75), # Centered roughly between Lidl and Aldi
        fontsize=8,
        ha='center',
        va='center',
        color='#B7950B',
        fontweight='bold',
        alpha=0.7,
        style='italic'
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGO-BASED SCATTER (with standardised sizes)
    # ═══════════════════════════════════════════════════════════════════════════

    for i, retailer in enumerate(retailers):
        x = analytics[i]
        y = availability[i]
        category = categories[i]

        if logo_arrays.get(retailer) is not None:
            # Use consistent zoom since all logos are now pre-standardised to same width
            plot_logo(ax, x, y, logo_arrays[retailer], zoom=0.5)
        else:
            # Fallback: coloured marker if logo failed to load
            marker = 'D' if category == 'discounter' else 'o'
            ax.scatter(x, y, s=150, c=COLOURS[category], marker=marker,
                      edgecolors='white', linewidth=1.5, zorder=4)

    # ═══════════════════════════════════════════════════════════════════════════
    # AXES FORMATTING
    # ═══════════════════════════════════════════════════════════════════════════

    ax.set_xlabel('Analytics Maturity Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Shelf Availability (%)', fontsize=11, fontweight='bold')

    # Extend x-axis slightly to accommodate Ocado at 4.5
    ax.set_xlim(1, 4.8)
    ax.set_ylim(88, 102)

    ax.set_xticks([1.5, 2.5, 3.5, 4.5])
    ax.set_xticklabels(
        ['Descriptive', 'Diagnostic', 'Predictive', 'Prescriptive'],
        fontsize=9
    )

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)

    # ═══════════════════════════════════════════════════════════════════════════
    # TITLE & FOOTNOTES
    # ═══════════════════════════════════════════════════════════════════════════

    fig.suptitle(
        'Complexity-Efficiency Trade-Off',
        fontsize=13,
        fontweight='bold',
        y=0.95
   )

    fig.text(
        0.5, 0.90,
        'Demand consolidation enables superior availability across maturity levels',
        fontsize=10,
        style='italic',
        color='#C0392B',
        ha='center',
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════════

    if output_path:
        for fmt in ['png', 'pdf']:
            filepath = f'{output_path}.{fmt}'
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filepath}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("FIGURE 4.2: A complexity-efficiency trade-off")
    print("="*60)
    print(f"\nLogo directory: {LOGO_DIR}")
    print(f"Standard width: {STANDARD_WIDTH}px (discounters: {DISCOUNTER_WIDTH}px)")
    print("\nLoading and resizing logos...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'figure_4_2_complexity_efficiency_trade_off')
    generate_figure(output_path)
    
    print("\n✓ Figure 4.2 generated successfully.\n")


if __name__ == "__main__":
    main()