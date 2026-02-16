"""
═══════════════════════════════════════════════════════════════════════════════
FIGURE 1.1: UK GROCERY MARKET STRUCTURAL TRANSFORMATION ANALYSIS (2015-2025)
═══════════════════════════════════════════════════════════════════════════════

Advanced multi-panel visualisation examining the structural transformation of
the UK grocery retail market, incorporating market concentration metrics,
competitive dynamics analysis, and temporal market share evolution.

Academic Context:
    Dissertation: "The Role of Big Data in Optimising Inventory Management 
                   in the UK Food Retail Industry"
    Programme:    BA (Hons) Global Business Management
    Institution:  University of Suffolk
    
Data Sources (Harvard References):
    Kantar (2015) Grocery Market Share Data: 12 weeks to 8 November 2015. 
        London: Kantar Worldpanel.
    Kantar (2025) Grocery Market Share Data: 12 weeks to 2 November 2025. 
        London: Kantar Worldpanel.
        Available at: https://www.kantar.com/uki/campaigns/grocery-market-share

Theoretical Framework:
    Market concentration analysis utilises the Herfindahl-Hirschman Index (HHI),
    a standard measure in industrial organisation economics (Rhoades, 1993).
    
Version: 2.7 (2015 Data Update)
Python: 3.10+
Dependencies: matplotlib, numpy, scipy
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Get the directory where this script is located (works when run from anywhere)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate to repository root (two levels up from src/chapter1/)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Set output and data directories relative to repo root
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs')
DATA_DIR = os.path.join(REPO_ROOT, 'data')


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class RetailerCategory(Enum):
    """Classification of UK grocery retailers by strategic positioning."""
    BIG_FOUR = "Big Four"
    DISCOUNTER = "Discounter"
    CONVENIENCE = "Convenience"
    PREMIUM = "Premium"


@dataclass(frozen=True)
class ColourPalette:
    """
    Accessible colour palette following WCAG 2.1 AA guidelines.
    Colours selected for deuteranopia/protanopia accessibility.
    """
    # Primary comparison colours
    YEAR_2015: str = "#1B4F72"      # Deep navy
    YEAR_2025: str = "#D35400"      # Burnt orange
    
    # Category colours
    BIG_FOUR: str = "#2471A3"       # Steel blue
    DISCOUNTER: str = "#C0392B"     # Crimson
    CONVENIENCE: str = "#1E8449"    # Forest green
    PREMIUM: str = "#7D3C98"        # Royal purple
    
    # Change indicators
    POSITIVE: str = "#1D8348"       # Dark green
    NEGATIVE: str = "#922B21"       # Dark red
    NEUTRAL: str = "#5D6D7E"        # Slate grey
    
    # UI elements
    GRID: str = "#E8E8E8"
    BORDER: str = "#BDC3C7"
    BACKGROUND: str = "#FDFEFE"
    ANNOTATION_BG: str = "#FEF9E7"
    TEXT_PRIMARY: str = "#1C2833"
    TEXT_SECONDARY: str = "#566573"


@dataclass
class ChartConfig:
    """Configuration parameters for chart generation."""
    figure_width: float = 16
    figure_height: float = 10
    dpi: int = 300
    font_family: str = 'Arial'
    title_fontsize: int = 14
    label_fontsize: int = 11
    tick_fontsize: int = 9
    annotation_fontsize: int = 9
    
    # Output formats
    output_formats: Tuple[str, ...] = ('png', 'pdf', 'svg')


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Retailer:
    """Represents a single retailer with temporal market share data."""
    name: str
    category: RetailerCategory
    share_2015: float
    share_2025: float
    
    @property
    def change_pp(self) -> float:
        """Percentage point change."""
        return self.share_2025 - self.share_2015
    
    @property
    def change_relative(self) -> float:
        """Relative percentage change."""
        return (self.change_pp / self.share_2015) * 100 if self.share_2015 > 0 else 0
    
    @property
    def cagr(self) -> float:
        """Compound Annual Growth Rate over 10-year period (2015-2025)."""
        if self.share_2015 <= 0:
            return 0
        return ((self.share_2025 / self.share_2015) ** (1/10) - 1) * 100


@dataclass
class MarketData:
    """
    Comprehensive UK grocery market share dataset with analytical capabilities.
    
    Data verified from Kantar Worldpanel official releases (Nov 2015, Nov 2025).
    """
    retailers: List[Retailer] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.retailers:
            self._initialise_data()
    
    def _initialise_data(self):
        """Populate with verified Kantar market share data."""
        # UPDATED: 2015 Data (Nov 8th) and 2025 Data (Nov 2nd)
        raw_data = [
            ("Tesco", RetailerCategory.BIG_FOUR, 27.9, 28.2),       # Updated 2015
            ("Sainsbury's", RetailerCategory.BIG_FOUR, 16.6, 15.7), # Updated 2015
            ("Asda", RetailerCategory.BIG_FOUR, 16.4, 11.6),        # Updated 2015
            ("Aldi", RetailerCategory.DISCOUNTER, 5.6, 10.6),       # Updated 2015
            ("Morrisons", RetailerCategory.BIG_FOUR, 10.8, 8.3),    # Updated 2015
            ("Lidl", RetailerCategory.DISCOUNTER, 4.4, 8.2),        # Updated 2015
            ("Co-op", RetailerCategory.CONVENIENCE, 6.3, 5.4),      # Updated 2015
            ("Waitrose", RetailerCategory.PREMIUM, 5.2, 4.4),       # Unchanged
        ]
        
        self.retailers = [
            Retailer(name, cat, s15, s25) 
            for name, cat, s15, s25 in raw_data
        ]
    
    @property
    def names(self) -> List[str]:
        return [r.name for r in self.retailers]
    
    @property
    def shares_2015(self) -> np.ndarray:
        return np.array([r.share_2015 for r in self.retailers])
    
    @property
    def shares_2025(self) -> np.ndarray:
        return np.array([r.share_2025 for r in self.retailers])
    
    @property
    def changes_pp(self) -> np.ndarray:
        return np.array([r.change_pp for r in self.retailers])
    
    def get_by_category(self, category: RetailerCategory) -> List[Retailer]:
        """Filter retailers by category."""
        return [r for r in self.retailers if r.category == category]
    
    def category_total(self, category: RetailerCategory, year: int) -> float:
        """Calculate total market share for a category in given year."""
        retailers = self.get_by_category(category)
        if year == 2015:
            return sum(r.share_2015 for r in retailers)
        return sum(r.share_2025 for r in retailers)
    
    def calculate_hhi(self, year: int) -> float:
        """Calculate Herfindahl-Hirschman Index."""
        shares = self.shares_2015 if year == 2015 else self.shares_2025
        return np.sum(shares ** 2)
    
    def calculate_cr4(self, year: int) -> float:
        """Calculate Four-Firm Concentration Ratio (CR4)."""
        shares = self.shares_2015 if year == 2015 else self.shares_2025
        return np.sum(np.sort(shares)[-4:])
    
    def calculate_entropy(self, year: int) -> float:
        """Calculate market entropy."""
        shares = self.shares_2015 if year == 2015 else self.shares_2025
        shares_normalised = shares / 100
        shares_positive = shares_normalised[shares_normalised > 0]
        return -np.sum(shares_positive * np.log(shares_positive))


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MarketAnalysisVisualisation:
    
    def __init__(self, data: MarketData, config: ChartConfig = None, 
                 colours: ColourPalette = None):
        self.data = data
        self.config = config or ChartConfig()
        self.colours = colours or ColourPalette()
        self.fig = None
        self.axes = {}
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        plt.rcParams.update({
            'font.family': self.config.font_family,
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': self.config.tick_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'axes.titlesize': self.config.label_fontsize,
            'xtick.labelsize': self.config.tick_fontsize,
            'ytick.labelsize': self.config.tick_fontsize,
            'legend.fontsize': self.config.tick_fontsize,
            'figure.titlesize': self.config.title_fontsize,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,
            'grid.alpha': 0.3,
            'grid.linestyle': '-',
        })
    
    def generate(self, output_path: Optional[str] = None) -> plt.Figure:
        self.fig = plt.figure(figsize=(self.config.figure_width, 
                                       self.config.figure_height),
                             facecolor=self.colours.BACKGROUND)
        
        gs = gridspec.GridSpec(2, 4, figure=self.fig,
                              height_ratios=[1.1, 0.9],
                              width_ratios=[1.5, 1, 1, 1],
                              hspace=0.35, wspace=0.4)
        
        self.axes['comparison'] = self.fig.add_subplot(gs[0, :2])
        self.axes['changes'] = self.fig.add_subplot(gs[0, 2:])
        self.axes['concentration'] = self.fig.add_subplot(gs[1, :2])
        self.axes['categories'] = self.fig.add_subplot(gs[1, 2:])
        
        self._plot_market_comparison()
        self._plot_change_waterfall()
        self._plot_concentration_analysis()
        self._plot_category_evolution()
        self._add_figure_elements()
        
        if output_path:
            self._save(output_path)
        return self.fig
    
    def _plot_market_comparison(self):
        """Panel (a): Grouped bar chart comparing 2015 vs 2025."""
        ax = self.axes['comparison']
        x = np.arange(len(self.data.names))
        width = 0.38
        
        bars_2015 = ax.bar(x - width/2, self.data.shares_2015, width,
                          label='2015', color=self.colours.YEAR_2015,
                          edgecolor='white', linewidth=0.8, zorder=3)
        bars_2025 = ax.bar(x + width/2, self.data.shares_2025, width,
                          label='2025', color=self.colours.YEAR_2025,
                          edgecolor='white', linewidth=0.8, zorder=3)
        
        self._add_bar_labels(ax, bars_2015, fontsize=8, offset=-0.05)
        self._add_bar_labels(ax, bars_2025, fontsize=8, offset=0.05)
        
        for i, change in enumerate(self.data.changes_pp):
            colour = self.colours.POSITIVE if change > 0 else self.colours.NEGATIVE
            symbol = '▲' if change > 0 else '▼'
            y_pos = max(self.data.shares_2015[i], self.data.shares_2025[i]) + 2
            ax.annotate(f'{symbol}\n{abs(change):.1f}', xy=(x[i] + width/2 + 0.05, y_pos),
                       fontsize=7, ha='center', va='bottom', color=colour,
                       fontweight='bold', linespacing=0.8)
    
        ax.set_xlabel('Retailer', fontweight='bold', labelpad=10)
        ax.set_ylabel('Market Share (%)', fontweight='bold', labelpad=10)
        ax.set_title('(a) Market Share Comparison by Retailer', 
                    fontweight='bold', loc='left', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(self.data.names, rotation=0)
        ax.set_ylim(0, 36)
        ax.set_yticks(np.arange(0, 40, 5))
        ax.legend(loc='center right').get_frame().set_linewidth(0)
        self._add_discounter_callout(ax)
    
    def _plot_change_waterfall(self):
        """Panel (b): Horizontal bar chart showing market share changes."""
        ax = self.axes['changes']
        sorted_indices = np.argsort(self.data.changes_pp)
        sorted_names = [self.data.names[i] for i in sorted_indices]
        sorted_changes = self.data.changes_pp[sorted_indices]
        sorted_retailers = [self.data.retailers[i] for i in sorted_indices]
        y = np.arange(len(sorted_names))
        
        colours = [self.colours.POSITIVE if c > 0 else self.colours.NEGATIVE 
                  for c in sorted_changes]
        
        bars = ax.barh(y, sorted_changes, height=0.7, color=colours,
                      edgecolor='white', linewidth=0.8, zorder=3)
        
        for i, (bar, change, retailer) in enumerate(zip(bars, sorted_changes, sorted_retailers)):
            width = bar.get_width()
            label_x = width + 0.2 if width >= 0 else width - 0.2
            ha = 'left' if width >= 0 else 'right'
            cagr = retailer.cagr
            label = f'{change:+.1f}pp'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, label,
                   va='center', ha=ha, fontsize=9, fontweight='bold',
                   color=self.colours.POSITIVE if change > 0 else self.colours.NEGATIVE)
        
        ax.axvline(x=0, color=self.colours.TEXT_PRIMARY, linewidth=1.2, zorder=2)
        ax.set_xlabel('Change in Market Share (Percentage Points)', 
                     fontweight='bold', labelpad=10)
        ax.set_title('(b) Market Share Change (2015 → 2025)', 
                    fontweight='bold', loc='left', pad=12)
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_names)
        
        # Extended range to prevent overlap
        ax.set_xlim(-7, 7)
        
        ax.xaxis.grid(False)
        self._add_cagr_annotation(ax, sorted_retailers, sorted_changes)
    
    def _plot_concentration_analysis(self):
        """Panel (c): Market concentration metrics."""
        ax = self.axes['concentration']
        hhi_2015 = self.data.calculate_hhi(2015)
        hhi_2025 = self.data.calculate_hhi(2025)
        cr4_2015 = self.data.calculate_cr4(2015)
        cr4_2025 = self.data.calculate_cr4(2025)
        entropy_2015 = self.data.calculate_entropy(2015)
        entropy_2025 = self.data.calculate_entropy(2025)
        
        metrics = ['HHI\n(÷100)', 'CR4\n(%)', 'Entropy\n(×10)']
        values_2015 = [hhi_2015/100, cr4_2015, entropy_2015*10]
        values_2025 = [hhi_2025/100, cr4_2025, entropy_2025*10]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars_2015 = ax.bar(x - width/2, values_2015, width, label='2015',
                          color=self.colours.YEAR_2015, edgecolor='white',
                          linewidth=0.8, zorder=3)
        bars_2025 = ax.bar(x + width/2, values_2025, width, label='2025',
                          color=self.colours.YEAR_2025, edgecolor='white',
                          linewidth=0.8, zorder=3)
        
        for bars in [bars_2015, bars_2025]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Index Value (Scaled)', fontweight='bold', labelpad=10)
        ax.set_title('(c) Market Concentration Metrics', 
                    fontweight='bold', loc='left', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 85)
        ax.yaxis.grid(False)
        ax.legend(loc='upper left').get_frame().set_linewidth(0)
        
        interpretation = (
            f"HHI Analysis:\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"2015: {hhi_2015:.0f} → 2025: {hhi_2025:.0f}\n"
            f"Change: {hhi_2025-hhi_2015:+.0f} ({(hhi_2025-hhi_2015)/hhi_2015*100:+.1f}%)\n\n"
            f"Interpretation:\n"
            f"Market trending toward\n"
            f"deconcentration"
        )
        props = dict(boxstyle='round,pad=0.4', facecolor=self.colours.ANNOTATION_BG,
                    edgecolor=self.colours.BORDER, linewidth=1, alpha=0.95)
        ax.text(0.98, 0.98, interpretation, transform=ax.transAxes, fontsize=8,
               va='top', ha='right', bbox=props, family='monospace', linespacing=1.3)
    
    def _plot_category_evolution(self):
        """Panel (d): Category-level market share evolution (2015-2025)."""
        ax = self.axes['categories']
        categories = [RetailerCategory.BIG_FOUR, RetailerCategory.DISCOUNTER,
                     RetailerCategory.CONVENIENCE, RetailerCategory.PREMIUM]
        
        cat_colours = {
            RetailerCategory.BIG_FOUR: self.colours.BIG_FOUR,
            RetailerCategory.DISCOUNTER: self.colours.DISCOUNTER,
            RetailerCategory.CONVENIENCE: self.colours.CONVENIENCE,
            RetailerCategory.PREMIUM: self.colours.PREMIUM,
        }
        
        years = ['2015', '2025']
        
        for i, cat in enumerate(categories):
            values = [self.data.category_total(cat, 2015),
                     self.data.category_total(cat, 2025)]
            
            ax.plot(years, values, 'o-', color=cat_colours[cat], 
                   linewidth=2.5, markersize=10, label=cat.value, zorder=3)
            
            for j, (year, val) in enumerate(zip(years, values)):
                dist = 6
                y_offset = 0
                if cat == RetailerCategory.PREMIUM:
                    y_offset = -8
                
                if j == 0:
                    xytext = (-dist, y_offset)
                    ha = 'right'
                else:
                    xytext = (dist, y_offset)
                    ha = 'left'
                    
                ax.annotate(f'{val:.1f}%', xy=(j, val), xytext=xytext,
                           textcoords='offset points', ha=ha, va='center',
                           fontsize=9, fontweight='bold', color=cat_colours[cat])
        
        for cat in categories:
            val_2015 = self.data.category_total(cat, 2015)
            val_2025 = self.data.category_total(cat, 2025)
            change = val_2025 - val_2015
            mid_y = (val_2015 + val_2025) / 2
            arrow = '↑' if change > 0 else '↓'
            colour = cat_colours[cat]
            
            if cat == RetailerCategory.PREMIUM:
                xytext_offset = (0, -5) 
                va_arrow = 'top'
            else:
                xytext_offset = (0, 5) 
                va_arrow = 'bottom'

            ax.annotate(f'{arrow}{abs(change):.1f}pp', xy=(0.5, mid_y),
                       xytext=xytext_offset, textcoords='offset points',
                       fontsize=8, ha='center', va=va_arrow, color=colour,
                       fontweight='bold', alpha=0.8)
        
        ax.set_ylabel('Combined Market Share (%)', fontweight='bold', labelpad=10)
        ax.set_title('(d) Category Evolution', fontweight='bold', loc='left', pad=12)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(0, 80)
        ax.yaxis.grid(False)
        
        # Legend moved to center left
        ax.legend(loc='center left').get_frame().set_linewidth(0)
        
        # Calculate dynamic insights
        bf_2015 = self.data.category_total(RetailerCategory.BIG_FOUR, 2015)
        bf_2025 = self.data.category_total(RetailerCategory.BIG_FOUR, 2025)
        disc_2015 = self.data.category_total(RetailerCategory.DISCOUNTER, 2015)
        disc_2025 = self.data.category_total(RetailerCategory.DISCOUNTER, 2025)
        
        # Estimate revenue shift: Market size £294bn approx (2024), 1% = £2.94bn
        rev_shift = abs(disc_2025 - disc_2015) * 2.94
        
        insight = (
            "Key Transformation:\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"Big Four: {bf_2015:.1f}% → {bf_2025:.1f}%\n"
            f"Discounters: {disc_2015:.1f}% → {disc_2025:.1f}%\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"~£{rev_shift:.1f}bn revenue shift"
        )
        
        props = dict(boxstyle='round,pad=0.4', facecolor=self.colours.ANNOTATION_BG,
                    edgecolor=self.colours.BORDER, linewidth=1, alpha=0.95)
        
        # Annotation moved to center right
        ax.text(0.98, 0.5, insight, transform=ax.transAxes, fontsize=8,
               va='center', ha='right', bbox=props, family='monospace', linespacing=1.3)
    
    def _add_bar_labels(self, ax, bars, fontsize=8, offset=0):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2 + offset, height + 0.4,
                   f'{height:.1f}', ha='center', va='bottom',
                   fontsize=fontsize, fontweight='bold', color=self.colours.TEXT_PRIMARY)
    
    def _add_discounter_callout(self, ax):
        disc_2015 = self.data.category_total(RetailerCategory.DISCOUNTER, 2015)
        disc_2025 = self.data.category_total(RetailerCategory.DISCOUNTER, 2025)
        growth = ((disc_2025 - disc_2015) / disc_2015) * 100
        
        text = (
            f"DISCOUNTER GROWTH\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"2015: {disc_2015:.1f}% combined\n"
            f"2025: {disc_2025:.1f}% combined\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Growth: +{growth:.0f}%\n"
            f"CAGR:   +{((disc_2025/disc_2015)**(1/10)-1)*100:.1f}%"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor=self.colours.ANNOTATION_BG,
                    edgecolor=self.colours.BORDER, linewidth=1, alpha=0.95)
        ax.text(0.98, 0.97, text, transform=ax.transAxes, fontsize=9,
               va='top', ha='right', bbox=props, family='monospace', linespacing=1.3)
    
    def _add_cagr_annotation(self, ax, retailers, changes):
        top_gainer = max(retailers, key=lambda r: r.cagr)
        top_loser = min(retailers, key=lambda r: r.cagr)
        text = (
            f"CAGR Analysis (10-yr):\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"▲ {top_gainer.name}: +{top_gainer.cagr:.1f}%\n"
            f"▼ {top_loser.name}: {top_loser.cagr:.1f}%"
        )
        props = dict(boxstyle='round,pad=0.4', facecolor=self.colours.ANNOTATION_BG,
                    edgecolor=self.colours.BORDER, linewidth=1, alpha=0.95)
        
        # Center-Right placement
        ax.text(0.98, 0.5, text, transform=ax.transAxes, fontsize=8,
               va='center', ha='right', bbox=props, family='monospace', linespacing=1.3)
    
    def _add_figure_elements(self):
        self.fig.suptitle(
            'Figure 1.1: UK Grocery Market Structural Transformation (2015–2025)',
            fontsize=self.config.title_fontsize, fontweight='bold', y=0.98
        )
        self.fig.text(0.5, 0.945, 
                     'Analysis of competitive dynamics, market concentration, and category evolution',
                     ha='center', fontsize=10, style='italic', color=self.colours.TEXT_SECONDARY)
        source = ('Source: Adapted from Kantar (2015; 2025) Grocery Market Share Data. '
                 'London: Kantar Worldpanel. Market concentration metrics calculated by author.')
        self.fig.text(0.5, 0.01, source, ha='center', fontsize=8,
                     style='italic', color=self.colours.TEXT_SECONDARY)
    
    def _save(self, base_path: str):
        for fmt in self.config.output_formats:
            filepath = f'{base_path}.{fmt}'
            self.fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                           facecolor=self.colours.BACKGROUND, edgecolor='none')
            print(f"  ✓ Saved: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_statistical_report(data: MarketData):
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " UK GROCERY MARKET STRUCTURAL TRANSFORMATION ANALYSIS ".center(78) + "║")
    print("║" + " 2015 → 2025 ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\n┌" + "─" * 78 + "┐")
    print("│" + " RETAILER PERFORMANCE METRICS ".center(78) + "│")
    print("├" + "─" * 78 + "┤")
    print("│ {:14s} │ {:>8s} │ {:>8s} │ {:>9s} │ {:>8s} │ {:14s} │".format(
        "Retailer", "2015 (%)", "2025 (%)", "Δ (pp)", "CAGR (%)", "Category"))
    print("├" + "─" * 78 + "┤")
    
    for r in data.retailers:
        arrow = "▲" if r.change_pp > 0 else "▼"
        print("│ {:14s} │ {:>8.1f} │ {:>8.1f} │ {:>1s} {:>6.1f} │ {:>+8.1f} │ {:14s} │".format(
            r.name, r.share_2015, r.share_2025, arrow, abs(r.change_pp), r.cagr, r.category.value))
    
    print("└" + "─" * 78 + "┘")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    market_data = MarketData()
    generate_statistical_report(market_data)
    
    print("\nGenerating publication-quality visualisation...")
    print("─" * 50)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    config = ChartConfig(
        figure_width=16,
        figure_height=10,
        dpi=300
    )
    viz = MarketAnalysisVisualisation(market_data, config)
    
    # Save to repository outputs folder
    output_path = os.path.join(OUTPUT_DIR, 'figure_1_1_market_transformation')
    fig = viz.generate(output_path=output_path)
    
    print("─" * 50)
    print(f"✓ Visualisation generation complete.")
    print(f"  Output directory: {OUTPUT_DIR}\n")
