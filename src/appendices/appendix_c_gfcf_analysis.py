"""
Appendix C: G7 Gross Fixed Capital Formation Analysis (1995-2023)

Multi-panel analysis demonstrating UK structural underinvestment compared to
other G7 nations - a key environmental barrier to digital transformation.

Data Source:
    World Bank – World Development Indicators (2024)
    Indicator: NE.GDI.FTOT.ZS (GFCF as % of GDP)
    
Author: Jonathan Duque González
Version: 1.0
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate to repository root (two levels up from src/appendices/)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# Set output and data directories relative to repo root
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs')
DATA_DIR = os.path.join(REPO_ROOT, 'data')


def load_data(filepath=None):
    """Load World Bank GFCF data."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'API_NE.GDI.FTOT.ZS_DS2_en_excel_v2_174226.xls')
    
    df = pd.read_excel(filepath, sheet_name='Data', skiprows=3)
    return df


def generate_figure(df, output_path=None):
    """Generate the comprehensive GFCF analysis figure."""
    
    g7_countries = ['Canada', 'France', 'Germany', 'Italy', 'Japan',
                    'United Kingdom', 'United States']
    
    g7_data = df[df['Country Name'].isin(g7_countries)].copy()
    
    year_columns = [col for col in df.columns
                    if isinstance(col, (int, str)) and str(col).isdigit()
                    and 1995 <= int(col) <= 2023]
    
    g7_transposed = g7_data.set_index('Country Name')[year_columns].T
    g7_transposed.index = g7_transposed.index.astype(int)
    
    other_g7 = [c for c in g7_transposed.columns if c != 'United Kingdom']

    # Colour scheme
    colors = {
        'United Kingdom': '#E8505B',
        'United States': '#2E86AB',
        'Japan': '#7A7A7A',
        'Germany': '#F9C74F',
        'France': '#14A76C',
        'Italy': '#6B4C9A',
        'Canada': '#F18F01'
    }

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Panel 1: Main time series
    ax1 = fig.add_subplot(gs[0:2, :])

    for country in g7_transposed.columns:
        ax1.plot(g7_transposed.index, g7_transposed[country],
                 label=country, color=colors.get(country, '#7A7A7A'),
                 linewidth=2.5 if country == 'United Kingdom' else 1.5,
                 alpha=1.0 if country == 'United Kingdom' else 0.8,
                 zorder=10 if country == 'United Kingdom' else 2)

    # Event markers
    ax1.axvline(x=2008, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=2016, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.annotate('Financial\nCrisis', xy=(2008, 29), fontsize=8, ha='center', color='gray')
    ax1.annotate('Brexit\nVote', xy=(2016, 29), fontsize=8, ha='center', color='gray')

    ax1.set_ylabel('GFCF (% of GDP)', fontsize=11, fontweight='semibold')
    ax1.set_xlabel('Year', fontsize=11, fontweight='semibold')
    ax1.set_title('G7 Countries: Gross Fixed Capital Formation (1995–2023)',
                  fontsize=13, fontweight='semibold', pad=10)
    ax1.legend(loc='lower right', framealpha=0.9, ncol=2, fontsize=9)
    ax1.set_ylim(14, 31)
    ax1.set_xlim(1995, 2023)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel 2: UK Gap over time
    ax2 = fig.add_subplot(gs[2, 0])
    other_g7_avg = g7_transposed[other_g7].mean(axis=1)
    gap = other_g7_avg - g7_transposed['United Kingdom']

    ax2.fill_between(gap.index, 0, gap.values, alpha=0.4, color='#E8505B')
    ax2.plot(gap.index, gap.values, color='#C0392B', linewidth=0.5)
    ax2.axhline(y=gap.mean(), color='black', linestyle='--',
                linewidth=1, label=f'Average: {gap.mean():.2f}pp')

    max_gap = gap.max()
    max_year = gap.idxmax()
    ax2.annotate(f'Peak: {max_gap:.1f}pp\n({max_year})', xy=(max_year, max_gap),
                 xytext=(max_year+3, max_gap+0.3), fontsize=9, color='#C0392B',
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1))

    ax2.set_xlabel('Year', fontsize=10, fontweight='semibold')
    ax2.set_ylabel('Gap (percentage points)', fontsize=10, fontweight='semibold')
    ax2.set_title('UK Investment Gap vs Other G7 Average', fontsize=11, fontweight='semibold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_ylim(0, 7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel 3: Period averages
    ax3 = fig.add_subplot(gs[2, 1])

    periods = {
        '1995–2000': (1995, 2000),
        '2001–2008': (2001, 2008),
        '2009–2015': (2009, 2015),
        '2016–2023': (2016, 2023)
    }

    period_data = []
    for period_name, (start, end) in periods.items():
        period_df = g7_transposed[(g7_transposed.index >= start) &
                                   (g7_transposed.index <= end)]
        uk_avg = period_df['United Kingdom'].mean()
        other_avg = period_df[other_g7].mean(axis=1).mean()
        period_data.append({
            'Period': period_name, 'UK': uk_avg,
            'Other G7': other_avg, 'Gap': other_avg - uk_avg
        })

    period_df_plot = pd.DataFrame(period_data)
    x = np.arange(len(period_df_plot))
    width = 0.35

    bars1 = ax3.bar(x - width/2, period_df_plot['UK'], width,
                    label='United Kingdom', color='#E8505B')
    bars2 = ax3.bar(x + width/2, period_df_plot['Other G7'], width,
                    label='Other G7 Average', color='#2E86AB')

    ax3.set_ylabel('GFCF (% of GDP)', fontsize=10, fontweight='semibold')
    ax3.set_title('Investment by Period', fontsize=11, fontweight='semibold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(period_df_plot['Period'], fontsize=9)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_ylim(0, 28)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Value labels
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 1.5,
                f'{bar.get_height():.1f}%', ha='center', va='top',
                fontsize=8, color='white', fontweight='light')
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 1.5,
                f'{bar.get_height():.1f}%', ha='center', va='top',
                fontsize=8, color='white', fontweight='light')

    # Gap annotations
    for i, gap_val in enumerate(period_df_plot['Gap']):
        ax3.annotate(f'{gap_val:.1f}pp\ngap',
                     xy=(i, period_df_plot['Other G7'].iloc[i] + 0.5),
                     fontsize=9, ha='center', fontweight='semibold', color='#E8505B')

    plt.suptitle('UK Investment Underperformance — A Structural Problem',
                 fontsize=14, fontweight='semibold', y=0.995)

    plt.figtext(0.5, 0.01,
                'Source: World Bank – World Development Indicators (2024). '
                'GFCF includes investment in fixed assets by businesses, governments, and households.',
                ha='center', fontsize=9, style='italic')

    if output_path:
        for fmt in ['png', 'pdf']:
            filepath = f'{output_path}.{fmt}'
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filepath}")

    return fig


def main():
    """Generate Appendix C figure."""
    print("\n" + "="*60)
    print("APPENDIX C: G7 GFCF Analysis")
    print("="*60)

    try:
        df = load_data()
        output_path = os.path.join(OUTPUT_DIR, 'appendix_c_gfcf_analysis')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        generate_figure(df, output_path)
        print("\n✓ Figure generation complete.\n")
    except FileNotFoundError as e:
        print(f"\n✗ Data file not found: {e}")
        print("  Please download from: https://data.worldbank.org/indicator/NE.GDI.FTOT.ZS")
        print("  Place in: ./data/API_NE.GDI.FTOT.ZS_DS2_en_excel_v2_174226.xls\n")


if __name__ == "__main__":
    main()
