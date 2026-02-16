"""
Figure 1.2: UK Food Price Inflation Analysis (2015-2025)

Two-panel analysis showing food CPI trends and cumulative price increases,
demonstrating the efficiency imperative driving digital transformation.

Data Source:
    ONS (2025) CPI Annual Rate: Food and Non-Alcoholic Beverages (Series D7G8)
    
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
    """Load ONS Food CPI data."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'series-160126.xls')
    
    df = pd.read_excel(filepath, sheet_name='data', skiprows=7)
    df.columns = ['Period', 'Food_CPI_Annual_Rate']
    df['Food_CPI_Annual_Rate'] = pd.to_numeric(df['Food_CPI_Annual_Rate'], errors='coerce')
    return df


def generate_figure(df, output_path=None):
    """Generate the food inflation analysis figure."""
    
    # Filter monthly data
    df_monthly = df[df['Period'].astype(str).str.match(r'^\d{4}\s[A-Z]{3}$', na=False)].copy()
    
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    
    def parse_period(p):
        parts = str(p).split()
        year = int(parts[0])
        month = month_map.get(parts[1], 1)
        return pd.Timestamp(year=year, month=month, day=1)
    
    df_monthly['Date'] = df_monthly['Period'].apply(parse_period)
    df_monthly = df_monthly.sort_values('Date')
    df_plot = df_monthly[df_monthly['Date'] >= '2015-01-01'].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Time series
    ax1.plot(df_plot['Date'], df_plot['Food_CPI_Annual_Rate'],
             color='#E8505B', linewidth=2)
    ax1.fill_between(df_plot['Date'], 0, df_plot['Food_CPI_Annual_Rate'],
                     where=(df_plot['Food_CPI_Annual_Rate'] > 0),
                     color='#E8505B', alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.axhline(y=2, color='#14A76C', linewidth=1.5, linestyle='--',
                label='BoE Target (2%)')
    ax1.axvspan(pd.Timestamp('2022-02-01'), pd.Timestamp('2023-12-01'),
                color='gray', alpha=0.1, label='Supply Chain Disruption')

    ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Annual Inflation Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('UK Food Inflation (2015–2025)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(-5, 22)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Cumulative increase
    novembers = df_monthly[df_monthly['Period'].str.contains('NOV', na=False)]
    novembers = novembers[novembers['Date'] >= '2020-11-01'].copy()

    index_values = [100]
    for i in range(1, len(novembers)):
        rate = novembers.iloc[i]['Food_CPI_Annual_Rate']
        index_values.append(index_values[-1] * (1 + rate/100))

    x_labels = ['Nov\n2020', 'Nov\n2021', 'Nov\n2022', 'Nov\n2023', 'Nov\n2024', 'Nov\n2025']
    # Handle case where we have fewer data points
    x_labels = x_labels[:len(index_values)]
    
    bars = ax2.bar(x_labels, index_values,
                   color=['#7A7A7A'] + ['#E8505B']*(len(index_values)-1),
                   edgecolor='black')

    for bar, val in zip(bars, index_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}',
                 ha='center', fontsize=10, fontweight='bold')

    ax2.axhline(y=100, color='black', linewidth=1)
    ax2.set_ylabel('Price Index (Nov 2020 = 100)', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Food Price Increase', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 160)
    ax2.grid(False, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('UK Food Price Inflation — Efficiency Imperative Driver',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.figtext(0.5, -0.02,
                'Source: ONS (2025) CPI Annual Rate: Food and Non-Alcoholic Beverages (Series D7G8)',
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    if output_path:
        for fmt in ['png', 'pdf']:
            filepath = f'{output_path}.{fmt}'
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filepath}")

    return fig


def main():
    """Generate Figure 1.2: UK Food Inflation Analysis."""
    print("\n" + "="*60)
    print("Figure 1.2: UK Food Inflation Analysis")
    print("="*60)

    try:
        df = load_data()
        output_path = os.path.join(OUTPUT_DIR, 'figure_1_2_food_inflation')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        generate_figure(df, output_path)
        print("\n✓ Figure generation complete.\n")
    except FileNotFoundError as e:
        print(f"\n✗ Data file not found: {e}")
        print("  Please download from: https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7g8/mm23")
        print("  Place in: ./data/series-160126.xls\n")


if __name__ == "__main__":
    main()
