"""
Appendix B.3: N-gram Frequency Analysis

Linguistic pattern analysis of extraction corpus to validate thematic saturation
and keyword convergence with the theoretical framework.

Data Source:
    Secondary Data Extraction Matrix v2.3

Author: Jonathan Duque González
Version: 2.0
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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


# Extended stopwords for retail/business context
STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who',
    'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'our', 'their', 'your',
    'his', 'her', 'my', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'any',
    'about', 'over', 'up', 'down', 'out', 'off', 'if', 'while', 'because', 'until',
    'being', 'having', 'doing', 'per', 'cent', 'including', 'across', 'within', 'using',
    'based', 'new', 'year', 'years', 'including', 'well', 'use', 'used', 'make', 'made',
    'one', 'two', 'three', 'first', 'second', 'last', 'many', 'much', 'get', 'got'
])


def load_data(filepath=None):
    """Load extraction matrix data."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'Secondary_Data_Extraction_Matrix_v2.3.xlsx')
    
    df = pd.read_excel(filepath, sheet_name='Extraction_Log')
    df_inc = df[df['Reviewer_Decision'] == 'Include'].copy()
    return df_inc


def preprocess_text(text):
    """Clean text for n-gram analysis."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_ngrams(text, n, top_k=20):
    """Extract n-grams from text."""
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=list(STOPWORDS), min_df=1)
    ngram_matrix = vectorizer.fit_transform([text])
    ngram_counts = dict(zip(vectorizer.get_feature_names_out(), ngram_matrix.toarray()[0]))
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_ngrams


def generate_figure(unigrams, bigrams, trigrams, n_extracts, output_path=None):
    """Generate the three-panel N-gram analysis figure."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    colors = {
        'unigram': '#2E86AB',
        'bigram': '#A23B72',
        'trigram': '#14A76C'
    }

    # Panel A: Unigrams
    ax1 = axes[0]
    words_1 = [item[0] for item in unigrams][::-1]
    counts_1 = [item[1] for item in unigrams][::-1]

    bars1 = ax1.barh(words_1, counts_1, color=colors['unigram'],
                     edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Panel A: Top Unigrams\n(Single Words)', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', alpha=0.3)

    for bar, count in zip(bars1, counts_1):
        ax1.text(count + 2, bar.get_y() + bar.get_height()/2, str(count),
                 va='center', fontsize=9, fontweight='bold')

    # Panel B: Bigrams
    ax2 = axes[1]
    words_2 = [item[0] for item in bigrams][::-1]
    counts_2 = [item[1] for item in bigrams][::-1]

    bars2 = ax2.barh(words_2, counts_2, color=colors['bigram'],
                     edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Panel B: Top Bigrams\n(Two-Word Phrases)', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3)

    for bar, count in zip(bars2, counts_2):
        ax2.text(count + 1, bar.get_y() + bar.get_height()/2, str(count),
                 va='center', fontsize=9, fontweight='bold')

    # Panel C: Trigrams
    ax3 = axes[2]
    words_3 = [item[0] for item in trigrams][::-1]
    counts_3 = [item[1] for item in trigrams][::-1]

    bars3 = ax3.barh(words_3, counts_3, color=colors['trigram'],
                     edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Panel C: Top Trigrams\n(Three-Word Phrases)', fontsize=12, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='x', alpha=0.3)

    for bar, count in zip(bars3, counts_3):
        ax3.text(count + 0.5, bar.get_y() + bar.get_height()/2, str(count),
                 va='center', fontsize=9, fontweight='bold')

    plt.suptitle(f'Appendix B.3: N-gram Frequency Analysis of Extraction Corpus (n={n_extracts} extracts)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.figtext(0.5, -0.02,
                'Source: Secondary Data Extraction Matrix v2.3. Stopwords removed. '
                'Analysis validates thematic saturation and keyword convergence.',
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    if output_path:
        for fmt in ['png', 'pdf']:
            filepath = f'{output_path}.{fmt}'
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filepath}")

    return fig


def main():
    """Generate N-gram analysis and figure."""
    print("\n" + "="*60)
    print("APPENDIX B.3: N-gram Frequency Analysis")
    print("="*60)

    try:
        df_inc = load_data()
        texts = df_inc['Verbatim_Extract'].dropna().tolist()
        corpus = ' '.join(texts)
        corpus_clean = preprocess_text(corpus)

        print(f"\nCorpus: {len(corpus_clean.split())} words from {len(texts)} extracts")

        # Extract n-grams
        unigrams = get_ngrams(corpus_clean, 1, 15)
        bigrams = get_ngrams(corpus_clean, 2, 15)
        trigrams = get_ngrams(corpus_clean, 3, 12)

        # Print summary
        print("\nTop 10 Unigrams:")
        for word, count in unigrams[:10]:
            print(f"  {word}: {count}")

        print("\nTop 10 Bigrams:")
        for phrase, count in bigrams[:10]:
            print(f"  {phrase}: {count}")

        print("\nTop 10 Trigrams:")
        for phrase, count in trigrams[:10]:
            print(f"  {phrase}: {count}")

        # Save to Excel
        ngram_output = os.path.join(OUTPUT_DIR, 'Ngram_Analysis.xlsx')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with pd.ExcelWriter(ngram_output) as writer:
            pd.DataFrame(unigrams, columns=['Unigram', 'Frequency']).to_excel(
                writer, sheet_name='Unigrams', index=False)
            pd.DataFrame(bigrams, columns=['Bigram', 'Frequency']).to_excel(
                writer, sheet_name='Bigrams', index=False)
            pd.DataFrame(trigrams, columns=['Trigram', 'Frequency']).to_excel(
                writer, sheet_name='Trigrams', index=False)
        print(f"\n✓ N-gram data saved to: {ngram_output}")

        # Generate figure
        output_path = os.path.join(OUTPUT_DIR, 'appendix_b3_ngram_analysis')
        generate_figure(unigrams, bigrams, trigrams, len(texts), output_path)

        print("\n✓ Analysis complete.\n")

    except FileNotFoundError as e:
        print(f"\n✗ Data file not found: {e}")
        print("  Please download the extraction matrix from the Google Sheets link")
        print("  Place in: ./data/Secondary_Data_Extraction_Matrix_v2.3.xlsx\n")


if __name__ == "__main__":
    main()
