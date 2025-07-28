# analysis.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
import warnings
import logging
import argparse

# For progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# For lemmatization
try:
    import spacy
    nlp = spacy.blank('en')
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# For NLTK stopwords
import nltk
nltk.download('stopwords', quiet=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Configurable parameters
def get_config():
    return {
        'chunk_size': 100000,
        'search_volume_col': 'search_volume',
        'competition_col': 'competition',
        'keyword_col': 'keyword',
        'language_col': 'language',  # If not present, will be ignored
        'top_n_keywords': 20,
        'max_clusters': 10,
        'random_state': 42,
        'out_dir': 'output',
        'report_file': 'output/report.md',
        'fig_dir': 'output/figs',
        'missing_threshold': 0.2,  # If >20% missing, drop column
        'outlier_zscore': 3,
        'min_search_volume': 10,  # For high-potential keywords
        'max_competition': 0.3,   # For high-potential keywords
    }

# Utility: ensure output dirs exist
def ensure_dirs(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    os.makedirs(cfg['fig_dir'], exist_ok=True)

# 1. Load & Optimize Data
def load_and_optimize_csv(filepath, cfg):
    logging.info(f"Loading data from {filepath} in chunks...")
    # First, get dtypes from a small sample
    sample = pd.read_csv(filepath, nrows=1000)
    dtypes = {}
    for col in sample.columns:
        if sample[col].dtype == 'object':
            dtypes[col] = 'category'
        elif pd.api.types.is_integer_dtype(sample[col]):
            dtypes[col] = pd.api.types.infer_dtype(sample[col], skipna=True)
        elif pd.api.types.is_float_dtype(sample[col]):
            dtypes[col] = 'float32'
        else:
            dtypes[col] = sample[col].dtype
    # Now, read in chunks and concatenate
    chunks = []
    for chunk in tqdm(pd.read_csv(filepath, dtype=dtypes, chunksize=cfg['chunk_size']), desc='Reading CSV'):
        # Downcast numerics
        for col in chunk.select_dtypes(include=['int', 'float']).columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast='float')
        # Convert text to category
        for col in chunk.select_dtypes(include=['object']).columns:
            chunk[col] = chunk[col].astype('category')
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    logging.info(f"Loaded {len(df)} rows.")
    return df

# 2. Data Quality Checks
def data_quality_checks(df, cfg):
    logging.info("Running data quality checks...")
    # Missing values
    missing = df.isnull().mean()
    dropped_cols = []
    for col, frac in missing.items():
        if frac > cfg['missing_threshold']:
            df = df.drop(columns=[col])
            dropped_cols.append(col)
    if dropped_cols:
        logging.info(f"Dropped columns with >{cfg['missing_threshold']*100}% missing: {dropped_cols}")
    # Fill remaining missing
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('unknown')
    # Outlier detection (z-score)
    for col in [cfg['search_volume_col'], cfg['competition_col']]:
        if col in df.columns:
            col_z = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
            outliers = (np.abs(col_z) > cfg['outlier_zscore']).sum()
            logging.info(f"{col}: {outliers} outliers (z>{cfg['outlier_zscore']})")
    return df

# 2b. Text Normalization
def normalize_texts(df, cfg):
    logging.info("Normalizing keyword text...")
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [stemmer.stem(t) for t in tokens]
        return ' '.join(tokens)
    df['keyword_clean'] = df[cfg['keyword_col']].astype(str).apply(clean_text)
    return df

# 3. Exploratory Analysis
def exploratory_analysis(df, cfg):
    logging.info("Running exploratory analysis...")
    # Competition buckets
    if df[cfg['competition_col']].max() > 1.0:
        # Assume 0-100 scale, normalize
        df[cfg['competition_col']] = df[cfg['competition_col']] / 100.0
    bins = [0, 0.33, 0.66, 1.0]
    labels = ['Low', 'Medium', 'High']
    df['competition_level'] = pd.cut(df[cfg['competition_col']], bins=bins, labels=labels, include_lowest=True)
    # High-value keywords
    high_value = df[(df[cfg['search_volume_col']] >= cfg['min_search_volume']) &
                    (df[cfg['competition_col']] <= cfg['max_competition'])]
    # Per-language trends
    lang_stats = None
    if cfg['language_col'] in df.columns:
        lang_stats = df.groupby(cfg['language_col'])[[cfg['search_volume_col'], cfg['competition_col']]].agg(['mean', 'median', 'count'])
    return {
        'competition_dist': df['competition_level'].value_counts(),
        'high_value_keywords': high_value,
        'lang_stats': lang_stats,
        'df': df
    }

# 4. Keyword Clustering
def keyword_clustering(df, cfg):
    logging.info("Clustering keywords...")
    # Use only cleaned keywords
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['keyword_clean'])
    # Elbow method for k
    inertias = []
    K_range = range(2, cfg['max_clusters']+1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=cfg['random_state'], n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    # Find elbow (simple: where inertia drop slows)
    diffs = np.diff(inertias)
    elbow_k = K_range[np.argmin(diffs)+1] if len(diffs) > 1 else 3
    logging.info(f"Optimal k (elbow): {elbow_k}")
    # Final clustering
    km = KMeans(n_clusters=elbow_k, random_state=cfg['random_state'], n_init=10)
    clusters = km.fit_predict(X)
    df['cluster'] = clusters
    # Cluster validation
    sil = silhouette_score(X, clusters)
    logging.info(f"Silhouette score: {sil:.3f}")
    return df, km, vectorizer, elbow_k, sil

# 5. Visualizations
def make_visualizations(df, analysis, cfg):
    logging.info("Generating visualizations...")
    figs = {}
    # Histogram: search volume
    plt.figure(figsize=(8,4))
    sns.histplot(df[cfg['search_volume_col']], bins=30, kde=True)
    plt.title('Search Volume Distribution')
    plt.xlabel('Search Volume')
    plt.tight_layout()
    fig1 = os.path.join(cfg['fig_dir'], 'search_volume_hist.png')
    plt.savefig(fig1)
    figs['search_volume_hist'] = fig1
    plt.close()
    # Scatter: search volume vs competition
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=cfg['competition_col'], y=cfg['search_volume_col'], data=df, alpha=0.3)
    plt.title('Search Volume vs Competition')
    plt.xlabel('Competition')
    plt.ylabel('Search Volume')
    plt.tight_layout()
    fig2 = os.path.join(cfg['fig_dir'], 'search_vs_competition.png')
    plt.savefig(fig2)
    figs['search_vs_competition'] = fig2
    plt.close()
    # Bar: top keywords per cluster
    top_keywords = {}
    for c in sorted(df['cluster'].unique()):
        sub = df[df['cluster']==c]
        top = sub.nlargest(cfg['top_n_keywords'], cfg['search_volume_col'])
        top_keywords[c] = top[cfg['keyword_col']].tolist()
        plt.figure(figsize=(8,4))
        sns.barplot(x=cfg['search_volume_col'], y=cfg['keyword_col'], data=top, orient='h')
        plt.title(f'Top {cfg["top_n_keywords"]} Keywords in Cluster {c}')
        plt.tight_layout()
        figc = os.path.join(cfg['fig_dir'], f'cluster_{c}_top_keywords.png')
        plt.savefig(figc)
        figs[f'cluster_{c}_top_keywords'] = figc
        plt.close()
    # Per-language bar chart
    if cfg['language_col'] in df.columns:
        lang_counts = df[cfg['language_col']].value_counts().head(10)
        plt.figure(figsize=(8,4))
        sns.barplot(x=lang_counts.index, y=lang_counts.values)
        plt.title('Top Languages by Keyword Count')
        plt.xlabel('Language')
        plt.ylabel('Keyword Count')
        plt.tight_layout()
        figl = os.path.join(cfg['fig_dir'], 'top_languages.png')
        plt.savefig(figl)
        figs['top_languages'] = figl
        plt.close()
    return figs, top_keywords

# 6. Reporting
def generate_report(analysis, clustering, figs, top_keywords, cfg):
    logging.info("Generating report...")
    with open(cfg['report_file'], 'w', encoding='utf-8') as f:
        f.write("# SEO Keyword Analysis Report\n\n")
        # 1. Executive summary
        f.write("## Executive Summary\n\n")
        f.write(f"- Total keywords analyzed: {len(analysis['df'])}\n")
        f.write(f"- High-value keywords (high search, low competition): {len(analysis['high_value_keywords'])}\n")
        f.write(f"- Optimal number of keyword clusters: {clustering['elbow_k']} (Silhouette: {clustering['silhouette']:.2f})\n\n")
        # 2. Detailed metric analysis
        f.write("## Detailed Metric Analysis\n\n")
        f.write("### Competition Level Distribution\n")
        f.write(str(analysis['competition_dist']))
        f.write("\n\n")
        if analysis['lang_stats'] is not None:
            f.write("### Per-Language Trends\n")
            f.write(str(analysis['lang_stats']))
            f.write("\n\n")
        # 3. Visualizations
        f.write("## Visualizations\n\n")
        for desc, path in figs.items():
            f.write(f"![{desc}]({path})\n\n")
        # 4. Actionable recommendations
        f.write("## Actionable Recommendations\n\n")
        f.write("- Focus content on high-value keywords (high search, low competition).\n")
        f.write("- Target clusters with the most high-potential keywords.\n")
        if analysis['lang_stats'] is not None:
            f.write("- Prioritize languages with high search volume and low competition.\n")
        f.write("- Regularly update keyword list and re-cluster as trends shift.\n\n")
        # 5. Examples of high-potential keywords
        f.write("## High-Potential Keywords\n\n")
        top_high_value = analysis['high_value_keywords'].nlargest(cfg['top_n_keywords'], cfg['search_volume_col'])
        for _, row in top_high_value.iterrows():
            f.write(f"- {row[cfg['keyword_col']]} (Search: {row[cfg['search_volume_col']]}, Competition: {row[cfg['competition_col']]})\n")
        f.write("\n")
        # 6. Top keywords per cluster
        f.write("## Top Keywords per Cluster\n\n")
        for c, kws in top_keywords.items():
            f.write(f"### Cluster {c}\n")
            for kw in kws:
                f.write(f"- {kw}\n")
            f.write("\n")
    logging.info(f"Report written to {cfg['report_file']}")

# 7. Main pipeline
def main(filepath):
    cfg = get_config()
    ensure_dirs(cfg)
    if not os.path.exists(filepath):
        logging.error(f"Input file {filepath} not found.")
        sys.exit(1)
    df = load_and_optimize_csv(filepath, cfg)
    df = data_quality_checks(df, cfg)
    df = normalize_texts(df, cfg)
    analysis = exploratory_analysis(df, cfg)
    df, km, vectorizer, elbow_k, sil = keyword_clustering(analysis['df'], cfg)
    figs, top_keywords = make_visualizations(df, analysis, cfg)
    clustering = {'elbow_k': elbow_k, 'silhouette': sil}
    generate_report(analysis, clustering, figs, top_keywords, cfg)
    print(f"\nAnalysis complete. See {cfg['report_file']} for the report.")

# 8. Unit tests for critical functions
def test_load_and_optimize_csv():
    # Create a small test CSV
    test_csv = 'test_keywords.csv'
    pd.DataFrame({
        'keyword': ['apple', 'banana', 'carrot'],
        'search_volume': [100, 200, 150],
        'competition': [0.1, 0.5, 0.2],
        'language': ['en', 'en', 'en']
    }).to_csv(test_csv, index=False)
    cfg = get_config()
    df = load_and_optimize_csv(test_csv, cfg)
    assert len(df) == 3
    os.remove(test_csv)
    print('test_load_and_optimize_csv passed')

def test_keyword_clustering():
    # Simple test
    df = pd.DataFrame({
        'keyword': ['apple pie', 'banana bread', 'carrot cake', 'apple tart', 'banana split'],
        'search_volume': [100, 200, 150, 120, 180],
        'competition': [0.1, 0.5, 0.2, 0.15, 0.45],
        'keyword_clean': ['apple pie', 'banana bread', 'carrot cake', 'apple tart', 'banana split']
    })
    cfg = get_config()
    df, km, vectorizer, elbow_k, sil = keyword_clustering(df, cfg)
    assert 'cluster' in df.columns
    print('test_keyword_clustering passed')

def run_tests():
    test_load_and_optimize_csv()
    test_keyword_clustering()
    print('All tests passed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEO Keyword Analysis')
    parser.add_argument('--input', type=str, default='output-keywords-lans.csv', help='Input CSV file')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    args = parser.parse_args()
    if args.test:
        run_tests()
    else:
        main(args.input)
