# SEO Keyword Analysis Pipeline

This project provides a production-ready, modular Python pipeline for analyzing keyword datasets (e.g., `output-keywords-lans.csv`) to generate actionable SEO and content strategy insights. It is optimized for Google Colab and Docker (CPU-only), and is easily adaptable for future datasets.

## Features
- **Efficient Data Loading:** Chunked reading, memory optimization (categoricals, downcasting numerics).
- **Data Quality Checks:** Handles missing values, outliers, and normalizes text (lowercase, punctuation removal, stemming).
- **Exploratory Analysis:** Competition buckets, high-value keyword identification, per-language trends (if language info present).
- **Keyword Clustering:** TF-IDF vectorization and K-means clustering (elbow method for optimal k, silhouette validation).
- **Visualizations:** Histograms, scatter plots, bar charts (matplotlib/seaborn).
- **Reporting:** Markdown report with executive summary, metrics, visualizations, recommendations, and high-potential keyword examples.
- **Modular & Testable:** Functions for each task, error handling, progress tracking, and unit tests for critical functions.
- **Configurable:** All key parameters (chunk size, thresholds, etc.) are easily adjustable.

## How to Run

### On Google Colab
1. Upload `analysis.py` and your `output-keywords-lans.csv` to the Colab environment.
2. Install dependencies:
   ```python
   !pip install pandas numpy matplotlib seaborn scikit-learn tqdm nltk spacy
   import nltk; nltk.download('stopwords')
   import spacy; spacy.blank('en')
   ```
3. Run the script:
   ```python
   !python analysis.py --input output-keywords-lans.csv
   ```
4. The report and figures will be saved in the `output/` directory.

### With Docker
1. Build the Docker image:
   ```sh
   docker build -t seo-keyword-analysis .
   ```
2. Run the container, mounting your CSV file:
   ```sh
   docker run --rm -v $(pwd)/output-keywords-lans.csv:/app/output-keywords-lans.csv seo-keyword-analysis
   ```
3. The report and figures will be in the `output/` directory inside the container (mount a volume if you want to access them on the host).

### Run Unit Tests
```sh
python analysis.py --test
```

## Output
- **Markdown Report:** `output/report.md` with all findings, recommendations, and visualizations.
- **Figures:** Saved in `output/figs/`.

## How It Works
1. **Load & Optimize:** Reads the CSV in chunks, optimizes memory usage.
2. **Quality Checks:** Handles missing data, outliers, and normalizes keyword text.
3. **Exploratory Analysis:** Analyzes search volume, competition, and language trends.
4. **Clustering:** Groups keywords using TF-IDF and K-means, finds optimal cluster count.
5. **Visualization:** Generates and saves key plots.
6. **Reporting:** Compiles all results into a markdown report with actionable insights.

## Customization
- Adjust parameters in `get_config()` in `analysis.py` for chunk size, thresholds, etc.
- Add or modify columns as needed for your dataset.

## Requirements
- Python 3.10+
- pandas, numpy, matplotlib, seaborn, scikit-learn, tqdm, nltk, spacy

## Notes
- If your dataset is very large, increase the chunk size or run on a machine with more memory.
- The script is robust to missing columns (e.g., language) and will skip language-specific analysis if not present.

---

**For any issues, please check the logs or open an issue.**
