# Galaxy-classification-with-dimensionality-reduction-and-clustering


## Part 1: Data Processing

The workflow combines:
- **Photometry** (KiDS + VIKING)
- **Spectroscopic data** (GAMA)
- **Derived properties** (stellar mass, morphology)

The output is a clean, machine-learning-ready dataset that will be used in:
- **Dimensionality Reduction** (PCA, UMAP, t-SNE)
- **Clustering** (K-Means, GMM, DBSCAN, HDBSCAN)

## Requirements

Install Python packages in the VS Code terminal or Command Prompt:

```bash
pip install pandas numpy astropy scikit-learn pyarrow
```
Visual Studio Code is recommended to run the script. 
Go to src folder and run prepare_data.py.
This creates new, cleaned data in data/processed named galaxies_clean.csv and galaxies_clean.parquet.
