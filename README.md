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


# Part 2: Dimensionality Reduction

## Overview

In this stage, we apply **dimensionality reduction** techniques to transform the high-dimensional galaxy dataset into a low-dimensional representation (embedding).

## Goal

- Reduce the feature space (≈20+ dimensions) into **2D embeddings**
- Compare different dimensionality reduction methods
- Understand how galaxy properties are distributed
- Evaluate whether known galaxy types align with the embedding


## Methods

### 1. PCA (Principal Component Analysis)
- Linear dimensionality reduction
- Captures global variance
- Used as a baseline method

### 2. UMAP (Uniform Manifold Approximation and Projection)
- Non-linear method
- Preserves both local and global structure
- Best suited for clustering

### 3. t-SNE (t-distributed Stochastic Neighbor Embedding)
- Focuses on local relationships
- Good for visualization
- Applied to a subset of the data due to computational cost


## Requirements

Install Python packages in the VS Code terminal or Command Prompt:

```bash
pip install pandas numpy scikit-learn umap-learn matplotlib pyarrow
```

In the src folder, run file called embeddings.py
This file, if successful, creates new files in data/processed called galaxies_embeddings.csv and galaxies_embeddings.parquet.
The script also creates a new folder called "results" and inside, there should be graphs called:
- pca_embedding.png
- pca_by_morphology.png
- umap_embedding.png
- umap_by_morphology.png
- tsne_embedding.png
- tsne_by_morphology.png