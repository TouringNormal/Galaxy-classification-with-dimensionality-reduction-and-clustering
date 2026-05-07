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

In the src folder, run a file called embeddings.py
This file, if successful, creates new files in data/processed called galaxies_embeddings.csv and galaxies_embeddings.parquet.
The script also creates a new folder called "results" and inside, there should be graphs called:
- pca_embedding.png
- pca_by_morphology.png
- umap_embedding.png
- umap_by_morphology.png
- tsne_embedding.png
- tsne_by_morphology.png

# Galaxy Classification – Part 3: Clustering

## Overview

In this stage, we applied clustering algorithms to the low-dimensional galaxy embeddings produced in Part 2.

## Goal

- Apply clustering methods to the UMAP embedding
- Compare clustering quality
- Validate clusters using known morphology labels

## Methods Used

### 1. K-Means
- Distance-based clustering algorithm
- Requires predefined number of clusters
- Used as a baseline clustering method

### 2. Gaussian Mixture Model (GMM)
- Probabilistic clustering algorithm
- Allows overlapping cluster distributions
- Better suited for continuous galaxy populations

## Requirements

Install Python packages in the VS Code terminal or Command Prompt:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow
```

In the src folder, run a file called clustering.py
This file, if successful, creates new files in data/results called: 
- clustering_metrics.csv
- galaxier_clustered.csv
- morphology_subset_clustered.csv

The script also creates new graphs into the results/plots folder and inside, there should be graphs called:
- morphology_labels_subset.png
- gmm_clusters_full.png
- gmm_clusters_subset.png
- kmeans_clusters_full.png
- kmeans_clusters_subset.png


# Final results:

- Final cleaned dataset contained **190,617 galaxies**
- Morphology-labelled evaluation subset contained **6,738 galaxies**
- UMAP produced the clearest low-dimensional representation of galaxy structure.
- K-Means and GMM clustering were applied on the UMAP embedding using **7 clusters**
- Full-population clustering achieved:
  - **K-Means:** silhouette = 0.376, ARI = 0.196, NMI = 0.251
  - **GMM:** silhouette = 0.344, ARI = 0.275, NMI = 0.288
- Morphology-subset clustering achieved:
  - **K-Means:** silhouette = 0.431, ARI = 0.146, NMI = 0.232
  - **GMM:** silhouette = 0.269, ARI = 0.140, NMI = 0.254