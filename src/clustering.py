from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

DATA_FILE = Path("../data/processed/galaxies_embeddings.csv")
OUT_DIR = Path("../results")
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FULL_OUTPUT_FILE = OUT_DIR / "galaxies_clustered.csv"
SUBSET_OUTPUT_FILE = OUT_DIR / "morphology_subset_clustered.csv"
N_CLUSTERS = 7


def load_data():
    print("Loading embedding dataset...")

    df = pd.read_csv(DATA_FILE)

    required_cols = ["umap_1", "umap_2"]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols)

    X = df[["umap_1", "umap_2"]].copy()

    print(f"Rows: {df.shape[0]}")
    print("Using UMAP embedding for clustering.")

    return df, X


def run_kmeans(X):
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    return labels, model


def run_gmm(X):
    model = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="full",
        random_state=42
    )

    labels = model.fit_predict(X)

    return labels, model


def evaluate_clustering(X, labels, method_name):
    silhouette = silhouette_score(X, labels)

    print(f"\n{method_name} silhouette score:")
    print(silhouette)

    return silhouette


def evaluate_against_morphology(df, cluster_col, method_name):
    if "HUBBLE_TYPE" not in df.columns:
        print(f"\nNo HUBBLE_TYPE column found for {method_name}.")
        return None, None

    subset = df.dropna(subset=["HUBBLE_TYPE"]).copy()

    if subset.empty:
        print(f"\nNo morphology subset available for {method_name}.")
        return None, None

    encoder = LabelEncoder()

    true_labels = encoder.fit_transform(subset["HUBBLE_TYPE"])
    pred_labels = subset[cluster_col]

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    print(f"\n{method_name} morphology comparison:")
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Normalized Mutual Information (NMI): {nmi}")

    return ari, nmi


def plot_clusters(df, cluster_col, title, filename, point_size=2):
    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        df["umap_1"],
        df["umap_2"],
        c=df[cluster_col],
        s=point_size,
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel("umap_1")
    plt.ylabel("umap_2")

    handles, _ = scatter.legend_elements()

    plt.legend(
        handles,
        sorted(df[cluster_col].unique()),
        title="Cluster",
        markerscale=3,
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


def plot_morphology(df, title, filename, point_size=5):
    if "HUBBLE_TYPE" not in df.columns:
        return

    subset = df.dropna(subset=["HUBBLE_TYPE"]).copy()

    if subset.empty:
        return

    labels = subset["HUBBLE_TYPE"].astype("category")
    codes = labels.cat.codes

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        subset["umap_1"],
        subset["umap_2"],
        c=codes,
        s=point_size,
        alpha=0.8
    )

    plt.title(title)
    plt.xlabel("umap_1")
    plt.ylabel("umap_2")

    handles, _ = scatter.legend_elements()

    plt.legend(
        handles,
        labels.cat.categories,
        title="HUBBLE_TYPE",
        markerscale=3,
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


def run_full_population_clustering(df, X):
    print("FULL POPULATION CLUSTERING")

    results = []

    # K-Means on all galaxies

    print("\nRunning K-Means on full dataset")
    kmeans_labels, _ = run_kmeans(X)

    df["kmeans_cluster"] = kmeans_labels

    kmeans_sil = evaluate_clustering(
        X,
        kmeans_labels,
        "Full K-Means"
    )

    kmeans_ari, kmeans_nmi = evaluate_against_morphology(
        df,
        "kmeans_cluster",
        "Full K-Means"
    )

    plot_clusters(
        df,
        "kmeans_cluster",
        "K-Means Clusters on UMAP Embedding",
        "kmeans_clusters_full.png",
        point_size=2
    )

    results.append({
        "analysis": "full_population",
        "method": "KMeans",
        "silhouette": kmeans_sil,
        "ARI": kmeans_ari,
        "NMI": kmeans_nmi
    })

    # GMM on all galaxies

    print("\nRunning GMM on full dataset")
    gmm_labels, _ = run_gmm(X)

    df["gmm_cluster"] = gmm_labels

    gmm_sil = evaluate_clustering(
        X,
        gmm_labels,
        "Full GMM"
    )

    gmm_ari, gmm_nmi = evaluate_against_morphology(
        df,
        "gmm_cluster",
        "Full GMM"
    )

    plot_clusters(
        df,
        "gmm_cluster",
        "GMM Clusters on UMAP Embedding",
        "gmm_clusters_full.png",
        point_size=2
    )

    results.append({
        "analysis": "full_population",
        "method": "GMM",
        "silhouette": gmm_sil,
        "ARI": gmm_ari,
        "NMI": gmm_nmi
    })

    return df, results


def run_morphology_subset_clustering(df):
    print("MORPHOLOGY SUBSET CLUSTERING")

    if "HUBBLE_TYPE" not in df.columns:
        print("No HUBBLE_TYPE column found. Skipping subset clustering.")
        return None, []

    subset = df.dropna(subset=["HUBBLE_TYPE"]).copy()

    if subset.empty:
        print("No labelled morphology rows found. Skipping subset clustering.")
        return None, []

    X_subset = subset[["umap_1", "umap_2"]].copy()

    print(f"Subset rows: {subset.shape[0]}")

    results = []

    # Plot original morphology labels
    plot_morphology(
        subset,
        "Morphology Labels on UMAP Subset",
        "morphology_labels_subset.png",
        point_size=8
    )

    # K-Means on morphology subset

    print("\nRunning K-Means on morphology subset")
    subset_kmeans_labels, _ = run_kmeans(X_subset)

    subset["subset_kmeans_cluster"] = subset_kmeans_labels

    subset_kmeans_sil = evaluate_clustering(
        X_subset,
        subset_kmeans_labels,
        "Subset K-Means"
    )

    subset_kmeans_ari, subset_kmeans_nmi = evaluate_against_morphology(
        subset,
        "subset_kmeans_cluster",
        "Subset K-Means"
    )

    plot_clusters(
        subset,
        "subset_kmeans_cluster",
        "K-Means Clusters on UMAP Morphology Subset",
        "kmeans_clusters_subset.png",
        point_size=8
    )

    results.append({
        "analysis": "morphology_subset",
        "method": "KMeans",
        "silhouette": subset_kmeans_sil,
        "ARI": subset_kmeans_ari,
        "NMI": subset_kmeans_nmi
    })

    # GMM on morphology subset

    print("\nRunning GMM on morphology subset")
    subset_gmm_labels, _ = run_gmm(X_subset)

    subset["subset_gmm_cluster"] = subset_gmm_labels

    subset_gmm_sil = evaluate_clustering(
        X_subset,
        subset_gmm_labels,
        "Subset GMM"
    )

    subset_gmm_ari, subset_gmm_nmi = evaluate_against_morphology(
        subset,
        "subset_gmm_cluster",
        "Subset GMM"
    )

    plot_clusters(
        subset,
        "subset_gmm_cluster",
        "GMM Clusters on UMAP Morphology Subset",
        "gmm_clusters_subset.png",
        point_size=8
    )

    results.append({
        "analysis": "morphology_subset",
        "method": "GMM",
        "silhouette": subset_gmm_sil,
        "ARI": subset_gmm_ari,
        "NMI": subset_gmm_nmi
    })

    return subset, results


def main():
    df, X = load_data()

    # Full-population clustering
    df, full_results = run_full_population_clustering(df, X)

    # Morphology-subset clustering
    subset, subset_results = run_morphology_subset_clustering(df)

    # Save outputs
    df.to_csv(FULL_OUTPUT_FILE, index=False)

    if subset is not None:
        subset.to_csv(SUBSET_OUTPUT_FILE, index=False)

    all_results = pd.DataFrame(full_results + subset_results)
    all_results.to_csv(OUT_DIR / "clustering_metrics.csv", index=False)

    print("\nComplete!")

if __name__ == "__main__":
    main()