from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
warnings.filterwarnings("ignore")

DATA_FILE = Path("../data/processed/galaxies_clean.csv")
OUT_DIR = Path("../data/processed")
PLOT_DIR = Path("../results/plots")

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "galaxies_embeddings.csv"
OUTPUT_PARQUET = OUT_DIR / "galaxies_embeddings.parquet"


def load_data():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_FILE)

    feature_cols = [c for c in df.columns if c.startswith("scaled_")]

    if len(feature_cols) == 0:
        raise ValueError("No scaled feature columns found. Check Part 1 output.")

    X = df[feature_cols].copy()

    print(f"Rows: {df.shape[0]}")
    print(f"ML features: {len(feature_cols)}")
    print("Features used:")
    print(feature_cols)

    return df, X, feature_cols


def run_pca(X):
    print("\nRunning PCA...")

    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(X)

    print("PCA explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())

    return embedding


def run_umap(X):
    print("\nRunning UMAP...")

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    embedding = umap_model.fit_transform(X)

    return embedding


def run_tsne(X, max_points=30000):
    print("\nRunning t-SNE...")

    if X.shape[0] > max_points:
        print(f"Dataset is large, sampling {max_points} rows for t-SNE.")
        sample_idx = np.random.default_rng(42).choice(
            X.index,
            size=max_points,
            replace=False
        )
        X_sample = X.loc[sample_idx]
    else:
        sample_idx = X.index
        X_sample = X

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    embedding = tsne.fit_transform(X_sample)

    return sample_idx, embedding


def plot_embedding(df, x_col, y_col, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], s=1, alpha=0.5)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


def plot_embedding_by_morphology(df, x_col, y_col, title, filename):
    if "HUBBLE_TYPE" not in df.columns:
        print("No HUBBLE_TYPE column found, skipping morphology plot.")
        return

    subset = df.dropna(subset=["HUBBLE_TYPE"])

    if subset.empty:
        print("No morphology labels available, skipping morphology plot.")
        return

    plt.figure(figsize=(9, 7))

    labels = subset["HUBBLE_TYPE"].astype("category")
    codes = labels.cat.codes

    scatter = plt.scatter(
        subset[x_col],
        subset[y_col],
        c=codes,
        s=4,
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

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


def enforce_numeric_columns(df):
    allowed_text_cols = ["HUBBLE_TYPE"]

    for col in df.columns:
        if col not in allowed_text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_check_cols = [
        c for c in df.columns
        if (
            c.startswith("mag_")
            or c.startswith("scaled_")
            or c.startswith("pca_")
            or c.startswith("umap_")
            or c.startswith("tsne_")
            or c in [
                "uberID", "CATAID", "RA", "Dec",
                "spec_z", "stellar_mass",
                "u_g", "g_r", "r_i", "Z_Y", "Y_J", "J_H", "H_K"
            ]
        )
    ]

    bad_cols = [
        col for col in numeric_check_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if bad_cols:
        raise TypeError(f"These columns are not numeric: {bad_cols}")

    nan_counts = df[numeric_check_cols].isna().sum()
    problematic_nans = nan_counts[nan_counts > 0]

    if not problematic_nans.empty:
        print("\nWarning: some numeric columns contain NaN values:")
        print(problematic_nans)

    print("All required feature and embedding columns are numeric.")

    return df


def main():
    df, X, feature_cols = load_data()

    # PCA

    pca_embedding = run_pca(X)
    df["pca_1"] = pca_embedding[:, 0]
    df["pca_2"] = pca_embedding[:, 1]

    plot_embedding(
        df,
        "pca_1",
        "pca_2",
        "PCA embedding of galaxies",
        "pca_embedding.png"
    )

    plot_embedding_by_morphology(
        df,
        "pca_1",
        "pca_2",
        "PCA embedding coloured by morphology",
        "pca_by_morphology.png"
    )

    # UMAP

    umap_embedding = run_umap(X)
    df["umap_1"] = umap_embedding[:, 0]
    df["umap_2"] = umap_embedding[:, 1]

    plot_embedding(
        df,
        "umap_1",
        "umap_2",
        "UMAP embedding of galaxies",
        "umap_embedding.png"
    )

    plot_embedding_by_morphology(
        df,
        "umap_1",
        "umap_2",
        "UMAP embedding coloured by morphology",
        "umap_by_morphology.png"
    )

    # t-SNE

    tsne_idx, tsne_embedding = run_tsne(X, max_points=30000)

    df["tsne_1"] = np.nan
    df["tsne_2"] = np.nan

    df.loc[tsne_idx, "tsne_1"] = tsne_embedding[:, 0]
    df.loc[tsne_idx, "tsne_2"] = tsne_embedding[:, 1]

    tsne_subset = df.dropna(subset=["tsne_1", "tsne_2"])

    plot_embedding(
        tsne_subset,
        "tsne_1",
        "tsne_2",
        "t-SNE embedding of galaxies",
        "tsne_embedding.png"
    )

    plot_embedding_by_morphology(
        tsne_subset,
        "tsne_1",
        "tsne_2",
        "t-SNE embedding coloured by morphology",
        "tsne_by_morphology.png"
    )

    df = enforce_numeric_columns(df)

    # Save embedding dataset

    df.to_csv(OUTPUT_FILE, index=False)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(OUTPUT_PARQUET)

    print("\nPlots saved in:")
    print(PLOT_DIR)

if __name__ == "__main__":
    main()