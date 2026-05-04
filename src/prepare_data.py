from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.units import UnitsWarning
from sklearn.preprocessing import StandardScaler


warnings.simplefilter("ignore", UnitsWarning)

RAW_DIR = Path("../data/raw")
OUT_DIR = Path("../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCIENCE_FILE = RAW_DIR / "gkvScienceCatv02.fits"
MASS_FILE = RAW_DIR / "StellarMassesGKVv24.fits"
MORPH_FILE = RAW_DIR / "gkvMorphologyv02.fits"

OUTPUT_PARQUET = OUT_DIR / "galaxies_clean.parquet"
OUTPUT_CSV = OUT_DIR / "galaxies_clean.csv"

FLUX_COLS = [
    "flux_ut", "flux_gt", "flux_rt", "flux_it",
    "flux_Zt", "flux_Yt", "flux_Jt", "flux_Ht", "flux_Kt"
]

MAG_COLS = [
    "mag_u", "mag_g", "mag_r", "mag_i",
    "mag_Z", "mag_Y", "mag_J", "mag_H", "mag_K"
]

COLOUR_COLS = [
    "u_g", "g_r", "r_i",
    "Z_Y", "Y_J", "J_H", "H_K"
]

PHYSICAL_COLS = [
    "spec_z", "stellar_mass"
]

FEATURE_COLS = MAG_COLS + COLOUR_COLS + PHYSICAL_COLS


def read_fits(path):
    print(f"Reading {path}")
    return Table.read(path).to_pandas()


def flux_to_mag(flux):
    return 8.9 - 2.5 * np.log10(flux)


def clean_hubble_type(value):
    if pd.isna(value):
        return np.nan

    value = str(value)
    value = value.replace("b'", "").replace("'", "")
    value = value.strip()

    if value in ["", "nan", "None"]:
        return np.nan

    return value


def main():
    # ==========================================================
    # 1. Read main science table
    # Contains KiDS + VIKING fluxes and spectroscopic redshift.
    # ==========================================================
    science = read_fits(SCIENCE_FILE)

    science_cols = [
        "uberID", "CATAID", "RAcen", "Deccen",
        *FLUX_COLS,
        "Z", "NQ", "SC"
    ]

    missing_science_cols = [c for c in science_cols if c not in science.columns]
    if missing_science_cols:
        raise KeyError(f"Missing columns in science table: {missing_science_cols}")

    science = science[science_cols]

    # Science-quality filtering
    science = science[
        (science["NQ"] > 2) &
        (science["SC"] >= 7) &
        (science["Z"] > 0) &
        (science["Z"] < 0.6)
    ]

    # Require valid fluxes in all KiDS + VIKING bands
    for col in FLUX_COLS:
        science = science[science[col] > 0]

    # ==========================================================
    # 2. Convert fluxes to magnitudes
    # ==========================================================
    for col in FLUX_COLS:
        band = col.replace("flux_", "").replace("t", "")
        science[f"mag_{band}"] = flux_to_mag(science[col])

    # ==========================================================
    # 3. Create colour features
    # Colours are important for separating galaxy populations.
    # ==========================================================
    science["u_g"] = science["mag_u"] - science["mag_g"]
    science["g_r"] = science["mag_g"] - science["mag_r"]
    science["r_i"] = science["mag_r"] - science["mag_i"]

    science["Z_Y"] = science["mag_Z"] - science["mag_Y"]
    science["Y_J"] = science["mag_Y"] - science["mag_J"]
    science["J_H"] = science["mag_J"] - science["mag_H"]
    science["H_K"] = science["mag_H"] - science["mag_K"]

    # ==========================================================
    # 4. Read and merge stellar mass table
    # StellarMassesGKVv24 joins by uberID.
    # ==========================================================
    masses = read_fits(MASS_FILE)

    mass_cols = ["uberID", "logmstar"]
    missing_mass_cols = [c for c in mass_cols if c not in masses.columns]
    if missing_mass_cols:
        raise KeyError(f"Missing columns in stellar mass table: {missing_mass_cols}")

    masses = masses[mass_cols]

    df = science.merge(masses, on="uberID", how="inner")

    # ==========================================================
    # 5. Read and merge morphology table
    # Morphology is used for evaluation, not for unsupervised training.
    # ==========================================================
    morph = read_fits(MORPH_FILE)

    morph_cols = [c for c in ["uberID", "CATAID", "HUBBLE_TYPE"] if c in morph.columns]
    if len(morph_cols) == 0:
        print("Warning: no usable morphology columns found.")
    else:
        morph = morph[morph_cols]

        if "uberID" in morph.columns:
            df = df.merge(morph, on="uberID", how="left", suffixes=("", "_morph"))
        elif "CATAID" in morph.columns:
            df = df.merge(morph, on="CATAID", how="left", suffixes=("", "_morph"))

    # ==========================================================
    # 6. Rename core columns
    # ==========================================================
    df = df.rename(columns={
        "RAcen": "RA",
        "Deccen": "Dec",
        "Z": "spec_z",
        "logmstar": "stellar_mass"
    })

    if "CATAID_morph" in df.columns:
        df = df.drop(columns=["CATAID_morph"])

    if "HUBBLE_TYPE" in df.columns:
        df["HUBBLE_TYPE"] = df["HUBBLE_TYPE"].apply(clean_hubble_type)

        # Remove non-galaxy / unclear labels from evaluation labels
        bad_labels = ["X", "Artifact", "Star"]
        df["HUBBLE_TYPE"] = df["HUBBLE_TYPE"].replace(bad_labels, np.nan)

    # ==========================================================
    # 7. Drop rows missing required ML features
    # ==========================================================
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise KeyError(f"Missing required ML feature columns: {missing_features}")

    df = df.dropna(subset=FEATURE_COLS)

    # ==========================================================
    # 8. Final scientific sanity filters
    # ==========================================================
    df = df[
        (df["spec_z"] > 0) &
        (df["spec_z"] < 0.6) &
        (df["stellar_mass"] > 6) &
        (df["stellar_mass"] < 12.5)
    ]

    for col in MAG_COLS:
        df = df[(df[col] > 5) & (df[col] < 30)]

    # ==========================================================
    # 9. Scale ML features for PCA / UMAP / t-SNE / clustering
    # ==========================================================
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[FEATURE_COLS])

    scaled_cols = [f"scaled_{c}" for c in FEATURE_COLS]

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=scaled_cols,
        index=df.index
    )

    # ==========================================================
    # 10. Build final output table
    # Raw features are kept for interpretation.
    # Scaled features are used for machine learning.
    # HUBBLE_TYPE is kept only for later evaluation.
    # ==========================================================
    keep_cols = [
        "uberID", "CATAID", "RA", "Dec",
        "spec_z", "stellar_mass",
        "HUBBLE_TYPE",
        *MAG_COLS,
        *COLOUR_COLS
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    final_df = pd.concat(
        [
            df[keep_cols].reset_index(drop=True),
            scaled_df.reset_index(drop=True)
        ],
        axis=1
    )

    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # ==========================================================
    # 11. Save outputs
    # ==========================================================
    final_df.to_parquet(OUTPUT_PARQUET, index=False)
    final_df.to_csv(OUTPUT_CSV, index=False)


    print("\nSaved:")
    print(OUTPUT_PARQUET)
    print(OUTPUT_CSV)

    print("\nFinal shape:")
    print(final_df.shape)

    print("\nML features to use in Part 2:")
    print(scaled_cols)

    if "HUBBLE_TYPE" in final_df.columns:
        print("\nMorphology labels available for evaluation:")
        print(final_df["HUBBLE_TYPE"].notna().sum())

        print("\nMorphology label counts:")
        print(final_df["HUBBLE_TYPE"].value_counts(dropna=False))


if __name__ == "__main__":
    main()