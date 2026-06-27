"""Run PCA on a raster stack point table and write simple diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


GRID_COLUMNS = ["x", "y", "row", "col"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA on a CSV table made by raster_stack_to_table.py."
    )
    parser.add_argument("input_csv", type=Path, help="Input raster stack CSV.")
    parser.add_argument("output_dir", type=Path, help="Directory for PCA outputs.")
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Specific raster/value columns to include. Defaults to all numeric non-grid columns.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of PCA components to keep. Default: 5.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row limit for PCA fitting and plotting.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for optional row sampling. Default: 42.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Do not standardize variables before PCA.",
    )
    return parser.parse_args()


def select_feature_columns(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [column for column in requested if column not in df.columns]
        if missing:
            raise SystemExit(f"Requested column(s) not found: {', '.join(missing)}")
        return requested

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in GRID_COLUMNS]


def save_explained_variance(pca: PCA, output_dir: Path) -> None:
    explained = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, pca.n_components_ + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
            "eigenvalue": pca.explained_variance_,
        }
    )
    explained.to_csv(output_dir / "explained_variance.csv", index=False)


def save_scores(
    df: pd.DataFrame,
    scores: np.ndarray,
    output_dir: Path,
) -> None:
    score_columns = [f"PC{i}" for i in range(1, scores.shape[1] + 1)]
    available_grid_columns = [column for column in GRID_COLUMNS if column in df.columns]
    scores_df = pd.concat(
        [
            df[available_grid_columns].reset_index(drop=True),
            pd.DataFrame(scores, columns=score_columns),
        ],
        axis=1,
    )
    scores_df.to_csv(output_dir / "pca_scores.csv", index=False)


def save_loadings(
    pca: PCA,
    feature_columns: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    component_columns = [f"PC{i}" for i in range(1, pca.n_components_ + 1)]
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_columns,
    )
    loadings.insert(0, "variable", feature_columns)
    loadings["loading_intensity_all_components"] = np.sqrt(
        np.square(pca.components_.T).sum(axis=1)
    )
    if pca.n_components_ >= 2:
        loadings["loading_intensity_pc1_pc2"] = np.sqrt(
            np.square(loadings["PC1"]) + np.square(loadings["PC2"])
        )
    loadings.to_csv(output_dir / "pca_loadings.csv", index=False)
    return loadings


def plot_scree(pca: PCA, output_dir: Path) -> None:
    components = np.arange(1, pca.n_components_ + 1)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(components, pca.explained_variance_ratio_, color="#4c78a8")
    ax.plot(components, cumulative, marker="o", color="#f58518")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_xticks(components)
    ax.set_ylim(0, min(1.0, max(cumulative[-1] * 1.08, 0.1)))
    fig.tight_layout()
    fig.savefig(output_dir / "scree_plot.png", dpi=180)
    plt.close(fig)


def plot_score_scatter(scores: np.ndarray, pca: PCA, output_dir: Path) -> None:
    if scores.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sample = scores
    if scores.shape[0] > 50_000:
        rng = np.random.default_rng(42)
        sample = scores[rng.choice(scores.shape[0], size=50_000, replace=False)]
    ax.scatter(sample[:, 0], sample[:, 1], s=3, alpha=0.25, color="#4c78a8")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Pixel scores")
    fig.tight_layout()
    fig.savefig(output_dir / "pc1_pc2_scores.png", dpi=180)
    plt.close(fig)


def plot_loading_vectors(loadings: pd.DataFrame, pca: PCA, output_dir: Path) -> None:
    if pca.n_components_ < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)

    for _, row in loadings.iterrows():
        ax.arrow(
            0,
            0,
            row["PC1"],
            row["PC2"],
            color="#e45756",
            alpha=0.75,
            head_width=0.025,
            length_includes_head=True,
        )
        ax.text(row["PC1"] * 1.08, row["PC2"] * 1.08, row["variable"], fontsize=8)

    limit = max(
        abs(loadings["PC1"]).max(),
        abs(loadings["PC2"]).max(),
        0.1,
    )
    ax.set_xlim(-limit * 1.25, limit * 1.25)
    ax.set_ylim(-limit * 1.25, limit * 1.25)
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("PC2 loading")
    ax.set_title("PCA loading vectors")
    fig.tight_layout()
    fig.savefig(output_dir / "pc1_pc2_loading_vectors.png", dpi=180)
    plt.close(fig)


def plot_loading_heatmap(loadings: pd.DataFrame, output_dir: Path) -> None:
    pc_columns = [column for column in loadings.columns if column.startswith("PC")]
    values = loadings[pc_columns].to_numpy()

    fig_height = max(5, min(18, 0.35 * len(loadings)))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(pc_columns)), labels=pc_columns)
    ax.set_yticks(np.arange(len(loadings)), labels=loadings["variable"])
    ax.set_title("PCA loadings")
    fig.colorbar(image, ax=ax, label="Loading")
    fig.tight_layout()
    fig.savefig(output_dir / "loading_heatmap.png", dpi=180)
    plt.close(fig)


def plot_loading_intensity(loadings: pd.DataFrame, output_dir: Path) -> None:
    intensity_column = (
        "loading_intensity_pc1_pc2"
        if "loading_intensity_pc1_pc2" in loadings.columns
        else "loading_intensity_all_components"
    )
    ranked = loadings.sort_values(intensity_column, ascending=True)

    fig_height = max(5, min(18, 0.35 * len(ranked)))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(ranked["variable"], ranked[intensity_column], color="#54a24b")
    ax.set_xlabel("Loading vector intensity")
    ax.set_title("Variable influence in PCA space")
    fig.tight_layout()
    fig.savefig(output_dir / "loading_intensity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV does not exist: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    feature_columns = select_feature_columns(df, args.columns)
    if len(feature_columns) < 2:
        raise SystemExit("PCA requires at least two numeric feature columns.")

    working_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    if args.max_rows and len(working_df) > args.max_rows:
        working_df = working_df.sample(n=args.max_rows, random_state=args.random_state)
    working_df = working_df.reset_index(drop=True)

    features = working_df[feature_columns].to_numpy(dtype=float)
    if args.no_standardize:
        transformed = features
    else:
        transformed = StandardScaler().fit_transform(features)

    n_components = min(args.n_components, transformed.shape[1], transformed.shape[0])
    pca = PCA(n_components=n_components, random_state=args.random_state)
    scores = pca.fit_transform(transformed)

    save_scores(working_df, scores, args.output_dir)
    save_explained_variance(pca, args.output_dir)
    loadings = save_loadings(pca, feature_columns, args.output_dir)
    plot_scree(pca, args.output_dir)
    plot_score_scatter(scores, pca, args.output_dir)
    plot_loading_vectors(loadings, pca, args.output_dir)
    plot_loading_heatmap(loadings, args.output_dir)
    plot_loading_intensity(loadings, args.output_dir)

    print(f"Used {len(working_df)} row(s) and {len(feature_columns)} variable(s)")
    print(f"Wrote PCA outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
