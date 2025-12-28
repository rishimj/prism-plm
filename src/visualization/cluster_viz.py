"""Clustering visualization with comprehensive logging."""
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging_utils import get_logger, LogContext
from src.utils import constants

logger = get_logger("visualization.cluster_viz")


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """Reduce embeddings to 2D for visualization.

    Args:
        embeddings: Array of shape [n_samples, n_features]
        method: Reduction method ("umap", "tsne", "pca")
        n_components: Number of output dimensions
        **kwargs: Additional arguments for the reducer

    Returns:
        Reduced array of shape [n_samples, n_components]
    """
    logger.info(f"Reducing dimensions using {method}")
    logger.debug(f"  Input shape: {embeddings.shape}")

    with LogContext(logger, f"Dimensionality reduction ({method})"):
        if method == "umap":
            from src.visualization.reducers.umap_reducer import UMAPReducer
            reducer = UMAPReducer(n_components=n_components, **kwargs)
        elif method == "tsne":
            from src.visualization.reducers.tsne_reducer import TSNEReducer
            reducer = TSNEReducer(n_components=n_components, **kwargs)
        elif method == "pca":
            from src.visualization.reducers.pca_reducer import PCAReducer
            reducer = PCAReducer(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        reduced = reducer.fit_transform(embeddings)

    logger.debug(f"  Output shape: {reduced.shape}")
    return reduced


class ClusterVisualizer:
    """Visualization tools for clustering results."""

    def __init__(
        self,
        output_dir: str = "visualizations",
        figure_width: int = 14,
        figure_height: int = 10,
        dpi: int = 300,
    ):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
            figure_width: Figure width in inches
            figure_height: Figure height in inches
            dpi: Resolution for saved figures
        """
        logger.debug(f"Initializing ClusterVisualizer")
        logger.debug(f"  output_dir: {output_dir}")
        logger.debug(f"  figure_size: {figure_width}x{figure_height}")
        logger.debug(f"  dpi: {dpi}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.dpi = dpi

    def plot_embedding_space(
        self,
        coords: np.ndarray,
        clusters: np.ndarray,
        save_name: str = "embedding_space.png",
        colormap: str = "tab10",
        point_size: int = 50,
        alpha: float = 0.6,
        title: str = "Embedding Space",
        save_formats: Optional[List[str]] = None,
    ) -> Path:
        """Plot 2D embedding space with cluster colors.

        Args:
            coords: 2D coordinates [n_samples, 2]
            clusters: Cluster labels [n_samples]
            save_name: Filename to save
            colormap: Matplotlib colormap name
            point_size: Size of scatter points
            alpha: Transparency of points
            title: Plot title
            save_formats: List of formats to save (default: just use save_name extension)

        Returns:
            Path to saved figure
        """
        logger.info(f"Creating embedding space visualization")
        logger.debug(f"  n_samples: {len(coords)}")
        logger.debug(f"  n_clusters: {len(np.unique(clusters))}")

        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))

        # Handle noise points (-1)
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters[unique_clusters >= 0])

        # Get colormap
        if n_clusters <= 10:
            cmap = plt.get_cmap(colormap)
        else:
            cmap = plt.get_cmap("tab20")

        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            if cluster_id == -1:
                # Noise points in gray
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=point_size // 2,
                    c="gray",
                    alpha=alpha / 2,
                    label="Noise",
                )
            else:
                color_idx = cluster_id % 20
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=point_size,
                    c=[cmap(color_idx)],
                    alpha=alpha,
                    label=f"Cluster {cluster_id}",
                )

        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Legend with cluster counts
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
            title="Clusters",
        )
        legend.get_title().set_fontweight("bold")

        plt.tight_layout()

        # Save
        filepath = self.output_dir / save_name
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved embedding space plot to: {filepath}")
        return filepath

    def plot_cluster_grid(
        self,
        coords: np.ndarray,
        clusters: np.ndarray,
        n_clusters: Optional[int] = None,
        save_name: str = "cluster_grid.png",
    ) -> Path:
        """Plot individual cluster subplots.

        Args:
            coords: 2D coordinates [n_samples, 2]
            clusters: Cluster labels [n_samples]
            n_clusters: Number of clusters (auto-detected if None)
            save_name: Filename to save

        Returns:
            Path to saved figure
        """
        logger.info(f"Creating cluster grid visualization")

        unique_clusters = np.unique(clusters)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # Exclude noise

        if n_clusters is None:
            n_clusters = len(unique_clusters)

        # Calculate grid dimensions
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for idx, cluster_id in enumerate(unique_clusters[:n_clusters]):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Plot all points in light gray
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=10,
                c="lightgray",
                alpha=0.3,
            )

            # Highlight current cluster
            mask = clusters == cluster_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=30,
                c="steelblue",
                alpha=0.7,
            )

            ax.set_title(f"Cluster {cluster_id} (n={mask.sum()})")
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide empty subplots
        for idx in range(len(unique_clusters), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.suptitle("Individual Clusters", fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = self.output_dir / save_name
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved cluster grid to: {filepath}")
        return filepath

    def plot_activation_heatmap(
        self,
        activations: np.ndarray,
        save_name: str = "activation_heatmap.png",
        vmax: Optional[float] = None,
    ) -> Path:
        """Plot activation heatmap.

        Args:
            activations: 2D array of activations [positions, features]
            save_name: Filename to save
            vmax: Maximum value for colormap

        Returns:
            Path to saved figure
        """
        logger.info(f"Creating activation heatmap")
        logger.debug(f"  Shape: {activations.shape}")

        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height // 2))

        if vmax is None:
            vmax = np.percentile(np.abs(activations), 99)

        im = ax.imshow(
            activations.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )

        ax.set_xlabel("Sequence Position", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title("Activation Heatmap", fontsize=14, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Activation")
        plt.tight_layout()

        filepath = self.output_dir / save_name
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved activation heatmap to: {filepath}")
        return filepath

    def create_full_report(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
        metadata: List[Dict[str, Any]],
        reduction_method: str = "umap",
        experiment_name: str = "experiment",
        **kwargs,
    ) -> Path:
        """Create full visualization report.

        Args:
            embeddings: High-dimensional embeddings
            clusters: Cluster labels
            metadata: Sample metadata
            reduction_method: Method for dimensionality reduction
            experiment_name: Name for output files
            **kwargs: Additional arguments for reduction

        Returns:
            Path to output directory
        """
        logger.info(f"Creating full visualization report for {experiment_name}")

        # Create experiment subdirectory
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        with LogContext(logger, "Creating full report"):
            # 1. Reduce dimensions
            coords = reduce_dimensions(embeddings, method=reduction_method, **kwargs)

            # 2. Embedding space plot
            self.plot_embedding_space(
                coords,
                clusters,
                save_name=f"embedding_space_{reduction_method}.png",
            )

            # 3. Cluster grid
            self.plot_cluster_grid(coords, clusters)

            # 4. Save cluster statistics
            from src.analysis.clustering import compute_cluster_statistics

            stats = compute_cluster_statistics(embeddings, clusters)

            # Convert numpy arrays and types to JSON-serializable format
            stats_json = {}
            for k, v in stats.items():
                # Convert numpy int keys to Python int
                key = int(k) if isinstance(k, (np.integer, np.int64)) else k
                stats_json[key] = {}
                for inner_key, val in v.items():
                    if isinstance(val, np.ndarray):
                        stats_json[key][inner_key] = val.tolist()
                    elif isinstance(val, (np.integer, np.int64)):
                        stats_json[key][inner_key] = int(val)
                    elif isinstance(val, (np.floating, np.float64)):
                        stats_json[key][inner_key] = float(val)
                    else:
                        stats_json[key][inner_key] = val

            stats_path = self.output_dir / "cluster_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(stats_json, f, indent=2)

            logger.info(f"Saved cluster statistics to: {stats_path}")

        return self.output_dir


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "reduce_dimensions",
    "ClusterVisualizer",
]

