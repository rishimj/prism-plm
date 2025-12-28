#!/usr/bin/env python3
"""Clustering visualization script for PRISM-Bio.

This script loads embeddings or activations and creates clustering visualizations.

Usage:
    python scripts/run_clustering_viz.py --config configs/default.yaml
    python scripts/run_clustering_viz.py --embeddings path/to/embeddings.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.utils.logging_utils import setup_logging, log_system_info
from src.utils import constants, helpers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PRISM-Bio Clustering Visualization",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to saved embeddings (.pt or .npy file)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--reduction-method",
        type=str,
        choices=["umap", "tsne", "pca"],
        default="umap",
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        choices=["kmeans", "hdbscan", "spectral", "agglomerative"],
        default="kmeans",
        help="Clustering algorithm",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters (for kmeans/spectral/agglomerative)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration if provided
    if args.config:
        config = load_config(config_path=args.config)
    else:
        config = load_config()

    # Override with CLI args
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(config.output_dir) / "visualizations"

    # Setup logging
    logger, log_file = setup_logging(
        config.to_dict(),
        mode="clustering_viz",
        verbose=args.verbose or config.verbose,
    )

    log_system_info(logger)

    logger.info("=" * 80)
    logger.info("PRISM-BIO CLUSTERING VISUALIZATION")
    logger.info("=" * 80)

    try:
        # Load embeddings
        if args.embeddings:
            logger.info(f"Loading embeddings from: {args.embeddings}")
            if args.embeddings.suffix == ".pt":
                embeddings = torch.load(args.embeddings).numpy()
            elif args.embeddings.suffix == ".npy":
                embeddings = np.load(args.embeddings)
            else:
                raise ValueError(f"Unknown file format: {args.embeddings.suffix}")
        else:
            # Generate random embeddings for demo
            logger.warning("No embeddings provided, generating random demo data")
            np.random.seed(42)
            # Create 3 clusters
            embeddings = np.vstack([
                np.random.randn(50, 64) + np.array([5, 0] + [0] * 62),
                np.random.randn(50, 64) + np.array([-5, 0] + [0] * 62),
                np.random.randn(50, 64) + np.array([0, 5] + [0] * 62),
            ])

        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Load metadata if provided
        if args.metadata:
            with open(args.metadata) as f:
                metadata = json.load(f)
        else:
            metadata = [{"id": f"sample_{i}"} for i in range(len(embeddings))]

        # Cluster
        logger.info("")
        logger.info("=" * 80)
        logger.info("CLUSTERING")
        logger.info("=" * 80)

        from src.analysis.clustering import cluster_embeddings, compute_cluster_statistics

        labels = cluster_embeddings(
            embeddings,
            method=args.clustering_method,
            n_clusters=args.n_clusters,
        )

        stats = compute_cluster_statistics(embeddings, labels)

        # Visualize
        logger.info("")
        logger.info("=" * 80)
        logger.info("VISUALIZATION")
        logger.info("=" * 80)

        from src.visualization import ClusterVisualizer

        viz = ClusterVisualizer(output_dir=str(output_dir))

        viz.create_full_report(
            embeddings,
            labels,
            metadata,
            reduction_method=args.reduction_method,
            experiment_name="clustering_analysis",
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Visualizations saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

