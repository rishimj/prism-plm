"""HDBSCAN clustering with comprehensive logging."""
import numpy as np
from typing import Optional

from src.utils.logging_utils import get_logger, LogContext
from src.config.registry import CLUSTERING_REGISTRY

logger = get_logger("analysis.clustering.hdbscan")


@CLUSTERING_REGISTRY.register("hdbscan")
class HDBSCANClusterer:
    """HDBSCAN clustering implementation with logging."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        random_seed: int = 42,
        **kwargs,
    ):
        """Initialize the clusterer.

        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood (defaults to min_cluster_size)
            cluster_selection_epsilon: Distance threshold for cluster merging
            metric: Distance metric to use
            random_seed: Random seed (used if algorithm supports it)
            **kwargs: Additional arguments (ignored)
        """
        logger.debug(f"Initializing HDBSCANClusterer")
        logger.debug(f"  min_cluster_size: {min_cluster_size}")
        logger.debug(f"  min_samples: {min_samples}")
        logger.debug(f"  cluster_selection_epsilon: {cluster_selection_epsilon}")
        logger.debug(f"  metric: {metric}")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.random_seed = random_seed

        self.model = None
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN and return cluster labels.

        Args:
            embeddings: Array of shape [n_samples, n_features]

        Returns:
            Array of cluster labels [n_samples] (-1 for noise)
        """
        try:
            import hdbscan
        except ImportError:
            logger.error("HDBSCAN not installed. Install with: pip install hdbscan")
            raise

        logger.info(f"Running HDBSCAN clustering")
        logger.debug(f"  Input shape: {embeddings.shape}")
        logger.debug(f"  min_cluster_size: {self.min_cluster_size}")

        with LogContext(
            logger,
            "HDBSCAN fit_predict",
            n_samples=embeddings.shape[0],
            n_features=embeddings.shape[1],
        ):
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=self.metric,
            )

            logger.debug("Fitting HDBSCAN model...")
            self.labels_ = self.model.fit_predict(embeddings)
            self.probabilities_ = self.model.probabilities_

            # Log cluster statistics
            unique = np.unique(self.labels_)
            n_clusters = len(unique[unique >= 0])
            n_noise = np.sum(self.labels_ == -1)

            logger.info(f"Clustering complete. Found {n_clusters} clusters")
            logger.info(f"  Noise points: {n_noise} ({100*n_noise/len(self.labels_):.1f}%)")

            for cluster_id in unique:
                count = np.sum(self.labels_ == cluster_id)
                pct = 100 * count / len(self.labels_)
                label_str = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
                logger.info(f"  {label_str}: {count} samples ({pct:.1f}%)")

        return self.labels_

    def get_statistics(self) -> dict:
        """Get cluster statistics.

        Returns:
            Dictionary with clustering statistics
        """
        if self.labels_ is None:
            logger.warning("No clustering results available. Call fit_predict first.")
            return {}

        unique, counts = np.unique(self.labels_, return_counts=True)

        n_clusters = len(unique[unique >= 0])
        n_noise = int(np.sum(self.labels_ == -1))

        stats = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_sizes": {
                int(k): int(v) for k, v in zip(unique, counts) if k >= 0
            },
        }

        logger.debug(f"Cluster statistics: {stats}")
        return stats






