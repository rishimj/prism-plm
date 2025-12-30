"""KMeans clustering with comprehensive logging."""
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional

from src.utils.logging_utils import get_logger, LogContext
from src.config.registry import CLUSTERING_REGISTRY

logger = get_logger("analysis.clustering.kmeans")


@CLUSTERING_REGISTRY.register("kmeans")
class KMeansClusterer:
    """KMeans clustering implementation with logging."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_seed: int = 42,
        kmeans_n_init: int = 10,
        kmeans_max_iter: int = 300,
        **kwargs,
    ):
        """Initialize the clusterer.

        Args:
            n_clusters: Number of clusters
            random_seed: Random seed for reproducibility
            kmeans_n_init: Number of initializations
            kmeans_max_iter: Maximum iterations per initialization
            **kwargs: Additional arguments (ignored)
        """
        logger.debug(f"Initializing KMeansClusterer")
        logger.debug(f"  n_clusters: {n_clusters}")
        logger.debug(f"  random_seed: {random_seed}")
        logger.debug(f"  n_init: {kmeans_n_init}")
        logger.debug(f"  max_iter: {kmeans_max_iter}")

        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.n_init = kmeans_n_init
        self.max_iter = kmeans_max_iter

        self.model: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit KMeans and return cluster labels.

        Args:
            embeddings: Array of shape [n_samples, n_features]

        Returns:
            Array of cluster labels [n_samples]
        """
        logger.info(f"Running KMeans clustering")
        logger.debug(f"  Input shape: {embeddings.shape}")
        logger.debug(f"  n_clusters: {self.n_clusters}")

        with LogContext(
            logger,
            "KMeans fit_predict",
            n_samples=embeddings.shape[0],
            n_features=embeddings.shape[1],
        ):
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_seed,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )

            logger.debug("Fitting KMeans model...")
            self.labels_ = self.model.fit_predict(embeddings)
            self.cluster_centers_ = self.model.cluster_centers_

            # Log cluster statistics
            unique, counts = np.unique(self.labels_, return_counts=True)
            logger.info(f"Clustering complete. Found {len(unique)} clusters:")
            for cluster_id, count in zip(unique, counts):
                pct = 100 * count / len(self.labels_)
                logger.info(f"  Cluster {cluster_id}: {count} samples ({pct:.1f}%)")

            logger.debug(f"  Inertia: {self.model.inertia_:.4f}")
            logger.debug(f"  Iterations: {self.model.n_iter_}")

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

        stats = {
            "n_clusters": len(unique),
            "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
            "inertia": float(self.model.inertia_) if self.model else None,
            "n_iter": int(self.model.n_iter_) if self.model else None,
        }

        logger.debug(f"Cluster statistics: {stats}")
        return stats

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings.

        Args:
            embeddings: Array of shape [n_samples, n_features]

        Returns:
            Array of cluster labels [n_samples]
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_predict first.")

        logger.debug(f"Predicting labels for {embeddings.shape[0]} samples")
        return self.model.predict(embeddings)






