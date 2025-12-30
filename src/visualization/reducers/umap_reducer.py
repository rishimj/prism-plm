"""UMAP dimensionality reducer."""
import numpy as np
from typing import Optional

from src.utils.logging_utils import get_logger
from src.config.registry import REDUCER_REGISTRY

logger = get_logger("visualization.reducers.umap")


@REDUCER_REGISTRY.register("umap")
class UMAPReducer:
    """UMAP dimensionality reduction."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize the reducer.

        Args:
            n_components: Number of output dimensions
            n_neighbors: Number of neighbors for local structure
            min_dist: Minimum distance between points in output
            metric: Distance metric to use
            random_state: Random seed
            **kwargs: Additional arguments (ignored)
        """
        logger.debug(f"Initializing UMAPReducer")
        logger.debug(f"  n_components: {n_components}")
        logger.debug(f"  n_neighbors: {n_neighbors}")
        logger.debug(f"  min_dist: {min_dist}")
        logger.debug(f"  metric: {metric}")

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

        self.model = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit UMAP and transform embeddings.

        Args:
            embeddings: Array of shape [n_samples, n_features]

        Returns:
            Reduced array of shape [n_samples, n_components]
        """
        try:
            import umap
        except ImportError:
            logger.error("UMAP not installed. Install with: pip install umap-learn")
            raise

        logger.info(f"Reducing {embeddings.shape} to {self.n_components}D with UMAP")

        # Adjust n_neighbors if too high for dataset
        n_samples = embeddings.shape[0]
        effective_neighbors = min(self.n_neighbors, n_samples - 1)
        if effective_neighbors < self.n_neighbors:
            logger.warning(
                f"Reducing n_neighbors from {self.n_neighbors} to {effective_neighbors} "
                f"for {n_samples} samples"
            )

        self.model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=effective_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )

        reduced = self.model.fit_transform(embeddings)

        logger.info(f"UMAP complete")

        return reduced

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted UMAP.

        Args:
            embeddings: Array of shape [n_samples, n_features]

        Returns:
            Reduced array of shape [n_samples, n_components]
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_transform first.")
        return self.model.transform(embeddings)






