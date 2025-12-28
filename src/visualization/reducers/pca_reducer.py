"""PCA dimensionality reducer."""
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional

from src.utils.logging_utils import get_logger
from src.config.registry import REDUCER_REGISTRY

logger = get_logger("visualization.reducers.pca")


@REDUCER_REGISTRY.register("pca")
class PCAReducer:
    """PCA dimensionality reduction."""

    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the reducer.

        Args:
            n_components: Number of output dimensions
            random_state: Random seed (optional)
            **kwargs: Additional arguments (ignored)
        """
        logger.debug(f"Initializing PCAReducer")
        logger.debug(f"  n_components: {n_components}")

        self.n_components = n_components
        self.random_state = random_state

        self.model: Optional[PCA] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit PCA and transform embeddings.

        Args:
            embeddings: Array of shape [n_samples, n_features]

        Returns:
            Reduced array of shape [n_samples, n_components]
        """
        logger.info(f"Reducing {embeddings.shape} to {self.n_components}D with PCA")

        self.model = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        )

        reduced = self.model.fit_transform(embeddings)
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_

        total_var = sum(self.explained_variance_ratio_)
        logger.info(f"Explained variance: {total_var*100:.1f}%")
        for i, var in enumerate(self.explained_variance_ratio_):
            logger.debug(f"  PC{i+1}: {var*100:.1f}%")

        return reduced

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using fitted PCA.

        Args:
            embeddings: Array of shape [n_samples, n_features]

        Returns:
            Reduced array of shape [n_samples, n_components]
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_transform first.")
        return self.model.transform(embeddings)

