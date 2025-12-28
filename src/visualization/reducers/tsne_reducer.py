"""t-SNE dimensionality reducer."""
import numpy as np
from sklearn.manifold import TSNE
from typing import Optional, Union

from src.utils.logging_utils import get_logger
from src.config.registry import REDUCER_REGISTRY

logger = get_logger("visualization.reducers.tsne")


@REDUCER_REGISTRY.register("tsne")
class TSNEReducer:
    """t-SNE dimensionality reduction."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = "auto",
        n_iter: int = 1000,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize the reducer.

        Args:
            n_components: Number of output dimensions
            perplexity: Perplexity parameter (related to number of neighbors)
            learning_rate: Learning rate ("auto" or float)
            n_iter: Number of iterations
            random_state: Random seed
            **kwargs: Additional arguments (ignored)
        """
        logger.debug(f"Initializing TSNEReducer")
        logger.debug(f"  n_components: {n_components}")
        logger.debug(f"  perplexity: {perplexity}")
        logger.debug(f"  learning_rate: {learning_rate}")
        logger.debug(f"  n_iter: {n_iter}")

        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[TSNE] = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit t-SNE and transform embeddings.

        Args:
            embeddings: Array of shape [n_samples, n_features]

        Returns:
            Reduced array of shape [n_samples, n_components]
        """
        logger.info(f"Reducing {embeddings.shape} to {self.n_components}D with t-SNE")

        # Adjust perplexity if too high for dataset
        n_samples = embeddings.shape[0]
        effective_perplexity = min(self.perplexity, (n_samples - 1) / 3)
        if effective_perplexity < self.perplexity:
            logger.warning(
                f"Reducing perplexity from {self.perplexity} to {effective_perplexity} "
                f"for {n_samples} samples"
            )

        self.model = TSNE(
            n_components=self.n_components,
            perplexity=effective_perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.n_iter,  # Changed from n_iter to max_iter for sklearn >= 1.5
            random_state=self.random_state,
        )

        reduced = self.model.fit_transform(embeddings)

        logger.info(f"t-SNE complete. KL divergence: {self.model.kl_divergence_:.4f}")

        return reduced

