"""Base clustering functionality for PRISM-Bio."""
import numpy as np
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger, LogContext
from src.config.registry import CLUSTERING_REGISTRY

logger = get_logger("analysis.clustering.base")


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 5,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """Cluster embeddings using specified method.

    Args:
        embeddings: Array of shape [n_samples, n_features]
        method: Clustering method ("kmeans", "hdbscan", "spectral", "agglomerative", "dbscan")
        n_clusters: Number of clusters (for methods that require it)
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to the clusterer

    Returns:
        Array of cluster labels [n_samples]

    Raises:
        ValueError: If method is not supported
    """
    logger.info(f"Clustering {embeddings.shape[0]} embeddings using {method}")
    logger.debug(f"  Embedding shape: {embeddings.shape}")
    logger.debug(f"  n_clusters: {n_clusters}")
    logger.debug(f"  seed: {seed}")

    with LogContext(logger, f"Clustering with {method}", n_samples=embeddings.shape[0]):
        if method == "kmeans":
            from src.analysis.clustering.kmeans import KMeansClusterer
            clusterer = KMeansClusterer(n_clusters=n_clusters, random_seed=seed, **kwargs)
            labels = clusterer.fit_predict(embeddings)

        elif method == "hdbscan":
            from src.analysis.clustering.hdbscan_cluster import HDBSCANClusterer
            clusterer = HDBSCANClusterer(random_seed=seed, **kwargs)
            labels = clusterer.fit_predict(embeddings)

        elif method == "spectral":
            from sklearn.cluster import SpectralClustering
            logger.debug("Using sklearn SpectralClustering")
            clusterer = SpectralClustering(
                n_clusters=n_clusters,
                random_state=seed,
                affinity=kwargs.get("affinity", "rbf"),
                n_neighbors=kwargs.get("n_neighbors", 10),
            )
            labels = clusterer.fit_predict(embeddings)

        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            logger.debug("Using sklearn AgglomerativeClustering")
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=kwargs.get("linkage", "ward"),
            )
            labels = clusterer.fit_predict(embeddings)

        elif method == "dbscan":
            from sklearn.cluster import DBSCAN
            logger.debug("Using sklearn DBSCAN")
            clusterer = DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5),
            )
            labels = clusterer.fit_predict(embeddings)

        else:
            raise ValueError(
                f"Unknown clustering method: {method}. "
                f"Supported: kmeans, hdbscan, spectral, agglomerative, dbscan"
            )

    # Log results
    unique_labels = np.unique(labels)
    n_clusters_found = len(unique_labels[unique_labels >= 0])
    n_noise = np.sum(labels == -1) if -1 in labels else 0

    logger.info(f"Found {n_clusters_found} clusters")
    if n_noise > 0:
        logger.info(f"  Noise points: {n_noise}")

    return labels


def compute_cluster_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """Compute statistics for each cluster.

    Args:
        embeddings: Array of shape [n_samples, n_features]
        labels: Array of cluster labels [n_samples]

    Returns:
        Dictionary mapping cluster_id to statistics dict
    """
    logger.debug(f"Computing statistics for clusters")

    stats = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            # Noise points in HDBSCAN/DBSCAN
            continue

        mask = labels == label
        cluster_embeddings = embeddings[mask]

        centroid = np.mean(cluster_embeddings, axis=0)

        # Compute distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        stats[label] = {
            "size": int(np.sum(mask)),
            "centroid": centroid,
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "max_distance": float(np.max(distances)),
        }

        logger.debug(
            f"Cluster {label}: size={stats[label]['size']}, "
            f"mean_dist={stats[label]['mean_distance']:.4f}"
        )

    return stats


def compute_cluster_similarity(
    stats: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute pairwise cosine similarity between cluster centroids.

    This measures how similar clusters are to each other. Higher similarity
    between clusters suggests the neuron responds to similar patterns (less
    polysemantic), while lower similarity indicates more distinct response
    patterns (more polysemantic).

    Args:
        stats: Dictionary mapping cluster_id to statistics dict (from compute_cluster_statistics).
               Each entry must contain a 'centroid' key with the cluster centroid array.

    Returns:
        Dictionary with:
        - "similarity_matrix": 2D array of pairwise cosine similarities [n_clusters x n_clusters]
        - "cluster_ids": List of cluster IDs (for matrix indexing)
        - "per_cluster_avg": Dict mapping cluster_id -> avg similarity to other clusters
        - "overall_avg": Mean off-diagonal similarity (polysemanticity score)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("Computing inter-cluster cosine similarity")

    # Extract cluster IDs and centroids
    cluster_ids = sorted(stats.keys())
    n_clusters = len(cluster_ids)

    if n_clusters < 2:
        logger.warning("Need at least 2 clusters to compute similarity")
        return {
            "similarity_matrix": np.array([[1.0]]) if n_clusters == 1 else np.array([]),
            "cluster_ids": cluster_ids,
            "per_cluster_avg": {cid: 1.0 for cid in cluster_ids},
            "overall_avg": 1.0,
        }

    # Stack centroids into a matrix [n_clusters, n_features]
    centroids = np.vstack([stats[cid]["centroid"] for cid in cluster_ids])

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(centroids)

    # Compute per-cluster average similarity (excluding self-similarity)
    per_cluster_avg = {}
    for i, cid in enumerate(cluster_ids):
        # Get similarities to all other clusters (exclude diagonal)
        other_similarities = [similarity_matrix[i, j] for j in range(n_clusters) if j != i]
        per_cluster_avg[cid] = float(np.mean(other_similarities))

    # Compute overall average (mean of off-diagonal elements)
    # This is the polysemanticity score: lower = more distinct clusters
    off_diagonal_mask = ~np.eye(n_clusters, dtype=bool)
    overall_avg = float(np.mean(similarity_matrix[off_diagonal_mask]))

    logger.info(f"Inter-cluster similarity computed for {n_clusters} clusters")
    logger.info(f"  Overall average similarity: {overall_avg:.4f}")
    for cid in cluster_ids:
        logger.debug(f"  Cluster {cid} avg similarity to others: {per_cluster_avg[cid]:.4f}")

    return {
        "similarity_matrix": similarity_matrix.tolist(),  # Convert to list for JSON serialization
        "cluster_ids": cluster_ids,
        "per_cluster_avg": per_cluster_avg,
        "overall_avg": overall_avg,
    }


def get_representative_samples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    n_per_cluster: int = 10,
) -> Dict[int, List[Dict[str, Any]]]:
    """Get representative samples for each cluster.

    Selects samples closest to cluster centroid.

    Args:
        embeddings: Array of shape [n_samples, n_features]
        labels: Array of cluster labels [n_samples]
        metadata: List of metadata dicts for each sample
        n_per_cluster: Number of representatives per cluster

    Returns:
        Dictionary mapping cluster_id to list of sample metadata
    """
    logger.debug(f"Getting {n_per_cluster} representatives per cluster")

    representatives = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label
        indices = np.where(mask)[0]
        cluster_embeddings = embeddings[mask]

        # Compute centroid
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find closest samples
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        sorted_idx = np.argsort(distances)[:n_per_cluster]

        # Get metadata for closest samples
        representatives[label] = [metadata[indices[i]] for i in sorted_idx]

        logger.debug(f"Cluster {label}: selected {len(representatives[label])} representatives")

    return representatives






