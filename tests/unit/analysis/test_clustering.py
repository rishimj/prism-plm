"""Tests for clustering module."""
import numpy as np
import pytest


def to_numpy(arr):
    """Convert array to numpy, handling both torch tensors and numpy arrays."""
    if hasattr(arr, 'numpy'):
        return arr.numpy()
    return np.asarray(arr)


class TestClusterEmbeddings:
    """Tests for clustering algorithms."""

    @pytest.mark.unit
    def test_kmeans_returns_correct_shape(self, mock_embeddings):
        """KMeans should return cluster labels for each sample."""
        from src.analysis.clustering import cluster_embeddings
        data = to_numpy(mock_embeddings)
        labels = cluster_embeddings(data, method="kmeans", n_clusters=3)
        assert labels.shape == (data.shape[0],)

    @pytest.mark.unit
    def test_kmeans_returns_n_clusters(self, mock_embeddings):
        """KMeans should return at most n_clusters unique labels."""
        from src.analysis.clustering import cluster_embeddings
        data = to_numpy(mock_embeddings)
        labels = cluster_embeddings(data, method="kmeans", n_clusters=5)
        assert len(np.unique(labels)) <= 5

    @pytest.mark.unit
    def test_hdbscan_returns_labels(self, mock_embeddings):
        """HDBSCAN should return cluster labels (including -1 for noise)."""
        from src.analysis.clustering import cluster_embeddings
        data = to_numpy(mock_embeddings)
        labels = cluster_embeddings(data, method="hdbscan")
        assert labels.shape == (data.shape[0],)

    @pytest.mark.unit
    def test_finds_known_clusters(self, clustered_embeddings):
        """Should find the 3 known clusters in test data."""
        from src.analysis.clustering import cluster_embeddings
        data = to_numpy(clustered_embeddings)
        labels = cluster_embeddings(data, method="kmeans", n_clusters=3)
        
        # Each group of 30 should mostly be in same cluster
        cluster_0 = labels[:30]
        cluster_1 = labels[30:60]
        cluster_2 = labels[60:]

        # Most common label should dominate each group
        assert np.bincount(cluster_0).max() >= 25
        assert np.bincount(cluster_1).max() >= 25
        assert np.bincount(cluster_2).max() >= 25

    @pytest.mark.unit
    def test_reproducible_with_seed(self, mock_embeddings):
        """Same seed should give same results."""
        from src.analysis.clustering import cluster_embeddings
        data = to_numpy(mock_embeddings)
        labels1 = cluster_embeddings(data, method="kmeans", n_clusters=3, seed=42)
        labels2 = cluster_embeddings(data, method="kmeans", n_clusters=3, seed=42)
        np.testing.assert_array_equal(labels1, labels2)

    @pytest.mark.unit
    def test_invalid_method_raises(self, mock_embeddings):
        """Invalid clustering method should raise ValueError."""
        from src.analysis.clustering import cluster_embeddings
        data = to_numpy(mock_embeddings)
        with pytest.raises(ValueError):
            cluster_embeddings(data, method="invalid")

    @pytest.mark.unit
    def test_spectral_clustering(self, mock_embeddings):
        """Spectral clustering should work."""
        from src.analysis.clustering import cluster_embeddings
        data = to_numpy(mock_embeddings)
        labels = cluster_embeddings(data, method="spectral", n_clusters=3)
        assert labels.shape == (data.shape[0],)

    @pytest.mark.unit
    def test_agglomerative_clustering(self, mock_embeddings):
        """Agglomerative clustering should work."""
        from src.analysis.clustering import cluster_embeddings
        data = to_numpy(mock_embeddings)
        labels = cluster_embeddings(data, method="agglomerative", n_clusters=3)
        assert labels.shape == (data.shape[0],)


class TestClusterStatistics:
    """Tests for cluster statistics computation."""

    @pytest.mark.unit
    def test_returns_stats_for_each_cluster(self, clustered_embeddings):
        """Should return statistics for each cluster."""
        from src.analysis.clustering import compute_cluster_statistics
        data = to_numpy(clustered_embeddings)
        labels = np.repeat([0, 1, 2], 30)
        stats = compute_cluster_statistics(data, labels)
        assert len(stats) == 3

    @pytest.mark.unit
    def test_stats_include_size(self, clustered_embeddings):
        """Statistics should include cluster size."""
        from src.analysis.clustering import compute_cluster_statistics
        data = to_numpy(clustered_embeddings)
        labels = np.repeat([0, 1, 2], 30)
        stats = compute_cluster_statistics(data, labels)
        assert all('size' in s for s in stats.values())
        assert stats[0]['size'] == 30

    @pytest.mark.unit
    def test_stats_include_centroid(self, clustered_embeddings):
        """Statistics should include cluster centroid."""
        from src.analysis.clustering import compute_cluster_statistics
        data = to_numpy(clustered_embeddings)
        labels = np.repeat([0, 1, 2], 30)
        stats = compute_cluster_statistics(data, labels)
        assert all('centroid' in s for s in stats.values())
        assert stats[0]['centroid'].shape == (data.shape[1],)


class TestKMeansClusterer:
    """Tests for KMeansClusterer class."""

    @pytest.mark.unit
    def test_clusterer_exists(self):
        """KMeansClusterer should exist."""
        from src.analysis.clustering.kmeans import KMeansClusterer
        assert KMeansClusterer is not None

    @pytest.mark.unit
    def test_fit_predict(self, mock_embeddings):
        """fit_predict should return labels."""
        from src.analysis.clustering.kmeans import KMeansClusterer
        data = to_numpy(mock_embeddings)
        clusterer = KMeansClusterer(n_clusters=3)
        labels = clusterer.fit_predict(data)
        assert labels.shape == (data.shape[0],)

    @pytest.mark.unit
    def test_get_statistics(self, mock_embeddings):
        """get_statistics should return cluster info."""
        from src.analysis.clustering.kmeans import KMeansClusterer
        data = to_numpy(mock_embeddings)
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit_predict(data)
        stats = clusterer.get_statistics()
        assert 'n_clusters' in stats
        assert 'cluster_sizes' in stats


class TestHDBSCANClusterer:
    """Tests for HDBSCANClusterer class."""

    @pytest.mark.unit
    def test_clusterer_exists(self):
        """HDBSCANClusterer should exist."""
        from src.analysis.clustering.hdbscan_cluster import HDBSCANClusterer
        assert HDBSCANClusterer is not None

    @pytest.mark.unit
    def test_fit_predict(self, mock_embeddings):
        """fit_predict should return labels."""
        from src.analysis.clustering.hdbscan_cluster import HDBSCANClusterer
        data = to_numpy(mock_embeddings)
        clusterer = HDBSCANClusterer(min_cluster_size=2)
        labels = clusterer.fit_predict(data)
        assert labels.shape == (data.shape[0],)

    @pytest.mark.unit
    def test_handles_noise_points(self, mock_embeddings):
        """HDBSCAN may have noise points labeled as -1."""
        from src.analysis.clustering.hdbscan_cluster import HDBSCANClusterer
        data = to_numpy(mock_embeddings)
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        labels = clusterer.fit_predict(data)
        # -1 may or may not be present
        assert all(l >= -1 for l in labels)


class TestGetRepresentativeSamples:
    """Tests for getting representative samples from clusters."""

    @pytest.mark.unit
    def test_returns_samples_per_cluster(self, clustered_embeddings, valid_sequences):
        """Should return representative samples for each cluster."""
        from src.analysis.clustering import get_representative_samples
        data = to_numpy(clustered_embeddings)
        labels = np.repeat([0, 1, 2], 30)
        # Create metadata matching embeddings
        metadata = [{"id": f"seq_{i}", "sequence": f"SEQ{i}"} for i in range(90)]
        
        representatives = get_representative_samples(
            data, 
            labels, 
            metadata, 
            n_per_cluster=3
        )
        
        assert len(representatives) == 3  # 3 clusters
        for cluster_id, samples in representatives.items():
            assert len(samples) <= 3

    @pytest.mark.unit
    def test_respects_n_per_cluster(self, clustered_embeddings):
        """Should return at most n_per_cluster samples."""
        from src.analysis.clustering import get_representative_samples
        data = to_numpy(clustered_embeddings)
        labels = np.repeat([0, 1, 2], 30)
        metadata = [{"id": f"seq_{i}"} for i in range(90)]
        
        representatives = get_representative_samples(
            data,
            labels,
            metadata,
            n_per_cluster=5
        )
        
        for cluster_id, samples in representatives.items():
            assert len(samples) <= 5
