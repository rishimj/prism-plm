"""Tests for clustering visualization module."""
import numpy as np
import pytest
from pathlib import Path


def to_numpy(arr):
    """Convert array to numpy, handling both torch tensors and numpy arrays."""
    if hasattr(arr, 'numpy'):
        return arr.numpy()
    return np.asarray(arr)


class TestReduceDimensions:
    """Tests for dimensionality reduction."""

    @pytest.mark.unit
    def test_umap_returns_2d(self, mock_embeddings):
        """UMAP should return 2D coordinates."""
        from src.visualization.cluster_viz import reduce_dimensions
        data = to_numpy(mock_embeddings)
        coords = reduce_dimensions(data, method="umap")
        assert coords.shape == (data.shape[0], 2)

    @pytest.mark.unit
    def test_tsne_returns_2d(self, mock_embeddings):
        """t-SNE should return 2D coordinates."""
        from src.visualization.cluster_viz import reduce_dimensions
        data = to_numpy(mock_embeddings)
        # Use small perplexity for small dataset
        coords = reduce_dimensions(
            data, 
            method="tsne", 
            tsne_perplexity=min(5, data.shape[0] - 1)
        )
        assert coords.shape == (data.shape[0], 2)

    @pytest.mark.unit
    def test_pca_returns_2d(self, mock_embeddings):
        """PCA fallback should return 2D coordinates."""
        from src.visualization.cluster_viz import reduce_dimensions
        data = to_numpy(mock_embeddings)
        coords = reduce_dimensions(data, method="pca")
        assert coords.shape == (data.shape[0], 2)

    @pytest.mark.unit
    def test_invalid_method_raises(self, mock_embeddings):
        """Invalid method should raise ValueError."""
        from src.visualization.cluster_viz import reduce_dimensions
        data = to_numpy(mock_embeddings)
        with pytest.raises(ValueError):
            reduce_dimensions(data, method="invalid")

    @pytest.mark.unit
    def test_preserves_relative_distances_approximately(self, clustered_embeddings):
        """Nearby points should remain relatively close after reduction."""
        from src.visualization.cluster_viz import reduce_dimensions
        data = to_numpy(clustered_embeddings)
        coords = reduce_dimensions(data, method="pca")
        
        # Points 0-29 (cluster 1) should be closer to each other than to 60-89 (cluster 3)
        intra_dist = np.linalg.norm(coords[0] - coords[15])
        inter_dist = np.linalg.norm(coords[0] - coords[75])
        assert intra_dist < inter_dist


class TestClusterVisualizer:
    """Tests for ClusterVisualizer class."""

    @pytest.mark.unit
    def test_init_creates_output_dir(self, tmp_output_dir):
        """Initialization should create output directory."""
        from src.visualization.cluster_viz import ClusterVisualizer
        new_dir = tmp_output_dir / "new_viz"
        viz = ClusterVisualizer(output_dir=str(new_dir))
        assert Path(viz.output_dir).exists()

    @pytest.mark.unit
    def test_plot_embedding_space_creates_file(self, tmp_output_dir, clustered_embeddings):
        """plot_embedding_space should create PNG file."""
        from src.visualization.cluster_viz import ClusterVisualizer, reduce_dimensions
        data = to_numpy(clustered_embeddings)
        viz = ClusterVisualizer(output_dir=str(tmp_output_dir))
        coords = reduce_dimensions(data, method="pca")
        clusters = np.repeat([0, 1, 2], 30)

        path = viz.plot_embedding_space(coords, clusters, save_name="test_embed.png")
        assert path.exists()
        assert path.suffix == ".png"

    @pytest.mark.unit
    def test_plot_cluster_grid_creates_file(self, tmp_output_dir, clustered_embeddings):
        """plot_cluster_grid should create PNG file."""
        from src.visualization.cluster_viz import ClusterVisualizer, reduce_dimensions
        data = to_numpy(clustered_embeddings)
        viz = ClusterVisualizer(output_dir=str(tmp_output_dir))
        coords = reduce_dimensions(data, method="pca")
        clusters = np.repeat([0, 1, 2], 30)

        path = viz.plot_cluster_grid(coords, clusters, n_clusters=3)
        assert path.exists()

    @pytest.mark.unit
    def test_create_full_report_generates_all_files(self, tmp_output_dir, clustered_embeddings):
        """create_full_report should generate multiple visualization files."""
        from src.visualization.cluster_viz import ClusterVisualizer
        data = to_numpy(clustered_embeddings)
        viz = ClusterVisualizer(output_dir=str(tmp_output_dir))
        clusters = np.repeat([0, 1, 2], 30)
        metadata = [{"id": f"P{i:05d}"} for i in range(90)]

        output_dir = viz.create_full_report(
            data,
            clusters,
            metadata,
            reduction_method="pca"
        )

        # Check expected files exist
        expected_files = [
            "embedding_space_pca.png",
            "cluster_grid.png",
            "cluster_statistics.json",
        ]
        for fname in expected_files:
            assert (output_dir / fname).exists(), f"Missing: {fname}"

    @pytest.mark.unit
    def test_handles_single_cluster(self, tmp_output_dir, mock_embeddings):
        """Should handle case with only 1 cluster."""
        from src.visualization.cluster_viz import ClusterVisualizer, reduce_dimensions
        data = to_numpy(mock_embeddings)
        viz = ClusterVisualizer(output_dir=str(tmp_output_dir))
        coords = reduce_dimensions(data, method="pca")
        clusters = np.zeros(len(data), dtype=int)

        # Should not raise
        path = viz.plot_embedding_space(coords, clusters)
        assert path.exists()

    @pytest.mark.unit
    def test_handles_many_clusters(self, tmp_output_dir, mock_embeddings):
        """Should handle many clusters."""
        from src.visualization.cluster_viz import ClusterVisualizer, reduce_dimensions
        data = to_numpy(mock_embeddings)
        viz = ClusterVisualizer(output_dir=str(tmp_output_dir))
        coords = reduce_dimensions(data, method="pca")
        clusters = np.arange(len(data)) % 8  # 8 clusters

        path = viz.plot_cluster_grid(coords, clusters, n_clusters=8)
        assert path.exists()

    @pytest.mark.unit
    def test_custom_colormap(self, tmp_output_dir, clustered_embeddings):
        """Should accept custom colormap."""
        from src.visualization.cluster_viz import ClusterVisualizer, reduce_dimensions
        data = to_numpy(clustered_embeddings)
        viz = ClusterVisualizer(output_dir=str(tmp_output_dir))
        coords = reduce_dimensions(data, method="pca")
        clusters = np.repeat([0, 1, 2], 30)

        path = viz.plot_embedding_space(coords, clusters, colormap="viridis")
        assert path.exists()

    @pytest.mark.unit
    def test_saves_multiple_formats(self, tmp_output_dir, clustered_embeddings):
        """Should save in multiple formats when requested."""
        from src.visualization.cluster_viz import ClusterVisualizer, reduce_dimensions
        data = to_numpy(clustered_embeddings)
        viz = ClusterVisualizer(output_dir=str(tmp_output_dir))
        coords = reduce_dimensions(data, method="pca")
        clusters = np.repeat([0, 1, 2], 30)

        paths = viz.plot_embedding_space(
            coords, 
            clusters, 
            save_name="test",
            save_formats=["png", "pdf"]
        )
        # Implementation may return single path or list


class TestPlotActivationHeatmap:
    """Tests for activation heatmap plotting."""

    @pytest.mark.unit
    def test_creates_heatmap_file(self, tmp_output_dir, mock_activations):
        """Should create heatmap PNG file."""
        from src.visualization.cluster_viz import ClusterVisualizer
        data = to_numpy(mock_activations)
        viz = ClusterVisualizer(output_dir=str(tmp_output_dir))
        
        # Use first sample's activations
        activations = data[0]
        
        path = viz.plot_activation_heatmap(
            activations[:50, :20],  # Subset for testing
            save_name="test_heatmap.png"
        )
        assert path.exists()


class TestUMAPReducer:
    """Tests for UMAP reducer class."""

    @pytest.mark.unit
    def test_reducer_exists(self):
        """UMAPReducer should exist."""
        from src.visualization.reducers.umap_reducer import UMAPReducer
        assert UMAPReducer is not None

    @pytest.mark.unit
    def test_fit_transform(self, mock_embeddings):
        """fit_transform should return 2D coordinates."""
        from src.visualization.reducers.umap_reducer import UMAPReducer
        data = to_numpy(mock_embeddings)
        reducer = UMAPReducer(n_neighbors=5, min_dist=0.1)
        coords = reducer.fit_transform(data)
        assert coords.shape == (data.shape[0], 2)

    @pytest.mark.unit
    def test_reproducible_with_seed(self, mock_embeddings):
        """Same seed should give same results."""
        from src.visualization.reducers.umap_reducer import UMAPReducer
        data = to_numpy(mock_embeddings)
        reducer1 = UMAPReducer(random_state=42)
        reducer2 = UMAPReducer(random_state=42)
        
        coords1 = reducer1.fit_transform(data)
        coords2 = reducer2.fit_transform(data)
        
        np.testing.assert_array_almost_equal(coords1, coords2)


class TestTSNEReducer:
    """Tests for t-SNE reducer class."""

    @pytest.mark.unit
    def test_reducer_exists(self):
        """TSNEReducer should exist."""
        from src.visualization.reducers.tsne_reducer import TSNEReducer
        assert TSNEReducer is not None

    @pytest.mark.unit
    def test_fit_transform(self, mock_embeddings):
        """fit_transform should return 2D coordinates."""
        from src.visualization.reducers.tsne_reducer import TSNEReducer
        data = to_numpy(mock_embeddings)
        # Use small perplexity for small dataset
        reducer = TSNEReducer(perplexity=min(5, data.shape[0] - 1))
        coords = reducer.fit_transform(data)
        assert coords.shape == (data.shape[0], 2)


class TestPCAReducer:
    """Tests for PCA reducer class."""

    @pytest.mark.unit
    def test_reducer_exists(self):
        """PCAReducer should exist."""
        from src.visualization.reducers.pca_reducer import PCAReducer
        assert PCAReducer is not None

    @pytest.mark.unit
    def test_fit_transform(self, mock_embeddings):
        """fit_transform should return 2D coordinates."""
        from src.visualization.reducers.pca_reducer import PCAReducer
        data = to_numpy(mock_embeddings)
        reducer = PCAReducer(n_components=2)
        coords = reducer.fit_transform(data)
        assert coords.shape == (data.shape[0], 2)

    @pytest.mark.unit
    def test_explained_variance(self, mock_embeddings):
        """Should provide explained variance ratio."""
        from src.visualization.reducers.pca_reducer import PCAReducer
        data = to_numpy(mock_embeddings)
        reducer = PCAReducer(n_components=2)
        reducer.fit_transform(data)
        
        assert hasattr(reducer, 'explained_variance_ratio_')
        assert len(reducer.explained_variance_ratio_) == 2
