"""Tests for configuration schema validation."""
import pytest
from pydantic import ValidationError

from src.config.schema import (
    ClusteringConfig,
    Config,
    DatasetConfig,
    GOEnrichmentConfig,
    ModelConfig,
    SamplingConfig,
    VisualizationConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_valid_huggingface_config(self):
        """Valid HuggingFace config should work."""
        config = DatasetConfig(source="huggingface", hf_dataset_name="test/dataset")
        assert config.source == "huggingface"
        assert config.hf_dataset_name == "test/dataset"

    def test_valid_fasta_config(self):
        """Valid FASTA config should work."""
        config = DatasetConfig(source="fasta", file_path="/path/to/file.fasta")
        assert config.source == "fasta"

    def test_valid_csv_config(self):
        """Valid CSV config should work."""
        config = DatasetConfig(source="csv", file_path="/path/to/file.csv")
        assert config.source == "csv"

    def test_invalid_source_raises(self):
        """Invalid source should raise ValidationError."""
        with pytest.raises(ValidationError):
            DatasetConfig(source="invalid")

    def test_min_length_must_be_positive(self):
        """min_sequence_length must be >= 1."""
        with pytest.raises(ValidationError):
            DatasetConfig(min_sequence_length=0)

        with pytest.raises(ValidationError):
            DatasetConfig(min_sequence_length=-1)

    def test_max_length_must_be_positive(self):
        """max_sequence_length must be >= 1."""
        with pytest.raises(ValidationError):
            DatasetConfig(max_sequence_length=0)

    def test_default_values(self):
        """Default values should be set correctly."""
        config = DatasetConfig()
        assert config.source == "huggingface"
        assert config.streaming is True
        assert config.filter_invalid is True
        assert config.min_sequence_length == 50

    def test_ambiguous_handling_options(self):
        """All ambiguous handling options should be valid."""
        for option in ["keep", "remove", "replace"]:
            config = DatasetConfig(ambiguous_handling=option)
            assert config.ambiguous_handling == option


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_all_esm2_variants_valid(self):
        """All ESM-2 variants should be valid."""
        variants = [
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
            "facebook/esm2_t33_650M_UR50D",
            "facebook/esm2_t36_3B_UR50D",
        ]
        for variant in variants:
            config = ModelConfig(model_name=variant)
            assert config.model_name == variant

    def test_invalid_dtype_raises(self):
        """Invalid dtype should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(dtype="invalid")

    def test_valid_dtypes(self):
        """All valid dtypes should work."""
        for dtype in ["float32", "float16", "bfloat16"]:
            config = ModelConfig(dtype=dtype)
            assert config.dtype == dtype

    def test_batch_size_must_be_positive(self):
        """batch_size must be >= 1."""
        with pytest.raises(ValidationError):
            ModelConfig(batch_size=0)

    def test_layer_ids_cannot_be_empty(self):
        """layer_ids cannot be empty list."""
        with pytest.raises(ValidationError):
            ModelConfig(layer_ids=[])

    def test_layer_ids_must_be_non_negative(self):
        """layer_ids must be non-negative."""
        with pytest.raises(ValidationError):
            ModelConfig(layer_ids=[-1])

    def test_hook_type_options(self):
        """All hook type options should be valid."""
        for hook_type in ["mlp", "attention", "residual", "all"]:
            config = ModelConfig(hook_type=hook_type)
            assert config.hook_type == hook_type

    def test_aggregation_options(self):
        """All aggregation options should be valid."""
        for agg in ["mean", "max", "last", "first", "all"]:
            config = ModelConfig(aggregation=agg)
            assert config.aggregation == agg


class TestClusteringConfig:
    """Tests for ClusteringConfig."""

    def test_all_algorithms_valid(self):
        """All clustering algorithms should be valid."""
        for algo in ["kmeans", "hdbscan", "spectral", "agglomerative", "dbscan"]:
            config = ClusteringConfig(algorithm=algo)
            assert config.algorithm == algo

    def test_n_clusters_must_be_positive(self):
        """n_clusters must be >= 1."""
        with pytest.raises(ValidationError):
            ClusteringConfig(n_clusters=0)

    def test_default_n_clusters(self):
        """Default n_clusters should be 5."""
        config = ClusteringConfig()
        assert config.n_clusters == 5

    def test_kmeans_parameters(self):
        """KMeans-specific parameters should work."""
        config = ClusteringConfig(
            algorithm="kmeans",
            kmeans_n_init=20,
            kmeans_max_iter=500,
        )
        assert config.kmeans_n_init == 20
        assert config.kmeans_max_iter == 500

    def test_hdbscan_parameters(self):
        """HDBSCAN-specific parameters should work."""
        config = ClusteringConfig(
            algorithm="hdbscan",
            hdbscan_min_cluster_size=10,
            hdbscan_min_samples=5,
        )
        assert config.hdbscan_min_cluster_size == 10
        assert config.hdbscan_min_samples == 5

    def test_spectral_parameters(self):
        """Spectral-specific parameters should work."""
        config = ClusteringConfig(
            algorithm="spectral",
            spectral_affinity="nearest_neighbors",
            spectral_n_neighbors=20,
        )
        assert config.spectral_affinity == "nearest_neighbors"
        assert config.spectral_n_neighbors == 20

    def test_agglomerative_linkage_options(self):
        """All agglomerative linkage options should be valid."""
        for linkage in ["ward", "complete", "average", "single"]:
            config = ClusteringConfig(agglomerative_linkage=linkage)
            assert config.agglomerative_linkage == linkage


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_reduction_methods(self):
        """All reduction methods should be valid."""
        for method in ["umap", "tsne", "pca"]:
            config = VisualizationConfig(reduction_method=method)
            assert config.reduction_method == method

    def test_dpi_must_be_reasonable(self):
        """DPI must be >= 50."""
        with pytest.raises(ValidationError):
            VisualizationConfig(dpi=10)

    def test_umap_parameters(self):
        """UMAP parameters should work."""
        config = VisualizationConfig(
            reduction_method="umap",
            umap_n_neighbors=30,
            umap_min_dist=0.05,
        )
        assert config.umap_n_neighbors == 30
        assert config.umap_min_dist == 0.05

    def test_tsne_parameters(self):
        """t-SNE parameters should work."""
        config = VisualizationConfig(
            reduction_method="tsne",
            tsne_perplexity=50.0,
            tsne_n_iter=2000,
        )
        assert config.tsne_perplexity == 50.0
        assert config.tsne_n_iter == 2000

    def test_save_formats(self):
        """Save formats should accept list of strings."""
        config = VisualizationConfig(save_formats=["png", "pdf", "svg"])
        assert "png" in config.save_formats
        assert "pdf" in config.save_formats


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_percentile_range_valid(self):
        """Percentiles must be between 0 and 1."""
        config = SamplingConfig(start_percentile=0.5, end_percentile=0.9)
        assert config.start_percentile == 0.5
        assert config.end_percentile == 0.9

    def test_percentile_out_of_range_raises(self):
        """Percentiles outside [0, 1] should raise."""
        with pytest.raises(ValidationError):
            SamplingConfig(start_percentile=1.5)

        with pytest.raises(ValidationError):
            SamplingConfig(end_percentile=-0.1)

    def test_default_values(self):
        """Default values should be set correctly."""
        config = SamplingConfig()
        assert config.start_percentile == 0.99
        assert config.end_percentile == 1.0
        assert config.n_samples == 1000


class TestGOEnrichmentConfig:
    """Tests for GOEnrichmentConfig."""

    def test_p_value_threshold_range(self):
        """p_value_threshold must be in (0, 1]."""
        config = GOEnrichmentConfig(p_value_threshold=0.05)
        assert config.p_value_threshold == 0.05

        with pytest.raises(ValidationError):
            GOEnrichmentConfig(p_value_threshold=0.0)

        with pytest.raises(ValidationError):
            GOEnrichmentConfig(p_value_threshold=1.5)

    def test_correction_methods(self):
        """All correction methods should be valid."""
        for method in ["bonferroni", "fdr_bh", "none"]:
            config = GOEnrichmentConfig(correction_method=method)
            assert config.correction_method == method

    def test_default_namespaces(self):
        """Default namespaces should include all three GO domains."""
        config = GOEnrichmentConfig()
        assert "biological_process" in config.namespaces
        assert "molecular_function" in config.namespaces
        assert "cellular_component" in config.namespaces


class TestConfig:
    """Tests for master Config class."""

    def test_default_config_valid(self):
        """Default config should be valid."""
        config = Config()
        assert config.experiment_name == "default"
        assert config.random_seed == 42

    def test_nested_configs_accessible(self):
        """Nested configs should be accessible."""
        config = Config()
        assert hasattr(config, "dataset")
        assert hasattr(config, "model")
        assert hasattr(config, "clustering")
        assert hasattr(config, "visualization")

    def test_extra_fields_forbidden(self):
        """Unknown fields should raise error."""
        with pytest.raises(ValidationError):
            Config(unknown_field="value")

    def test_to_dict_method(self):
        """to_dict() should return dictionary."""
        config = Config()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "experiment_name" in d
        assert "dataset" in d

    def test_get_output_path(self):
        """get_output_path() should return Path object."""
        config = Config(output_dir="test_output")
        path = config.get_output_path()
        from pathlib import Path

        assert isinstance(path, Path)
        assert str(path) == "test_output"

    def test_full_config_from_dict(self, default_config_dict):
        """Should create config from full dictionary."""
        config = Config(**default_config_dict)
        assert config.experiment_name == "test_experiment"
        assert config.dataset.max_samples == 100
        assert config.model.layer_ids == [3]






