"""Pydantic configuration schema - fully typed and validated."""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================


class DatasetConfig(BaseModel):
    """Configuration for protein dataset loading."""

    # Dataset source
    source: Literal["huggingface", "fasta", "csv", "custom"] = "huggingface"

    # HuggingFace options
    hf_dataset_name: Optional[str] = "PolyAI/uniref50"
    hf_subset: Optional[str] = None
    hf_split: str = "train"

    # File options (for fasta/csv)
    file_path: Optional[str] = None
    sequence_column: str = "sequence"
    id_column: str = "id"

    # Processing options
    streaming: bool = True
    max_samples: Optional[int] = None
    min_sequence_length: int = 50
    max_sequence_length: int = 1024
    filter_invalid: bool = True
    ambiguous_handling: Literal["keep", "remove", "replace"] = "keep"

    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True

    @field_validator("min_sequence_length")
    @classmethod
    def validate_min_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_sequence_length must be >= 1")
        return v

    @field_validator("max_sequence_length")
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_sequence_length must be >= 1")
        return v


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================


class ModelConfig(BaseModel):
    """Configuration for protein language model."""

    # Model selection
    model_type: Literal["esm2", "prottrans", "custom"] = "esm2"
    model_name: str = "facebook/esm2_t36_3B_UR50D"

    # Alternative model names (for easy switching):
    # ESM-2 variants:
    #   facebook/esm2_t6_8M_UR50D      - 8M params, 6 layers
    #   facebook/esm2_t12_35M_UR50D    - 35M params, 12 layers
    #   facebook/esm2_t30_150M_UR50D   - 150M params, 30 layers
    #   facebook/esm2_t33_650M_UR50D   - 650M params, 33 layers
    #   facebook/esm2_t36_3B_UR50D     - 3B params, 36 layers

    # Hook configuration
    hook_type: Literal["mlp", "attention", "residual", "all"] = "all"
    layer_ids: List[int] = Field(default_factory=lambda: [18])
    unit_ids: Optional[List[int]] = None  # None = all units

    # Inference options
    batch_size: int = 8
    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"

    # Activation aggregation
    aggregation: Literal["mean", "max", "last", "first", "all"] = "mean"

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v

    @field_validator("layer_ids")
    @classmethod
    def validate_layer_ids(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("layer_ids cannot be empty")
        if any(x < 0 for x in v):
            raise ValueError("layer_ids must be non-negative")
        return v


# ============================================================================
# CLUSTERING CONFIGURATION
# ============================================================================


class ClusteringConfig(BaseModel):
    """Configuration for clustering algorithm."""

    # Algorithm selection
    algorithm: Literal[
        "kmeans", "hdbscan", "spectral", "agglomerative", "dbscan"
    ] = "kmeans"

    # Common parameters
    n_clusters: int = 5
    random_seed: int = 42

    # KMeans specific
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300

    # HDBSCAN specific
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: Optional[int] = None
    hdbscan_cluster_selection_epsilon: float = 0.0

    # Spectral specific
    spectral_affinity: Literal["rbf", "nearest_neighbors"] = "rbf"
    spectral_n_neighbors: int = 10

    # Agglomerative specific
    agglomerative_linkage: Literal["ward", "complete", "average", "single"] = "ward"
    agglomerative_distance_threshold: Optional[float] = None

    # DBSCAN specific
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5

    @field_validator("n_clusters")
    @classmethod
    def validate_n_clusters(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_clusters must be >= 1")
        return v


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================


class VisualizationConfig(BaseModel):
    """Configuration for visualization."""

    # Dimensionality reduction
    reduction_method: Literal["umap", "tsne", "pca"] = "umap"

    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    umap_random_state: int = 42

    # t-SNE parameters
    tsne_perplexity: float = 30.0
    tsne_learning_rate: Union[float, Literal["auto"]] = "auto"
    tsne_n_iter: int = 1000
    tsne_random_state: int = 42

    # PCA parameters
    pca_n_components: int = 2

    # Plot settings
    figure_width: int = 14
    figure_height: int = 10
    dpi: int = 300
    colormap: str = "tab10"
    point_size: int = 50
    alpha: float = 0.6

    # Output
    output_dir: str = "visualizations"
    save_formats: List[str] = Field(default_factory=lambda: ["png"])

    @field_validator("dpi")
    @classmethod
    def validate_dpi(cls, v: int) -> int:
        if v < 50:
            raise ValueError("dpi must be >= 50")
        return v


# ============================================================================
# SAMPLING CONFIGURATION
# ============================================================================


class SamplingConfig(BaseModel):
    """Configuration for percentile sampling."""

    start_percentile: float = 0.99
    end_percentile: float = 1.0
    n_samples: int = 1000
    activation_percentile: int = 90
    filter_positive_activations: bool = True
    max_cluster_size: int = 20

    @field_validator("start_percentile", "end_percentile")
    @classmethod
    def validate_percentile(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("percentile must be between 0 and 1")
        return v


# ============================================================================
# GO ENRICHMENT CONFIGURATION
# ============================================================================


class GOEnrichmentConfig(BaseModel):
    """Configuration for Gene Ontology enrichment."""

    enabled: bool = True
    obo_file: Optional[str] = None  # Auto-download if None
    annotation_file: Optional[str] = None

    # Statistical parameters
    p_value_threshold: float = 0.05
    correction_method: Literal["bonferroni", "fdr_bh", "none"] = "fdr_bh"
    min_genes: int = 3

    # GO namespaces to include
    namespaces: List[str] = Field(
        default_factory=lambda: [
            "biological_process",
            "molecular_function",
            "cellular_component",
        ]
    )

    @field_validator("p_value_threshold")
    @classmethod
    def validate_p_value(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("p_value_threshold must be between 0 and 1")
        return v


# ============================================================================
# TEXT GENERATION CONFIGURATION
# ============================================================================


class TextGenerationConfig(BaseModel):
    """Configuration for text/description generation."""

    generator_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    use_api: bool = False
    api_provider: Optional[Literal["openai", "gemini", "huggingface"]] = None

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True

    # Prompt configuration
    system_instruction: str = "assistant"


# ============================================================================
# MASTER CONFIGURATION
# ============================================================================


class Config(BaseModel):
    """Master configuration combining all components."""

    # Component configs
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    go_enrichment: GOEnrichmentConfig = Field(default_factory=GOEnrichmentConfig)
    text_generation: TextGenerationConfig = Field(default_factory=TextGenerationConfig)

    # Global settings
    experiment_name: str = "default"
    output_dir: str = "outputs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    random_seed: int = 42
    verbose: bool = True

    # Resource settings
    num_workers: int = 4
    use_gpu: bool = True

    model_config = {"extra": "forbid"}  # Raise error on unknown fields

    def get_output_path(self) -> Path:
        """Get the output directory path."""
        return Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

