"""Shared pytest fixtures for all PRISM-Bio tests."""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# TOKENIZER FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def mock_esm_tokenizer():
    """Mock ESM-2 tokenizer for tests that don't need real tokenization."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 1
    tokenizer.cls_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 3

    def mock_call(sequence, max_length=1024, padding="max_length", truncation=True, return_tensors="pt"):
        seq_len = min(len(sequence) + 2, max_length)  # +2 for CLS and EOS
        input_ids = torch.zeros(1, max_length, dtype=torch.long)
        attention_mask = torch.zeros(1, max_length, dtype=torch.long)

        input_ids[0, 0] = 0  # CLS
        for i, char in enumerate(sequence[:max_length - 2]):
            input_ids[0, i + 1] = ord(char) % 20 + 4  # Fake amino acid tokens
        input_ids[0, seq_len - 1] = 2  # EOS
        attention_mask[0, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    return tokenizer


@pytest.fixture(scope="session")
def esm_tokenizer_real():
    """Real ESM-2 tokenizer (skip if not installed)."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    except Exception:
        pytest.skip("ESM-2 tokenizer not available")


# ============================================================================
# SEQUENCE FIXTURES
# ============================================================================


@pytest.fixture
def valid_sequences() -> List[Dict[str, str]]:
    """Valid protein sequences of various lengths."""
    return [
        {
            "id": "P12345",
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAAL",
        },
        {
            "id": "Q67890",
            "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        },
        {"id": "P11111", "sequence": "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS"},
        {"id": "O99999", "sequence": "A" * 100},  # Repetitive
        {"id": "P55555", "sequence": "ACDEFGHIKLMNPQRSTVWY"},  # All standard AAs
    ]


@pytest.fixture
def invalid_sequences() -> List[Dict[str, str]]:
    """Invalid or edge-case sequences."""
    return [
        {"id": "BAD001", "sequence": ""},  # Empty
        {"id": "BAD002", "sequence": "MXBZOU"},  # Non-standard AAs
        {"id": "BAD003", "sequence": "MKT123ABC"},  # Numbers
        {"id": "BAD004", "sequence": "mktay"},  # Lowercase
        {"id": "BAD005", "sequence": "MKT AYI"},  # Spaces
        {"id": "BAD006", "sequence": "A"},  # Too short
        {"id": "BAD007", "sequence": "A" * 5000},  # Very long
    ]


@pytest.fixture
def edge_case_sequences() -> List[Dict[str, str]]:
    """Edge cases for boundary testing."""
    return [
        {"id": "EDGE01", "sequence": "M"},  # Minimum possible
        {"id": "EDGE02", "sequence": "A" * 49},  # Just below default min_length=50
        {"id": "EDGE03", "sequence": "A" * 50},  # Exactly min_length
        {"id": "EDGE04", "sequence": "A" * 51},  # Just above min_length
        {"id": "EDGE05", "sequence": "A" * 1023},  # Just below default max_length
        {"id": "EDGE06", "sequence": "A" * 1024},  # Exactly max_length
        {"id": "EDGE07", "sequence": "A" * 1025},  # Just above max_length
    ]


@pytest.fixture
def sample_sequences(valid_sequences) -> List[Dict[str, str]]:
    """Alias for valid_sequences."""
    return valid_sequences


# ============================================================================
# DATASET FIXTURES
# ============================================================================


@pytest.fixture
def mock_hf_dataset(valid_sequences):
    """Mock HuggingFace Dataset."""
    pytest.importorskip("datasets")
    from datasets import Dataset

    return Dataset.from_list(valid_sequences)


@pytest.fixture
def mock_streaming_dataset(valid_sequences):
    """Mock streaming dataset iterator."""

    def generator():
        for seq in valid_sequences:
            yield seq

    return generator


# ============================================================================
# EMBEDDING/ACTIVATION FIXTURES
# ============================================================================


@pytest.fixture
def mock_embeddings():
    """Mock 2D embeddings for clustering/viz [n_samples, n_features]."""
    np.random.seed(42)
    # Return 2D array for clustering and visualization tests
    return np.random.randn(100, 64).astype(np.float32)


@pytest.fixture
def mock_embeddings_3d():
    """Mock ESM-2 embeddings [batch, seq_len, hidden_dim]."""
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        return torch.randn(8, 100, 320)  # Small hidden dim for tests
    else:
        np.random.seed(42)
        return np.random.randn(8, 100, 320).astype(np.float32)


@pytest.fixture
def mock_activations():
    """Mock layer activations [batch, seq_len, hidden_dim]."""
    np.random.seed(42)
    # Return 3D array for activation tests
    return np.random.randn(8, 100, 64).astype(np.float32)


@pytest.fixture
def mock_attention_weights():
    """Mock attention weights [batch, heads, seq_len, seq_len]."""
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        weights = torch.randn(8, 20, 100, 100)
        return torch.softmax(weights, dim=-1)
    else:
        np.random.seed(42)
        weights = np.random.randn(8, 20, 100, 100).astype(np.float32)
        # Simple softmax approximation
        exp_weights = np.exp(weights - np.max(weights, axis=-1, keepdims=True))
        return exp_weights / np.sum(exp_weights, axis=-1, keepdims=True)


@pytest.fixture
def clustered_embeddings():
    """Embeddings with known cluster structure (3 clusters)."""
    np.random.seed(42)
    # 3 clear clusters in 64-dim space
    cluster1 = np.random.randn(30, 64) + np.array([5, 0] + [0] * 62)
    cluster2 = np.random.randn(30, 64) + np.array([-5, 0] + [0] * 62)
    cluster3 = np.random.randn(30, 64) + np.array([0, 5] + [0] * 62)
    data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
    if TORCH_AVAILABLE:
        return torch.tensor(data, dtype=torch.float32)
    else:
        return data


@pytest.fixture
def random_embeddings_2d() -> np.ndarray:
    """Random 2D embeddings for visualization tests."""
    np.random.seed(42)
    return np.random.randn(100, 2)


# ============================================================================
# GO ANNOTATION FIXTURES
# ============================================================================


@pytest.fixture
def sample_go_annotations() -> Dict[str, List[str]]:
    """Sample GO annotations for testing."""
    return {
        "P12345": [
            "GO:0003677",
            "GO:0005634",
            "GO:0006355",
        ],  # DNA binding, nucleus, transcription
        "Q67890": [
            "GO:0005833",
            "GO:0015671",
            "GO:0005344",
        ],  # Hemoglobin, oxygen transport
        "P11111": ["GO:0003824", "GO:0016787"],  # Catalytic activity
    }


@pytest.fixture
def go_obo_subset() -> Dict[str, Dict[str, str]]:
    """Subset of GO ontology for testing."""
    return {
        "GO:0003677": {
            "name": "DNA binding",
            "namespace": "molecular_function",
        },
        "GO:0005634": {"name": "nucleus", "namespace": "cellular_component"},
        "GO:0006355": {
            "name": "regulation of transcription",
            "namespace": "biological_process",
        },
        "GO:0003824": {
            "name": "catalytic activity",
            "namespace": "molecular_function",
        },
    }


# ============================================================================
# CONFIG FIXTURES
# ============================================================================


@pytest.fixture
def default_config_dict() -> Dict[str, Any]:
    """Default configuration as dictionary."""
    return {
        "experiment_name": "test_experiment",
        "output_dir": "test_outputs",
        "log_level": "DEBUG",
        "random_seed": 42,
        "verbose": True,
        "dataset": {
            "source": "huggingface",
            "hf_dataset_name": "test/dataset",
            "max_samples": 100,
            "min_sequence_length": 10,
            "max_sequence_length": 512,
        },
        "model": {
            "model_type": "esm2",
            "model_name": "facebook/esm2_t6_8M_UR50D",
            "layer_ids": [3],
            "batch_size": 8,
        },
        "clustering": {
            "algorithm": "kmeans",
            "n_clusters": 5,
        },
        "visualization": {
            "reduction_method": "pca",
            "dpi": 150,
        },
    }


@pytest.fixture
def minimal_config_dict() -> Dict[str, Any]:
    """Minimal configuration for testing defaults."""
    return {"experiment_name": "minimal_test"}


# ============================================================================
# PATH FIXTURES
# ============================================================================


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Temporary output directory with subdirectories."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "descriptions").mkdir()
    (output_dir / "visualizations").mkdir()
    (output_dir / "results").mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "activations").mkdir()
    return output_dir


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


# ============================================================================
# MOCK MODEL FIXTURES
# ============================================================================


@pytest.fixture
def mock_esm_model():
    """Mock ESM-2 model for testing."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)

    def mock_forward(input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden_size = 320
        # Return mock outputs
        return MagicMock(
            last_hidden_state=torch.randn(batch_size, seq_len, hidden_size)
        )

    model.forward = mock_forward
    model.__call__ = mock_forward
    return model


# ============================================================================
# CLUSTER RESULT FIXTURES
# ============================================================================


@pytest.fixture
def cluster_labels() -> np.ndarray:
    """Sample cluster labels."""
    np.random.seed(42)
    return np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])


@pytest.fixture
def cluster_statistics() -> Dict[str, Any]:
    """Sample cluster statistics."""
    return {
        "n_clusters": 3,
        "cluster_sizes": {0: 30, 1: 30, 2: 30},
        "inertia": 1234.56,
        "n_iter": 15,
    }


# ============================================================================
# ENVIRONMENT FIXTURES
# ============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("PRISM_BIO_EXPERIMENT_NAME", "env_test")
    monkeypatch.setenv("PRISM_BIO_MODEL__MODEL_NAME", "facebook/esm2_t6_8M_UR50D")
    monkeypatch.setenv("PRISM_BIO_CLUSTERING__N_CLUSTERS", "10")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "test_token")


@pytest.fixture
def clean_env(monkeypatch):
    """Remove PRISM_BIO environment variables."""
    for key in list(os.environ.keys()):
        if key.startswith("PRISM_BIO_"):
            monkeypatch.delenv(key, raising=False)


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def sample_descriptions() -> List[Dict[str, Any]]:
    """Sample feature descriptions for output testing."""
    return [
        {
            "layer": 18,
            "unit": 100,
            "description": "Activates on hydrophobic residues in alpha helices",
            "mean_activation": 0.856,
            "highlights": ["LEU", "ILE", "VAL"],
        },
        {
            "layer": 18,
            "unit": 100,
            "description": "Responds to charged amino acids at protein termini",
            "mean_activation": 0.723,
            "highlights": ["LYS", "ARG", "ASP"],
        },
    ]


@pytest.fixture
def sample_evaluation_results() -> List[Dict[str, Any]]:
    """Sample evaluation results for output testing."""
    return [
        {"layer": 18, "unit": 100, "score": 0.85, "method": "cosine"},
        {"layer": 18, "unit": 101, "score": 0.72, "method": "cosine"},
    ]

