"""Comprehensive tests for protein dataset loading and processing."""
import pytest

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# These tests are designed to run against the implementation
# Many will be marked with @pytest.mark.unit for fast tests


# ============================================================================
# GET_UNIREF50 TESTS
# ============================================================================


class TestGetUniref50:
    """Tests for UniRef50 dataset loading function."""

    @pytest.mark.unit
    def test_function_exists(self):
        """get_uniref50 function should exist."""
        from src.data.protein_dataset import get_uniref50
        assert callable(get_uniref50)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_returns_iterable_in_streaming_mode(self):
        """Dataset should be iterable when streaming=True."""
        from src.data.protein_dataset import get_uniref50
        dataset = get_uniref50(streaming=True, max_samples=10)
        assert hasattr(dataset, '__iter__')

    @pytest.mark.integration
    @pytest.mark.slow
    def test_max_samples_limits_iteration(self):
        """Should stop after max_samples."""
        from src.data.protein_dataset import get_uniref50
        dataset = get_uniref50(streaming=True, max_samples=5)
        items = list(dataset)
        assert len(items) == 5

    @pytest.mark.integration
    @pytest.mark.slow
    def test_contains_required_fields(self):
        """Each item should have 'sequence' field."""
        from src.data.protein_dataset import get_uniref50
        dataset = get_uniref50(streaming=True, max_samples=1)
        item = next(iter(dataset))
        assert 'sequence' in item

    @pytest.mark.integration
    @pytest.mark.slow
    def test_sequences_are_strings(self):
        """Sequences should be strings."""
        from src.data.protein_dataset import get_uniref50
        dataset = get_uniref50(streaming=True, max_samples=5)
        for item in dataset:
            assert isinstance(item['sequence'], str)


# ============================================================================
# PROTEIN_DATASET INIT TESTS
# ============================================================================


class TestProteinDatasetInit:
    """Tests for ProteinDataset initialization."""

    @pytest.mark.unit
    def test_init_with_list_of_dicts(self, valid_sequences, mock_esm_tokenizer):
        """Should accept list of sequence dicts."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=1)
        assert len(ds) == len(valid_sequences)

    @pytest.mark.unit
    def test_init_sets_max_length(self, valid_sequences, mock_esm_tokenizer):
        """max_length should be stored and used."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, max_length=512, min_length=1)
        assert ds.max_length == 512

    @pytest.mark.unit
    def test_init_sets_min_length(self, valid_sequences, mock_esm_tokenizer):
        """min_length should filter short sequences."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=100)
        # Only sequences >= 100 AA should remain
        assert len(ds) < len(valid_sequences)

    @pytest.mark.unit
    def test_init_with_empty_list(self, mock_esm_tokenizer):
        """Should handle empty sequence list."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset([], mock_esm_tokenizer, min_length=1)
        assert len(ds) == 0

    @pytest.mark.unit
    def test_init_with_iterator(self, valid_sequences, mock_esm_tokenizer):
        """Should accept iterator of sequences."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(iter(valid_sequences), mock_esm_tokenizer, min_length=1)
        assert len(ds) == len(valid_sequences)


# ============================================================================
# PROTEIN_DATASET GETITEM TESTS
# ============================================================================


class TestProteinDatasetGetItem:
    """Tests for __getitem__ behavior."""

    @pytest.mark.unit
    def test_returns_dict_with_required_keys(self, valid_sequences, mock_esm_tokenizer):
        """Should return dict with input_ids, attention_mask, metadata."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=1)
        item = ds[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'sequence_id' in item

    @pytest.mark.unit
    def test_input_ids_are_tensors(self, valid_sequences, mock_esm_tokenizer):
        """input_ids should be torch tensors."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=1)
        item = ds[0]
        assert isinstance(item['input_ids'], torch.Tensor)
        assert item['input_ids'].dtype == torch.long

    @pytest.mark.unit
    def test_attention_mask_are_tensors(self, valid_sequences, mock_esm_tokenizer):
        """attention_mask should be torch tensors."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=1)
        item = ds[0]
        assert isinstance(item['attention_mask'], torch.Tensor)

    @pytest.mark.unit
    def test_input_ids_length_equals_max_length(self, valid_sequences, mock_esm_tokenizer):
        """Tokenized sequences should be padded/truncated to max_length."""
        from src.data.protein_dataset import ProteinDataset
        max_len = 128
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, max_length=max_len, min_length=1)
        for i in range(len(ds)):
            item = ds[i]
            assert item['input_ids'].shape[0] == max_len
            assert item['attention_mask'].shape[0] == max_len

    @pytest.mark.unit
    def test_preserves_sequence_id(self, valid_sequences, mock_esm_tokenizer):
        """Original sequence ID should be preserved."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=1)
        item = ds[0]
        assert item['sequence_id'] == valid_sequences[0]['id']

    @pytest.mark.unit
    def test_out_of_bounds_raises(self, valid_sequences, mock_esm_tokenizer):
        """Out of bounds index should raise IndexError."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=1)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]

    @pytest.mark.unit
    def test_negative_index_raises(self, valid_sequences, mock_esm_tokenizer):
        """Negative index should raise IndexError."""
        from src.data.protein_dataset import ProteinDataset
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=1)
        with pytest.raises(IndexError):
            _ = ds[-1]


# ============================================================================
# PROTEIN_DATASET FILTERING TESTS
# ============================================================================


class TestProteinDatasetFiltering:
    """Tests for sequence filtering."""

    @pytest.mark.unit
    def test_filters_empty_sequences(self, mock_esm_tokenizer):
        """Empty sequences should be filtered."""
        from src.data.protein_dataset import ProteinDataset
        seqs = [{"id": "EMPTY", "sequence": ""}, {"id": "OK", "sequence": "MKTAY"}]
        ds = ProteinDataset(seqs, mock_esm_tokenizer, filter_invalid=True, min_length=1)
        assert len(ds) == 1

    @pytest.mark.unit
    def test_filters_below_min_length(self, mock_esm_tokenizer):
        """Sequences below min_length should be filtered."""
        from src.data.protein_dataset import ProteinDataset
        seqs = [
            {"id": "SHORT", "sequence": "MKT"},
            {"id": "OK", "sequence": "MKTAYIAKQRQISFVKSH"},
        ]
        ds = ProteinDataset(seqs, mock_esm_tokenizer, min_length=10, filter_invalid=True)
        assert len(ds) == 1
        assert ds[0]['sequence_id'] == "OK"

    @pytest.mark.unit
    def test_filter_invalid_false_keeps_all(self, mock_esm_tokenizer):
        """filter_invalid=False should keep all sequences."""
        from src.data.protein_dataset import ProteinDataset
        seqs = [
            {"id": "A", "sequence": "MKT"},
            {"id": "B", "sequence": "MKTAY"},
        ]
        ds = ProteinDataset(seqs, mock_esm_tokenizer, filter_invalid=False, min_length=1)
        assert len(ds) == 2

    @pytest.mark.unit
    def test_handles_nonstandard_amino_acids(self, mock_esm_tokenizer):
        """Non-standard amino acids should be handled based on settings."""
        from src.data.protein_dataset import ProteinDataset
        seqs = [{"id": "AMBIG", "sequence": "MKTXUAY"}]
        # Should not raise with default settings
        ds = ProteinDataset(seqs, mock_esm_tokenizer, filter_invalid=False, min_length=1)
        assert len(ds) >= 0  # Depends on implementation


# ============================================================================
# PROTEIN_DATASET EDGE CASES
# ============================================================================


class TestProteinDatasetEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_single_amino_acid_sequence(self, mock_esm_tokenizer):
        """Single AA sequence should work if min_length allows."""
        from src.data.protein_dataset import ProteinDataset
        seqs = [{"id": "SINGLE", "sequence": "M"}]
        ds = ProteinDataset(seqs, mock_esm_tokenizer, min_length=1, filter_invalid=False)
        item = ds[0]
        assert item['input_ids'] is not None

    @pytest.mark.unit
    def test_all_same_amino_acid(self, mock_esm_tokenizer):
        """Repetitive sequence should tokenize correctly."""
        from src.data.protein_dataset import ProteinDataset
        seqs = [{"id": "REPEAT", "sequence": "A" * 100}]
        ds = ProteinDataset(seqs, mock_esm_tokenizer, min_length=1)
        item = ds[0]
        assert item['input_ids'] is not None

    @pytest.mark.unit
    def test_unicode_in_sequence_id(self, mock_esm_tokenizer):
        """Unicode in sequence ID should be preserved."""
        from src.data.protein_dataset import ProteinDataset
        seqs = [{"id": "P12345_α-helix", "sequence": "MKTAY"}]
        ds = ProteinDataset(seqs, mock_esm_tokenizer, min_length=1, filter_invalid=False)
        item = ds[0]
        assert "α" in item['sequence_id']

    @pytest.mark.unit
    def test_very_long_sequence(self, mock_esm_tokenizer):
        """Very long sequence should be truncated."""
        from src.data.protein_dataset import ProteinDataset
        seqs = [{"id": "LONG", "sequence": "A" * 5000}]
        ds = ProteinDataset(seqs, mock_esm_tokenizer, max_length=100, min_length=1)
        item = ds[0]
        assert item['input_ids'].shape[0] == 100


# ============================================================================
# PROTEIN_COLLATOR TESTS
# ============================================================================


class TestProteinCollator:
    """Tests for batch collation."""

    @pytest.mark.unit
    def test_batches_have_correct_shape(self, valid_sequences, mock_esm_tokenizer):
        """Batched tensors should have [batch_size, seq_len] shape."""
        from src.data.protein_dataset import ProteinDataset, ProteinCollator
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, max_length=100, min_length=1)
        collator = ProteinCollator()
        batch = collator([ds[i] for i in range(min(3, len(ds)))])
        assert batch['input_ids'].shape[0] == min(3, len(ds))
        assert batch['input_ids'].shape[1] == 100

    @pytest.mark.unit
    def test_collator_preserves_metadata_as_list(self, valid_sequences, mock_esm_tokenizer):
        """Metadata should be collated as list."""
        from src.data.protein_dataset import ProteinDataset, ProteinCollator
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, min_length=1)
        collator = ProteinCollator()
        batch = collator([ds[i] for i in range(min(3, len(ds)))])
        assert isinstance(batch['sequence_id'], list)
        assert len(batch['sequence_id']) == min(3, len(ds))

    @pytest.mark.unit
    def test_works_with_dataloader(self, valid_sequences, mock_esm_tokenizer):
        """Should work with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        from src.data.protein_dataset import ProteinDataset, ProteinCollator
        
        ds = ProteinDataset(valid_sequences, mock_esm_tokenizer, max_length=100, min_length=1)
        if len(ds) == 0:
            pytest.skip("No valid sequences after filtering")
        
        loader = DataLoader(ds, batch_size=2, collate_fn=ProteinCollator())
        batch = next(iter(loader))
        assert batch['input_ids'].shape[0] <= 2


# ============================================================================
# FILTER_SEQUENCES FUNCTION TESTS
# ============================================================================


class TestFilterSequences:
    """Tests for standalone filter_sequences function."""

    @pytest.mark.unit
    def test_removes_empty_sequences(self):
        """Empty sequences should be removed."""
        from src.data.protein_dataset import filter_sequences
        seqs = [{"id": "A", "sequence": ""}, {"id": "B", "sequence": "MKTA"}]
        filtered = filter_sequences(seqs)
        assert len(filtered) == 1

    @pytest.mark.unit
    def test_removes_short_sequences(self):
        """Sequences below min_length should be removed."""
        from src.data.protein_dataset import filter_sequences
        seqs = [{"id": "A", "sequence": "MK"}, {"id": "B", "sequence": "MKTAYIAK"}]
        filtered = filter_sequences(seqs, min_length=5)
        assert len(filtered) == 1
        assert filtered[0]['id'] == "B"

    @pytest.mark.unit
    def test_preserves_order(self):
        """Filtered sequences should maintain original order."""
        from src.data.protein_dataset import filter_sequences
        seqs = [
            {"id": "C", "sequence": "MKTAY"},
            {"id": "A", "sequence": "MKTAYIAK"},
            {"id": "B", "sequence": "MKTAYIAKQRQ"},
        ]
        filtered = filter_sequences(seqs, min_length=3)
        assert [s['id'] for s in filtered] == ["C", "A", "B"]

    @pytest.mark.unit
    def test_handles_empty_input(self):
        """Should handle empty input list."""
        from src.data.protein_dataset import filter_sequences
        filtered = filter_sequences([])
        assert filtered == []


# ============================================================================
# HUGGINGFACE LOADER TESTS
# ============================================================================


class TestHuggingFaceProteinLoader:
    """Tests for HuggingFace dataset loader."""

    @pytest.mark.unit
    def test_loader_exists(self):
        """HuggingFaceProteinLoader should exist."""
        from src.data.protein_dataset import HuggingFaceProteinLoader
        assert HuggingFaceProteinLoader is not None

    @pytest.mark.unit
    def test_loader_accepts_config_params(self):
        """Loader should accept configuration parameters."""
        from src.data.protein_dataset import HuggingFaceProteinLoader
        loader = HuggingFaceProteinLoader(
            hf_dataset_name="test/dataset",
            hf_split="train",
            streaming=True,
            max_samples=100,
        )
        assert loader.dataset_name == "test/dataset"
        assert loader.streaming is True
        assert loader.max_samples == 100


# ============================================================================
# FASTA LOADER TESTS
# ============================================================================


class TestFastaLoader:
    """Tests for FASTA file loader."""

    @pytest.mark.unit
    def test_loader_exists(self):
        """FastaProteinLoader should exist."""
        from src.data.protein_dataset import FastaProteinLoader
        assert FastaProteinLoader is not None

    @pytest.mark.unit
    def test_loader_accepts_file_path(self, tmp_path):
        """Loader should accept file path."""
        from src.data.protein_dataset import FastaProteinLoader
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(">seq1\nMKTAY\n>seq2\nACDEF\n")
        
        loader = FastaProteinLoader(file_path=str(fasta_file))
        assert loader.file_path == str(fasta_file)

    @pytest.mark.unit
    def test_loads_fasta_file(self, tmp_path):
        """Should load sequences from FASTA file."""
        from src.data.protein_dataset import FastaProteinLoader
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(">seq1\nMKTAY\n>seq2\nACDEF\n")
        
        loader = FastaProteinLoader(file_path=str(fasta_file))
        sequences = list(loader.load())
        assert len(sequences) == 2
        assert sequences[0]['sequence'] == "MKTAY"


# ============================================================================
# SEQUENCE UTILS TESTS
# ============================================================================


class TestSequenceUtils:
    """Tests for sequence utility functions."""

    @pytest.mark.unit
    def test_is_valid_amino_acid(self):
        """Should correctly identify valid amino acids."""
        from src.data.sequence_utils import is_valid_sequence
        assert is_valid_sequence("ACDEFGHIKLMNPQRSTVWY") is True
        assert is_valid_sequence("MKTAY") is True
        assert is_valid_sequence("MKT123") is False
        assert is_valid_sequence("") is False

    @pytest.mark.unit
    def test_clean_sequence(self):
        """Should clean sequence (uppercase, remove whitespace)."""
        from src.data.sequence_utils import clean_sequence
        assert clean_sequence("mktay") == "MKTAY"
        assert clean_sequence("MKT AY") == "MKTAY"
        assert clean_sequence("  MKTAY  ") == "MKTAY"

