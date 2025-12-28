"""Protein dataset loading with comprehensive debug logging."""
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TorchDataset = object  # Fallback for type hints

from src.utils.logging_utils import get_logger, LogContext
from src.config.registry import DATASET_REGISTRY
from src.data.sequence_utils import is_valid_sequence, clean_sequence

logger = get_logger("data.protein_dataset")


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================


def filter_sequences(
    sequences: List[Dict[str, Any]],
    min_length: int = 50,
    max_length: int = 10000,
    require_valid: bool = True,
) -> List[Dict[str, Any]]:
    """Filter sequences based on length and validity.

    Args:
        sequences: List of sequence dictionaries with 'sequence' key
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        require_valid: Whether to require valid amino acid sequences

    Returns:
        Filtered list of sequences
    """
    logger.debug(
        f"Filtering {len(sequences)} sequences (min={min_length}, max={max_length})"
    )

    filtered = []
    stats = {"empty": 0, "too_short": 0, "too_long": 0, "invalid": 0, "valid": 0}

    for seq_dict in sequences:
        seq = seq_dict.get("sequence", "")

        if not seq:
            stats["empty"] += 1
            continue

        if len(seq) < min_length:
            stats["too_short"] += 1
            continue

        if len(seq) > max_length:
            stats["too_long"] += 1
            continue

        if require_valid and not is_valid_sequence(seq, allow_ambiguous=True):
            stats["invalid"] += 1
            continue

        stats["valid"] += 1
        filtered.append(seq_dict)

    logger.debug(f"Filtering stats: {stats}")
    logger.info(f"Kept {len(filtered)}/{len(sequences)} sequences after filtering")

    return filtered


# ============================================================================
# GET UNIREF50 CONVENIENCE FUNCTION
# ============================================================================


def get_uniref50(
    streaming: bool = True,
    max_samples: Optional[int] = None,
    split: str = "train",
    **kwargs,
) -> Iterator[Dict[str, Any]]:
    """Convenience function to load UniRef50 dataset.

    Args:
        streaming: Whether to stream (memory efficient)
        max_samples: Maximum samples to return
        split: Dataset split to use
        **kwargs: Additional arguments passed to loader

    Returns:
        Iterator of sequence dictionaries
    """
    loader = HuggingFaceProteinLoader(
        hf_dataset_name="PolyAI/uniref50",
        hf_split=split,
        streaming=streaming,
        max_samples=max_samples,
        **kwargs,
    )
    return loader.load()


# ============================================================================
# HUGGINGFACE LOADER
# ============================================================================


@DATASET_REGISTRY.register("huggingface")
class HuggingFaceProteinLoader:
    """Load protein sequences from HuggingFace datasets."""

    def __init__(
        self,
        hf_dataset_name: str,
        hf_subset: Optional[str] = None,
        hf_split: str = "train",
        streaming: bool = True,
        max_samples: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        sequence_column: str = "sequence",
        id_column: str = "id",
        **kwargs,
    ):
        """Initialize the loader.

        Args:
            hf_dataset_name: HuggingFace dataset name (e.g., "PolyAI/uniref50")
            hf_subset: Dataset subset/configuration name
            hf_split: Dataset split to load
            streaming: Whether to stream the dataset
            max_samples: Maximum samples to load
            cache_dir: Cache directory for downloaded data
            sequence_column: Column name for sequences
            id_column: Column name for sequence IDs
            **kwargs: Additional arguments (ignored)
        """
        logger.debug(f"Initializing HuggingFaceProteinLoader")
        logger.debug(f"  dataset_name: {hf_dataset_name}")
        logger.debug(f"  subset: {hf_subset}")
        logger.debug(f"  split: {hf_split}")
        logger.debug(f"  streaming: {streaming}")
        logger.debug(f"  max_samples: {max_samples}")
        logger.debug(f"  cache_dir: {cache_dir}")
        logger.debug(f"  sequence_column: {sequence_column}")
        logger.debug(f"  id_column: {id_column}")

        self.dataset_name = hf_dataset_name
        self.subset = hf_subset
        self.split = hf_split
        self.streaming = streaming
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        self.sequence_column = sequence_column
        self.id_column = id_column

    def load(self) -> Iterator[Dict[str, Any]]:
        """Load the dataset.

        Returns:
            Iterator of sequence dictionaries with 'id' and 'sequence' keys
        """
        from datasets import load_dataset

        with LogContext(
            logger,
            "Loading HuggingFace dataset",
            name=self.dataset_name,
            streaming=self.streaming,
        ):
            logger.debug(f"Calling datasets.load_dataset()")
            logger.debug(f"  path={self.dataset_name}")
            logger.debug(f"  name={self.subset}")
            logger.debug(f"  split={self.split}")

            dataset = load_dataset(
                path=self.dataset_name,
                name=self.subset,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=True,
            )

            logger.info(f"Dataset loaded successfully")
            if hasattr(dataset, "num_rows"):
                logger.info(f"  Total rows: {dataset.num_rows}")
            logger.debug(f"  Dataset type: {type(dataset)}")
            if hasattr(dataset, "features"):
                logger.debug(f"  Features: {dataset.features}")

        # Apply max_samples limit
        if self.max_samples:
            logger.debug(f"Applying max_samples limit: {self.max_samples}")
            dataset = dataset.take(self.max_samples)

        # Normalize to common format
        def normalize_item(item):
            """Normalize item to have 'id' and 'sequence' keys."""
            return {
                "id": item.get(self.id_column, item.get("name", item.get("entry", ""))),
                "sequence": item.get(self.sequence_column, ""),
                **{k: v for k, v in item.items() if k not in [self.id_column, self.sequence_column]},
            }

        for item in dataset:
            yield normalize_item(item)


# ============================================================================
# FASTA LOADER
# ============================================================================


@DATASET_REGISTRY.register("fasta")
class FastaProteinLoader:
    """Load protein sequences from FASTA files."""

    def __init__(
        self,
        file_path: str,
        max_samples: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the loader.

        Args:
            file_path: Path to FASTA file
            max_samples: Maximum samples to load
            **kwargs: Additional arguments (ignored)
        """
        logger.debug(f"Initializing FastaProteinLoader")
        logger.debug(f"  file_path: {file_path}")
        logger.debug(f"  max_samples: {max_samples}")

        self.file_path = file_path
        self.max_samples = max_samples

    def load(self) -> Iterator[Dict[str, Any]]:
        """Load sequences from FASTA file.

        Returns:
            Iterator of sequence dictionaries
        """
        from Bio import SeqIO

        with LogContext(logger, "Loading FASTA file", path=self.file_path):
            count = 0
            for record in SeqIO.parse(self.file_path, "fasta"):
                if self.max_samples and count >= self.max_samples:
                    break

                yield {
                    "id": record.id,
                    "sequence": str(record.seq),
                    "description": record.description,
                }
                count += 1

            logger.info(f"Loaded {count} sequences from FASTA")


# ============================================================================
# PYTORCH DATASET
# ============================================================================


class ProteinDataset(TorchDataset):
    """PyTorch Dataset for protein sequences."""

    def __init__(
        self,
        sequences: Union[List[Dict[str, Any]], Iterator[Dict[str, Any]]],
        tokenizer: Any,
        max_length: int = 1024,
        min_length: int = 50,
        filter_invalid: bool = True,
        **kwargs,
    ):
        """Initialize the dataset.

        Args:
            sequences: List or iterator of sequence dictionaries
            tokenizer: Tokenizer for encoding sequences
            max_length: Maximum sequence length (will truncate longer)
            min_length: Minimum sequence length (will filter shorter)
            filter_invalid: Whether to filter invalid sequences
            **kwargs: Additional arguments (ignored)
        """
        logger.debug(f"Initializing ProteinDataset")
        logger.debug(f"  max_length: {max_length}")
        logger.debug(f"  min_length: {min_length}")
        logger.debug(f"  filter_invalid: {filter_invalid}")
        logger.debug(f"  Tokenizer: {type(tokenizer).__name__}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length

        # Process sequences
        with LogContext(logger, "Processing sequences"):
            if isinstance(sequences, list):
                self.sequences = sequences
            else:
                logger.debug("Converting iterator to list...")
                self.sequences = list(sequences)

            logger.info(f"Loaded {len(self.sequences)} sequences")

            # Filter if requested
            if filter_invalid:
                self.sequences = filter_sequences(
                    self.sequences,
                    min_length=min_length,
                    max_length=max_length * 3,  # Leave room for tokenizer
                    require_valid=True,
                )

        # Log sequence length distribution
        if self.sequences:
            lengths = [len(s.get("sequence", "")) for s in self.sequences]
            logger.debug(f"Sequence length stats:")
            logger.debug(f"  Min: {min(lengths)}")
            logger.debug(f"  Max: {max(lengths)}")
            logger.debug(f"  Mean: {sum(lengths)/len(lengths):.1f}")

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a tokenized sequence.

        Args:
            idx: Index of sequence

        Returns:
            Dictionary with input_ids, attention_mask, and metadata

        Raises:
            IndexError: If index is out of bounds
        """
        if idx < 0 or idx >= len(self.sequences):
            logger.error(
                f"Index {idx} out of bounds (dataset size: {len(self.sequences)})"
            )
            raise IndexError(f"Index {idx} out of bounds")

        seq_dict = self.sequences[idx]
        sequence = seq_dict.get("sequence", "")
        seq_id = seq_dict.get(
            "id", seq_dict.get("name", seq_dict.get("entry", f"seq_{idx}"))
        )

        logger.debug(f"Tokenizing sequence {idx}: {seq_id} (length: {len(sequence)})")

        # Tokenize
        tokens = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "sequence_id": seq_id,
            "sequence": sequence,
        }

        logger.debug(f"  input_ids shape: {result['input_ids'].shape}")
        logger.debug(f"  attention_mask sum: {result['attention_mask'].sum().item()}")

        return result


class ProteinCollator:
    """Collate function for batching protein sequences."""

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate a batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ProteinCollator")

        logger.debug(f"Collating batch of {len(batch)} samples")

        result = {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "sequence_id": [b["sequence_id"] for b in batch],
        }

        logger.debug(f"  Batch input_ids shape: {result['input_ids'].shape}")
        logger.debug(f"  Batch attention_mask shape: {result['attention_mask'].shape}")

        return result


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "filter_sequences",
    "get_uniref50",
    "HuggingFaceProteinLoader",
    "FastaProteinLoader",
    "ProteinDataset",
    "ProteinCollator",
]

