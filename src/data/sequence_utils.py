"""Sequence utilities for protein data processing."""
import re
from typing import Optional

from src.utils.logging_utils import get_logger

logger = get_logger("data.sequence_utils")

# Standard amino acid alphabet
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Extended amino acid alphabet (includes ambiguous codes)
EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYXUBZO*")


def is_valid_sequence(
    sequence: str,
    allow_ambiguous: bool = True,
    min_length: int = 1,
) -> bool:
    """Check if a sequence contains only valid amino acids.

    Args:
        sequence: Protein sequence string
        allow_ambiguous: Whether to allow ambiguous amino acid codes (X, B, Z, etc.)
        min_length: Minimum required sequence length

    Returns:
        True if sequence is valid, False otherwise
    """
    if not sequence:
        return False

    if len(sequence) < min_length:
        return False

    valid_chars = EXTENDED_AA if allow_ambiguous else STANDARD_AA
    upper_seq = sequence.upper()

    for char in upper_seq:
        if char not in valid_chars:
            logger.debug(f"Invalid character '{char}' in sequence")
            return False

    return True


def clean_sequence(sequence: str, remove_gaps: bool = True) -> str:
    """Clean a protein sequence.

    - Convert to uppercase
    - Remove whitespace
    - Optionally remove gap characters

    Args:
        sequence: Raw sequence string
        remove_gaps: Whether to remove gap characters (-.)

    Returns:
        Cleaned sequence string
    """
    # Uppercase
    cleaned = sequence.upper()

    # Remove whitespace
    cleaned = "".join(cleaned.split())

    # Remove gaps if requested
    if remove_gaps:
        cleaned = re.sub(r"[-.]", "", cleaned)

    return cleaned


def get_sequence_stats(sequence: str) -> dict:
    """Get basic statistics for a sequence.

    Args:
        sequence: Protein sequence string

    Returns:
        Dictionary with sequence statistics
    """
    upper_seq = sequence.upper()
    length = len(sequence)

    if length == 0:
        return {
            "length": 0,
            "composition": {},
            "invalid_chars": set(),
            "is_valid": False,
        }

    # Amino acid composition
    composition = {}
    invalid_chars = set()
    for char in upper_seq:
        if char in EXTENDED_AA:
            composition[char] = composition.get(char, 0) + 1
        else:
            invalid_chars.add(char)

    # Convert counts to fractions
    for aa in composition:
        composition[aa] = composition[aa] / length

    return {
        "length": length,
        "composition": composition,
        "invalid_chars": invalid_chars,
        "is_valid": len(invalid_chars) == 0,
    }


def parse_fasta_header(header: str) -> dict:
    """Parse a FASTA header line.

    Handles common formats:
    - UniProt: >sp|P12345|PROTEIN_NAME Protein description OS=...
    - NCBI: >gi|123456|ref|NP_001234| description
    - Simple: >seqname description

    Args:
        header: FASTA header line (with or without '>')

    Returns:
        Dictionary with parsed fields
    """
    # Remove leading '>' if present
    if header.startswith(">"):
        header = header[1:]

    header = header.strip()

    result = {
        "raw_header": header,
        "id": "",
        "description": "",
        "accession": None,
        "entry_name": None,
        "organism": None,
    }

    if not header:
        return result

    # Split on first whitespace
    parts = header.split(None, 1)
    first_part = parts[0]
    description = parts[1] if len(parts) > 1 else ""

    # Try to parse UniProt format: sp|P12345|NAME or tr|A0A0A0|NAME
    uniprot_match = re.match(r"^(sp|tr)\|([A-Z0-9]+)\|(\S+)", first_part)
    if uniprot_match:
        result["database"] = uniprot_match.group(1)
        result["accession"] = uniprot_match.group(2)
        result["entry_name"] = uniprot_match.group(3)
        result["id"] = result["accession"]

        # Parse OS= organism
        os_match = re.search(r"OS=([^=]+?)(?:\s+\w+=|$)", description)
        if os_match:
            result["organism"] = os_match.group(1).strip()
    else:
        # Simple format
        result["id"] = first_part

    result["description"] = description
    return result


def truncate_sequence(
    sequence: str,
    max_length: int,
    truncation_mode: str = "end",
) -> str:
    """Truncate a sequence to maximum length.

    Args:
        sequence: Protein sequence
        max_length: Maximum length
        truncation_mode: How to truncate - "end", "start", or "center"

    Returns:
        Truncated sequence
    """
    if len(sequence) <= max_length:
        return sequence

    if truncation_mode == "end":
        return sequence[:max_length]
    elif truncation_mode == "start":
        return sequence[-max_length:]
    elif truncation_mode == "center":
        # Keep center portion
        start = (len(sequence) - max_length) // 2
        return sequence[start : start + max_length]
    else:
        raise ValueError(f"Unknown truncation_mode: {truncation_mode}")


def split_sequence_into_chunks(
    sequence: str,
    chunk_size: int,
    overlap: int = 0,
) -> list:
    """Split a long sequence into overlapping chunks.

    Args:
        sequence: Protein sequence
        chunk_size: Size of each chunk
        overlap: Number of overlapping amino acids between chunks

    Returns:
        List of sequence chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(sequence):
        end = min(start + chunk_size, len(sequence))
        chunks.append(sequence[start:end])
        start += step

        # Avoid tiny final chunks
        if len(sequence) - start < chunk_size // 2 and start < len(sequence):
            # Extend last chunk instead
            break

    return chunks

