"""Helper utilities for PRISM-Bio."""
import gc
import os
from typing import Optional

from src.utils.logging_utils import get_logger

logger = get_logger("utils.helpers")


def clear_gpu_cache(device: Optional[str] = None) -> None:
    """Clear GPU cache based on device type.

    Args:
        device: Device type ("cuda", "mps", or None for auto-detect)
    """
    try:
        import torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if device == "mps":
            torch.mps.empty_cache()
            logger.debug("Cleared MPS cache")
        elif device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

        # Also run garbage collection
        gc.collect()
        logger.debug("Ran garbage collection")

    except ImportError:
        logger.warning("PyTorch not available, cannot clear GPU cache")
    except Exception as e:
        logger.warning(f"Error clearing GPU cache: {e}")


def get_device(preferred: Optional[str] = None) -> str:
    """Get the best available device.

    Args:
        preferred: Preferred device ("cuda", "mps", "cpu", or None for auto)

    Returns:
        Device string to use with PyTorch
    """
    try:
        import torch

        if preferred is not None:
            if preferred == "cuda" and torch.cuda.is_available():
                logger.debug("Using preferred device: cuda")
                return "cuda"
            elif (
                preferred == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                logger.debug("Using preferred device: mps")
                return "mps"
            elif preferred == "cpu":
                logger.debug("Using preferred device: cpu")
                return "cpu"
            else:
                logger.warning(
                    f"Preferred device '{preferred}' not available, auto-detecting"
                )

        # Auto-detect
        if torch.cuda.is_available():
            logger.debug("Auto-detected device: cuda")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.debug("Auto-detected device: mps")
            return "mps"
        else:
            logger.debug("Auto-detected device: cpu")
            return "cpu"

    except ImportError:
        logger.warning("PyTorch not available, defaulting to cpu")
        return "cpu"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    random.seed(seed)
    logger.debug(f"Set Python random seed to {seed}")

    try:
        import numpy as np

        np.random.seed(seed)
        logger.debug(f"Set NumPy random seed to {seed}")
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"Set PyTorch random seed to {seed}")
    except ImportError:
        pass


def get_huggingface_token() -> Optional[str]:
    """Get HuggingFace token from environment.

    Returns:
        Token string or None if not set
    """
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        logger.debug("Found HuggingFace token in environment")
    else:
        logger.warning("No HuggingFace token found in environment")
    return token


def ensure_huggingface_token() -> str:
    """Ensure HuggingFace token is available.

    Returns:
        Token string

    Raises:
        ValueError: If token is not set
    """
    token = get_huggingface_token()
    if not token:
        raise ValueError(
            "HUGGING_FACE_HUB_TOKEN environment variable is required. "
            "Set it with: export HUGGING_FACE_HUB_TOKEN='your_token_here'"
        )
    return token


def format_size(num_bytes: int) -> str:
    """Format byte size to human-readable string.

    Args:
        num_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"






