"""Comprehensive logging setup - matches PRISM logging functionality."""
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.utils import constants


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Make a copy to avoid modifying the original record
        record_copy = logging.makeLogRecord(record.__dict__)
        color = self.COLORS.get(record_copy.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record_copy.levelname = f"{color}{record_copy.levelname}{reset}"
        return super().format(record_copy)


def setup_logging(
    config: Any,
    log_dir: Optional[str] = None,
    mode: str = "feature_description",
    verbose: bool = True,
) -> Tuple[logging.Logger, str]:
    """
    Set up logging to both console and file with config parameters logged at start.

    Matches PRISM logging functionality with enhancements for debugging.

    Args:
        config: Configuration object (Pydantic model or dict)
        log_dir: Directory to save log files (defaults to constants.LOGS_PATH)
        mode: Logging mode - determines filename format:
            - "feature_description": For feature description analysis
            - "evaluation": For evaluation runs
            - "meta_evaluation": For meta-evaluation runs
            - "clustering_viz": For clustering visualization
            - "go_enrichment": For GO enrichment analysis
        verbose: If True, use DEBUG level; otherwise INFO

    Returns:
        Tuple of (logger, log_filename)
    """
    # Use default log dir if not specified
    if log_dir is None:
        log_dir = constants.LOGS_PATH

    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    # Extract config values
    if hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    elif hasattr(config, "dict"):
        config_dict = config.dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {}

    # Extract model and method information for filename
    model_config = config_dict.get("model", {})
    model_name = model_config.get("model_name", "unknown")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    layer_ids = model_config.get("layer_ids", [0])
    layer_str = "-".join(map(str, layer_ids))

    unit_ids = model_config.get("unit_ids")
    if unit_ids:
        unit_str = "-".join(map(str, unit_ids[:3]))
        if len(unit_ids) > 3:
            unit_str += f"-etc{len(unit_ids)}"
    else:
        unit_str = "all"

    method_name = config_dict.get("experiment_name", "prism-bio")

    # Generate log filename based on mode
    if mode == "feature_description":
        log_filename = (
            f"{log_dir}/{model_name}_layer-{layer_str}_unit-{unit_str}_{timestamp}.log"
        )
    elif mode == "evaluation":
        log_filename = f"{log_dir}/evaluation_{model_name}_{method_name}_{timestamp}.log"
    elif mode == "meta_evaluation":
        log_filename = (
            f"{log_dir}/meta-evaluation_{model_name}_{method_name}_{timestamp}.log"
        )
    elif mode == "clustering_viz":
        log_filename = f"{log_dir}/clustering-viz_{model_name}_{timestamp}.log"
    elif mode == "go_enrichment":
        log_filename = f"{log_dir}/go-enrichment_{model_name}_{timestamp}.log"
    else:
        log_filename = f"{log_dir}/{mode}_{model_name}_{timestamp}.log"

    # Configure root logger for prism_bio
    logger = logging.getLogger("prism_bio")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler (always DEBUG level for comprehensive logs)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (colored, configurable level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log configuration at the start
    _log_config_header(logger, config_dict, log_filename, mode)

    return logger, log_filename


def _log_config_header(
    logger: logging.Logger, config_dict: Dict, log_filename: str, mode: str
) -> None:
    """Log configuration parameters at the beginning of the log file.

    Args:
        logger: The logger instance
        config_dict: Configuration dictionary to log
        log_filename: Path to the log file
        mode: The logging mode
    """
    logger.info("=" * 80)
    logger.info("PRISM-BIO ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode}")
    logger.info(f"Log file: {log_filename}")
    logger.info(
        f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
    )
    logger.info("")
    logger.info("=" * 80)
    logger.info("CONFIGURATION PARAMETERS")
    logger.info("=" * 80)

    def _log_nested(d: Dict, prefix: str = "") -> None:
        """Recursively log nested dictionary."""
        for key, value in sorted(d.items()):
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _log_nested(value, full_key)
            else:
                try:
                    if isinstance(value, (list, dict, tuple)):
                        value_str = json.dumps(value)
                    else:
                        value_str = str(value)
                except (TypeError, json.JSONDecodeError):
                    value_str = str(value)
                logger.info(f"  {full_key} = {value_str}")

    _log_nested(config_dict)

    logger.info("=" * 80)
    logger.info("ANALYSIS STARTED")
    logger.info("=" * 80)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name (will be prefixed with 'prism_bio.')

    Returns:
        Logger instance for the module
    """
    return logging.getLogger(f"prism_bio.{name}")


class LogContext:
    """Context manager for logging entry/exit of major operations.

    Usage:
        with LogContext(logger, "Loading dataset", name="uniref50"):
            # ... operation ...

    This will log:
        [START] Loading dataset (name=uniref50)
        [DONE] Loading dataset (took 5.23s)
    """

    def __init__(self, logger: logging.Logger, operation: str, **kwargs: Any):
        """Initialize the log context.

        Args:
            logger: Logger instance to use
            operation: Description of the operation
            **kwargs: Additional context parameters to log
        """
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time: Optional[datetime.datetime] = None

    def __enter__(self) -> "LogContext":
        """Enter the context and log start."""
        self.start_time = datetime.datetime.now()
        if self.kwargs:
            params = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
            self.logger.info(f"[START] {self.operation} ({params})")
        else:
            self.logger.info(f"[START] {self.operation}")
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Exit the context and log completion or failure."""
        if self.start_time is None:
            duration_seconds = 0.0
        else:
            duration = datetime.datetime.now() - self.start_time
            duration_seconds = duration.total_seconds()

        if exc_type is None:
            self.logger.info(
                f"[DONE] {self.operation} (took {duration_seconds:.2f}s)"
            )
        else:
            self.logger.error(
                f"[FAILED] {self.operation} after {duration_seconds:.2f}s: {exc_val}"
            )
        return False  # Don't suppress exceptions


def log_gpu_info(logger: logging.Logger) -> None:
    """Log GPU information if available.

    Args:
        logger: Logger instance to use
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name}")
                logger.info(
                    f"    Memory: {props.total_memory / 1024**3:.1f} GB"
                )
        else:
            logger.info("CUDA available: False")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available: True")
    except ImportError:
        logger.warning("PyTorch not available, cannot log GPU info")


def log_system_info(logger: logging.Logger) -> None:
    """Log system information.

    Args:
        logger: Logger instance to use
    """
    import platform
    import os

    logger.info("=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 80)
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Hostname: {platform.node()}")
    logger.info(f"Working directory: {os.getcwd()}")

    # Log SLURM info if available
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        logger.info(f"SLURM Job ID: {slurm_job_id}")
        logger.info(f"SLURM Node: {os.environ.get('SLURM_NODELIST', 'N/A')}")
        logger.info(f"SLURM GPUs: {os.environ.get('SLURM_GPUS', 'N/A')}")

    log_gpu_info(logger)
    logger.info("=" * 80)

