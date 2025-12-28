"""Tests for logging utilities."""
import logging
import os
from pathlib import Path

import pytest

from src.utils.logging_utils import (
    ColoredFormatter,
    LogContext,
    get_logger,
    log_system_info,
    setup_logging,
)


class TestColoredFormatter:
    """Tests for ColoredFormatter class."""

    def test_formatter_adds_colors(self):
        """Formatter should add ANSI color codes to levelname."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        # Should contain ANSI codes
        assert "\033[" in formatted

    def test_formatter_handles_all_levels(self):
        """Formatter should handle all log levels."""
        formatter = ColoredFormatter("%(levelname)s")
        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
        for level in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )
            # Should not raise
            formatter.format(record)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_log_file(self, tmp_path, default_config_dict):
        """setup_logging should create a log file."""
        logger, log_file = setup_logging(
            default_config_dict, log_dir=str(tmp_path), mode="feature_description"
        )
        assert Path(log_file).exists()
        assert Path(log_file).suffix == ".log"

    def test_logs_config_parameters(self, tmp_path, default_config_dict):
        """Log file should contain configuration parameters."""
        logger, log_file = setup_logging(
            default_config_dict, log_dir=str(tmp_path), mode="feature_description"
        )
        logger.info("Test message")

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        with open(log_file, "r") as f:
            content = f.read()

        assert "CONFIGURATION PARAMETERS" in content
        assert "experiment_name" in content

    def test_different_modes_create_different_filenames(self, tmp_path, default_config_dict):
        """Different modes should create different log filenames."""
        _, log1 = setup_logging(
            default_config_dict, log_dir=str(tmp_path), mode="feature_description"
        )
        _, log2 = setup_logging(
            default_config_dict, log_dir=str(tmp_path), mode="evaluation"
        )
        _, log3 = setup_logging(
            default_config_dict, log_dir=str(tmp_path), mode="meta_evaluation"
        )

        assert "layer" in log1
        assert "evaluation" in log2
        assert "meta-evaluation" in log3

    def test_verbose_mode_sets_debug_level(self, tmp_path, default_config_dict):
        """verbose=True should set DEBUG level."""
        logger, _ = setup_logging(
            default_config_dict, log_dir=str(tmp_path), verbose=True
        )
        assert logger.level == logging.DEBUG

    def test_non_verbose_mode_sets_info_level(self, tmp_path, default_config_dict):
        """verbose=False should set INFO level."""
        logger, _ = setup_logging(
            default_config_dict, log_dir=str(tmp_path), verbose=False
        )
        assert logger.level == logging.INFO

    def test_creates_log_directory_if_not_exists(self, tmp_path, default_config_dict):
        """Should create log directory if it doesn't exist."""
        new_log_dir = tmp_path / "new_logs" / "subdir"
        assert not new_log_dir.exists()

        setup_logging(default_config_dict, log_dir=str(new_log_dir))

        assert new_log_dir.exists()

    def test_handles_dict_config(self, tmp_path):
        """Should handle plain dictionary config."""
        config = {"experiment_name": "test", "model": {"model_name": "test_model"}}
        logger, log_file = setup_logging(config, log_dir=str(tmp_path))
        assert Path(log_file).exists()

    def test_handles_empty_config(self, tmp_path):
        """Should handle empty config gracefully."""
        logger, log_file = setup_logging({}, log_dir=str(tmp_path))
        assert Path(log_file).exists()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_with_prefix(self):
        """get_logger should return logger with prism_bio prefix."""
        logger = get_logger("test.module")
        assert logger.name == "prism_bio.test.module"

    def test_returns_child_of_prism_bio_logger(self):
        """Returned logger should be child of prism_bio logger."""
        logger = get_logger("child")
        parent = logging.getLogger("prism_bio")
        # Logger should propagate to parent
        assert logger.parent == parent or logger.name.startswith("prism_bio.")


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_logs_start_message(self, caplog):
        """LogContext should log start message."""
        logger = logging.getLogger("test_context")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO):
            with LogContext(logger, "Test operation"):
                pass

        assert "[START] Test operation" in caplog.text

    def test_logs_done_message(self, caplog):
        """LogContext should log done message on success."""
        logger = logging.getLogger("test_context")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO):
            with LogContext(logger, "Test operation"):
                pass

        assert "[DONE] Test operation" in caplog.text
        assert "took" in caplog.text

    def test_logs_failed_message_on_exception(self, caplog):
        """LogContext should log failed message on exception."""
        logger = logging.getLogger("test_context")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                with LogContext(logger, "Test operation"):
                    raise ValueError("Test error")

        assert "[FAILED] Test operation" in caplog.text

    def test_includes_kwargs_in_start_message(self, caplog):
        """LogContext should include kwargs in start message."""
        logger = logging.getLogger("test_context")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO):
            with LogContext(logger, "Loading", name="dataset", size=100):
                pass

        assert "name=dataset" in caplog.text
        assert "size=100" in caplog.text

    def test_does_not_suppress_exceptions(self):
        """LogContext should not suppress exceptions."""
        logger = logging.getLogger("test_context")

        with pytest.raises(RuntimeError):
            with LogContext(logger, "Test"):
                raise RuntimeError("Should propagate")


class TestLogSystemInfo:
    """Tests for log_system_info function."""

    def test_logs_python_version(self, caplog):
        """Should log Python version."""
        logger = logging.getLogger("test_system")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO):
            log_system_info(logger)

        assert "Python version" in caplog.text

    def test_logs_platform(self, caplog):
        """Should log platform info."""
        logger = logging.getLogger("test_system")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO):
            log_system_info(logger)

        assert "Platform" in caplog.text

