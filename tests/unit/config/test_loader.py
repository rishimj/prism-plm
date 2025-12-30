"""Tests for configuration loader."""
import os
from pathlib import Path

import pytest
import yaml

from src.config.loader import (
    _convert_value,
    _load_env_vars,
    create_default_config,
    deep_merge,
    load_config,
    load_yaml,
    save_config,
)
from src.config.schema import Config


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_loads_valid_yaml(self, tmp_path):
        """Should load valid YAML file."""
        yaml_content = {"experiment_name": "test", "random_seed": 123}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        result = load_yaml(yaml_file)
        assert result["experiment_name"] == "test"
        assert result["random_seed"] == 123

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        """Should return empty dict for missing file."""
        result = load_yaml(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        """Should return empty dict for empty file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.touch()

        result = load_yaml(yaml_file)
        assert result == {}

    def test_handles_nested_yaml(self, tmp_path):
        """Should handle nested YAML structure."""
        yaml_content = {
            "dataset": {"source": "huggingface", "max_samples": 100},
            "model": {"model_name": "test"},
        }
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        result = load_yaml(yaml_file)
        assert result["dataset"]["source"] == "huggingface"
        assert result["model"]["model_name"] == "test"


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_flat_merge(self):
        """Should merge flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Should deep merge nested dictionaries."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 10, "z": 20}}
        result = deep_merge(base, override)
        assert result["a"]["x"] == 1  # Preserved
        assert result["a"]["y"] == 10  # Overridden
        assert result["a"]["z"] == 20  # Added
        assert result["b"] == 3  # Preserved

    def test_override_with_non_dict(self):
        """Should override dict with non-dict."""
        base = {"a": {"x": 1}}
        override = {"a": "string"}
        result = deep_merge(base, override)
        assert result["a"] == "string"

    def test_original_not_modified(self):
        """Should not modify original dictionaries."""
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert "b" not in base


class TestConvertValue:
    """Tests for _convert_value function."""

    def test_converts_true(self):
        """Should convert 'true' to True."""
        assert _convert_value("true") is True
        assert _convert_value("True") is True
        assert _convert_value("TRUE") is True
        assert _convert_value("yes") is True
        assert _convert_value("1") == 1  # Numeric 1

    def test_converts_false(self):
        """Should convert 'false' to False."""
        assert _convert_value("false") is False
        assert _convert_value("False") is False
        assert _convert_value("no") is False
        assert _convert_value("0") == 0  # Numeric 0

    def test_converts_none(self):
        """Should convert 'none' to None."""
        assert _convert_value("none") is None
        assert _convert_value("null") is None
        assert _convert_value("") is None

    def test_converts_int(self):
        """Should convert integer strings."""
        assert _convert_value("42") == 42
        assert _convert_value("-10") == -10

    def test_converts_float(self):
        """Should convert float strings."""
        assert _convert_value("3.14") == 3.14
        assert _convert_value("-0.5") == -0.5

    def test_converts_list(self):
        """Should convert comma-separated values to list."""
        result = _convert_value("a,b,c")
        assert result == ["a", "b", "c"]

        result = _convert_value("1,2,3")
        assert result == [1, 2, 3]

    def test_keeps_string(self):
        """Should keep string if not convertible."""
        assert _convert_value("hello") == "hello"
        assert _convert_value("hello world") == "hello world"


class TestLoadEnvVars:
    """Tests for _load_env_vars function."""

    def test_loads_prefixed_vars(self, monkeypatch):
        """Should load variables with prefix."""
        monkeypatch.setenv("PRISM_BIO_EXPERIMENT_NAME", "env_test")
        result = _load_env_vars("PRISM_BIO_")
        assert result["experiment_name"] == "env_test"

    def test_handles_nested_vars(self, monkeypatch):
        """Should handle double underscore for nesting."""
        monkeypatch.setenv("PRISM_BIO_MODEL__MODEL_NAME", "test_model")
        result = _load_env_vars("PRISM_BIO_")
        assert result["model"]["model_name"] == "test_model"

    def test_ignores_non_prefixed_vars(self, monkeypatch):
        """Should ignore variables without prefix."""
        monkeypatch.setenv("OTHER_VAR", "value")
        result = _load_env_vars("PRISM_BIO_")
        assert "other_var" not in result

    def test_converts_values(self, monkeypatch):
        """Should convert values to appropriate types."""
        monkeypatch.setenv("PRISM_BIO_CLUSTERING__N_CLUSTERS", "10")
        result = _load_env_vars("PRISM_BIO_")
        assert result["clustering"]["n_clusters"] == 10


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_default_config(self):
        """Should return default config when no args."""
        config = load_config()
        assert isinstance(config, Config)
        assert config.experiment_name == "default"

    def test_loads_yaml_config(self, tmp_path):
        """Should load config from YAML file."""
        yaml_content = {"experiment_name": "from_yaml"}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(config_path=yaml_file)
        assert config.experiment_name == "from_yaml"

    def test_cli_overrides_yaml(self, tmp_path):
        """CLI should override YAML values."""
        yaml_content = {"experiment_name": "from_yaml"}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        cli_overrides = {"experiment_name": "from_cli"}
        config = load_config(config_path=yaml_file, cli_overrides=cli_overrides)
        assert config.experiment_name == "from_cli"

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        """ENV should override YAML values."""
        yaml_content = {"experiment_name": "from_yaml"}
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        monkeypatch.setenv("PRISM_BIO_EXPERIMENT_NAME", "from_env")
        config = load_config(config_path=yaml_file)
        assert config.experiment_name == "from_env"

    def test_cli_overrides_env(self, monkeypatch):
        """CLI should override ENV values."""
        monkeypatch.setenv("PRISM_BIO_EXPERIMENT_NAME", "from_env")
        cli_overrides = {"experiment_name": "from_cli"}
        config = load_config(cli_overrides=cli_overrides)
        assert config.experiment_name == "from_cli"

    def test_handles_missing_yaml(self, tmp_path):
        """Should handle missing YAML file gracefully."""
        config = load_config(config_path=tmp_path / "nonexistent.yaml")
        assert isinstance(config, Config)

    def test_validation_disabled(self, tmp_path):
        """Should skip validation when validate=False."""
        # This would normally fail validation but shouldn't with validate=False
        config = load_config(validate=False)
        assert config is not None


class TestSaveConfig:
    """Tests for save_config function."""

    def test_saves_config_to_yaml(self, tmp_path):
        """Should save config to YAML file."""
        config = Config(experiment_name="test_save")
        save_path = tmp_path / "saved_config.yaml"

        save_config(config, save_path)

        assert save_path.exists()

    def test_saved_config_loadable(self, tmp_path):
        """Saved config should be loadable."""
        config = Config(experiment_name="test_save")
        save_path = tmp_path / "saved_config.yaml"

        save_config(config, save_path)
        loaded = load_config(config_path=save_path)

        assert loaded.experiment_name == "test_save"

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories."""
        config = Config()
        save_path = tmp_path / "a" / "b" / "config.yaml"

        save_config(config, save_path)

        assert save_path.exists()


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_returns_config_object(self):
        """Should return Config object."""
        config = create_default_config()
        assert isinstance(config, Config)

    def test_has_default_values(self):
        """Should have default values."""
        config = create_default_config()
        assert config.experiment_name == "default"
        assert config.random_seed == 42






