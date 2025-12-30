"""Configuration loader with YAML, CLI, and ENV support."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.config.schema import Config
from src.utils.logging_utils import get_logger

logger = get_logger("config.loader")


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of configuration values
    """
    logger.debug(f"Loading YAML config from: {path}")

    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    if content is None:
        logger.warning(f"Empty config file: {path}")
        return {}

    logger.debug(f"Loaded {len(content)} top-level keys from {path}")
    return content


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _parse_env_key(key: str, prefix: str) -> list:
    """Parse environment variable key to nested config path.

    Example: PRISM_BIO_MODEL__MODEL_NAME -> ["model", "model_name"]

    Args:
        key: Environment variable key
        prefix: Prefix to strip

    Returns:
        List of nested keys
    """
    # Remove prefix
    config_key = key[len(prefix) :]

    # Split by double underscore for nesting
    parts = config_key.lower().split("__")

    return parts


def _convert_value(value: str) -> Any:
    """Convert string value to appropriate Python type.

    Args:
        value: String value from environment

    Returns:
        Converted value (bool, int, float, or string)
    """
    # Boolean
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # None
    if value.lower() in ("none", "null", ""):
        return None

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # List (comma-separated)
    if "," in value:
        items = [item.strip() for item in value.split(",")]
        # Try to convert each item
        return [_convert_value(item) for item in items]

    # String
    return value


def _load_env_vars(prefix: str = "PRISM_BIO_") -> Dict[str, Any]:
    """Load configuration from environment variables.

    Environment variables should be prefixed with PRISM_BIO_ and use
    double underscores for nested keys.

    Examples:
        PRISM_BIO_EXPERIMENT_NAME=my_exp
        PRISM_BIO_MODEL__MODEL_NAME=facebook/esm2_t6_8M_UR50D
        PRISM_BIO_CLUSTERING__N_CLUSTERS=10

    Args:
        prefix: Environment variable prefix

    Returns:
        Dictionary of configuration values
    """
    result: Dict[str, Any] = {}

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        parts = _parse_env_key(key, prefix)
        if not parts or not parts[0]:
            continue

        logger.debug(f"Found env var: {key} = {value[:50]}...")

        # Navigate/create nested structure
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value with type conversion
        current[parts[-1]] = _convert_value(value)

    if result:
        logger.debug(f"Loaded {len(result)} config values from environment")

    return result


def load_config(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    env_prefix: str = "PRISM_BIO_",
    validate: bool = True,
) -> Config:
    """Load configuration with priority: CLI > ENV > YAML > Defaults.

    Args:
        config_path: Path to YAML config file
        cli_overrides: Dictionary of CLI argument overrides
        env_prefix: Prefix for environment variables
        validate: Whether to validate the config

    Returns:
        Validated Config object
    """
    logger.info("Loading configuration...")

    config_dict: Dict[str, Any] = {}

    # 1. Load YAML if provided
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            yaml_config = load_yaml(config_path)
            config_dict = deep_merge(config_dict, yaml_config)
            logger.info(f"Loaded config from: {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}")

    # 2. Load environment variables
    env_config = _load_env_vars(env_prefix)
    if env_config:
        config_dict = deep_merge(config_dict, env_config)
        logger.debug("Applied environment variable overrides")

    # 3. Apply CLI overrides
    if cli_overrides:
        config_dict = deep_merge(config_dict, cli_overrides)
        logger.debug("Applied CLI overrides")

    # 4. Validate and create Config object
    if validate:
        try:
            config = Config(**config_dict)
            logger.info("Configuration validated successfully")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    else:
        config = Config.model_construct(**config_dict)
        logger.warning("Configuration loaded without validation")

    return config


def save_config(config: Config, path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object
        path: Output path
    """
    logger.info(f"Saving configuration to: {path}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump()

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to: {path}")


def create_default_config() -> Config:
    """Create a default configuration.

    Returns:
        Default Config object
    """
    return Config()






