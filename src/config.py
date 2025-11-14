"""Configuration management for FeatureForge."""

import os
import yaml
from typing import Dict, Any


class Config:
    """Load and manage configuration settings."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dot notation, e.g., 'data.raw_path')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.config[key]

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"
