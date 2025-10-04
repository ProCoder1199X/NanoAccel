"""
Configuration management for NanoAccel.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration management class for NanoAccel.
    
    Supports loading configuration from files (JSON/YAML) and environment variables.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._get_default_config()
        
        if self.config_path and self.config_path.exists():
            self.load_from_file(self.config_path)
        
        # Override with environment variables
        self.load_from_env()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "default_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "default_draft_model": "EleutherAI/pythia-70m",
                "cache_dir": None,
                "trust_remote_code": False
            },
            "quantization": {
                "enabled": False,
                "quant_type": "int8",
                "compute_dtype": "float32",
                "chunk_size": 1024
            },
            "generation": {
                "max_tokens": 50,
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True
            },
            "speculative_decoding": {
                "enabled": False,
                "gamma": 4,
                "early_exit_threshold": 0.9,
                "use_efficiency_cores": True
            },
            "system": {
                "cpu_optimization": True,
                "mixed_precision": False,
                "num_threads": None,
                "memory_fraction": 0.8
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None
            },
            "cache": {
                "enabled": True,
                "cache_dir": None,
                "max_size_gb": 10
            }
        }
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Deep merge configuration
            self._deep_merge(self.config, file_config)
            logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "NANOACCEL_MODEL": ("model", "default_model"),
            "NANOACCEL_DRAFT_MODEL": ("model", "default_draft_model"),
            "NANOACCEL_QUANT_ENABLED": ("quantization", "enabled"),
            "NANOACCEL_QUANT_TYPE": ("quantization", "quant_type"),
            "NANOACCEL_MAX_TOKENS": ("generation", "max_tokens"),
            "NANOACCEL_TEMPERATURE": ("generation", "temperature"),
            "NANOACCEL_TOP_P": ("generation", "top_p"),
            "NANOACCEL_TOP_K": ("generation", "top_k"),
            "NANOACCEL_SPECULATIVE": ("speculative_decoding", "enabled"),
            "NANOACCEL_GAMMA": ("speculative_decoding", "gamma"),
            "NANOACCEL_CPU_OPTIMIZATION": ("system", "cpu_optimization"),
            "NANOACCEL_MIXED_PRECISION": ("system", "mixed_precision"),
            "NANOACCEL_LOG_LEVEL": ("logging", "level"),
            "NANOACCEL_CACHE_DIR": ("cache", "cache_dir"),
            "NANOACCEL_NUM_THREADS": ("system", "num_threads")
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(value)
                self.config[section][key] = converted_value
                logger.debug(f"Set {section}.{key} = {converted_value} from {env_var}")
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key (optional)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if key is None:
            return self.config.get(section, default)
        
        section_config = self.config.get(section, {})
        return section_config.get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def save_to_file(self, config_path: Union[str, Path], format: str = "json") -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: File format ("json" or "yaml")
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == "yaml":
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved configuration to: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
        """
        self._deep_merge(self.config, updates)
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate model configuration
            model_config = self.config.get("model", {})
            if not isinstance(model_config.get("default_model"), str):
                logger.error("Invalid default_model: must be a string")
                return False
            
            # Validate quantization configuration
            quant_config = self.config.get("quantization", {})
            if quant_config.get("enabled", False):
                quant_type = quant_config.get("quant_type")
                if quant_type not in ["int2", "int4", "int8"]:
                    logger.error(f"Invalid quant_type: {quant_type}")
                    return False
            
            # Validate generation configuration
            gen_config = self.config.get("generation", {})
            if not (0 < gen_config.get("temperature", 1.0) <= 2.0):
                logger.error("Invalid temperature: must be between 0 and 2")
                return False
            
            if not (0 < gen_config.get("top_p", 0.9) <= 1.0):
                logger.error("Invalid top_p: must be between 0 and 1")
                return False
            
            if gen_config.get("top_k", 50) <= 0:
                logger.error("Invalid top_k: must be positive")
                return False
            
            # Validate speculative decoding configuration
            spec_config = self.config.get("speculative_decoding", {})
            if spec_config.get("gamma", 4) <= 0:
                logger.error("Invalid gamma: must be positive")
                return False
            
            if not (0 < spec_config.get("early_exit_threshold", 0.9) <= 1.0):
                logger.error("Invalid early_exit_threshold: must be between 0 and 1")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False


# Global configuration instance
_global_config = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Configuration instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config(config_path)
    
    return _global_config


def set_config(config: Config) -> None:
    """
    Set global configuration instance.
    
    Args:
        config: Configuration instance
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset global configuration instance."""
    global _global_config
    _global_config = None
