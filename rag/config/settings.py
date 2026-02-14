"""Configuration service for reading and managing INI configuration files."""

import configparser
import logging
from typing import Optional, Any
from pathlib import Path


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class ConfigService:
    """Service for reading configuration from INI files."""
    
    def __init__(self, config_file: str):
        """
        Initialize the configuration service.
        
        Args:
            config_file: Path to the INI configuration file
            
        Raises:
            ConfigError: If the configuration file cannot be read
        """
        self.config_file = Path(config_file)
        self.parser = configparser.ConfigParser()
        
        if not self.config_file.exists():
            raise ConfigError(f"Configuration file not found: {config_file}")
        
        try:
            self.parser.read(self.config_file)
        except Exception as e:
            raise ConfigError(f"Error reading configuration file: {e}")
    
    def get_string(self, section: str, key: str, default: Optional[str] = None) -> str:
        """
        Get a string value from the configuration.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            The configuration value
            
        Raises:
            ConfigError: If key not found and no default provided
        """
        try:
            return self.parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            raise ConfigError(f"Configuration key not found: [{section}] {key}") from e
    
    def get_int(self, section: str, key: str, default: Optional[int] = None) -> int:
        """Get an integer value from the configuration."""
        try:
            return self.parser.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            raise ConfigError(f"Configuration key not found: [{section}] {key}") from e
        except ValueError as e:
            raise ConfigError(f"Invalid integer value for [{section}] {key}") from e
    
    def get_float(self, section: str, key: str, default: Optional[float] = None) -> float:
        """Get a float value from the configuration."""
        try:
            return self.parser.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            raise ConfigError(f"Configuration key not found: [{section}] {key}") from e
        except ValueError as e:
            raise ConfigError(f"Invalid float value for [{section}] {key}") from e
    
    def get_boolean(self, section: str, key: str, default: Optional[bool] = None) -> bool:
        """Get a boolean value from the configuration."""
        try:
            return self.parser.getboolean(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            raise ConfigError(f"Configuration key not found: [{section}] {key}") from e
        except ValueError as e:
            raise ConfigError(f"Invalid boolean value for [{section}] {key}") from e
    
    def has_option(self, section: str, key: str) -> bool:
        """Check if a configuration option exists."""
        return self.parser.has_option(section, key)
    
    def get_list(self, section: str, key: str, separator: str = ',', 
                 default: Optional[list] = None) -> list:
        """
        Get a list value from the configuration.
        
        Args:
            section: Configuration section
            key: Configuration key
            separator: Character to split the value on
            default: Default value if key not found
            
        Returns:
            List of string values
        """
        try:
            value = self.parser.get(section, key)
            return [item.strip() for item in value.split(separator)]
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if default is not None:
                return default
            raise ConfigError(f"Configuration key not found: [{section}] {key}") from e
