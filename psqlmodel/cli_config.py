"""
PSQLModel CLI Configuration - Connection Profile Storage.

Manages saved database connection profiles in ~/.psqlmodel/config.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


CONFIG_DIR = Path.home() / ".psqlmodel"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _ensure_config_dir() -> None:
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_config() -> Dict[str, Any]:
    """Load config from file."""
    if not CONFIG_FILE.exists():
        return {"profiles": {}, "default": None}
    
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            # Ensure required keys exist
            if "profiles" not in config:
                config["profiles"] = {}
            if "default" not in config:
                config["default"] = None
            return config
    except (json.JSONDecodeError, IOError):
        return {"profiles": {}, "default": None}


def _save_config(config: Dict[str, Any]) -> None:
    """Save config to file."""
    _ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def save_profile(
    name: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    host: str = "localhost",
    port: int = 5432,
    database: Optional[str] = None,
    models_path: Optional[str] = None,
    migrations_path: Optional[str] = None,
    set_default: bool = False,
) -> None:
    """
    Save a connection profile.
    
    Args:
        name: Profile name (identifier)
        username: Database username
        password: Database password
        host: Database host
        port: Database port
        database: Database name
        models_path: Path to models
        migrations_path: Path to migrations
        set_default: If True, set as default profile
    """
    config = _load_config()
    
    profile = {
        "username": username,
        "password": password,
        "host": host,
        "port": port,
        "database": database,
        "models_path": models_path,
        "migrations_path": migrations_path,
    }
    
    # Remove None values
    profile = {k: v for k, v in profile.items() if v is not None}
    
    config["profiles"][name] = profile
    
    if set_default or len(config["profiles"]) == 1:
        config["default"] = name
    
    _save_config(config)


def remove_profile(name: str) -> bool:
    """
    Remove a connection profile.
    
    Returns:
        True if profile was removed, False if not found
    """
    config = _load_config()
    
    if name not in config["profiles"]:
        return False
    
    del config["profiles"][name]
    
    # If was default, clear default
    if config["default"] == name:
        config["default"] = None
        # Set first remaining profile as default if any
        if config["profiles"]:
            config["default"] = next(iter(config["profiles"]))
    
    _save_config(config)
    return True


def list_profiles() -> Dict[str, Dict[str, Any]]:
    """
    List all saved profiles.
    
    Returns:
        Dict mapping profile name to profile data
    """
    config = _load_config()
    return config.get("profiles", {})


def get_profile(name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get a specific profile or the default one.
    
    Args:
        name: Profile name, or None to get default
        
    Returns:
        Profile dict or None if not found
    """
    config = _load_config()
    
    if name is None:
        name = config.get("default")
    
    if name is None:
        return None
    
    return config["profiles"].get(name)


def set_default_profile(name: str) -> bool:
    """
    Set a profile as default.
    
    Returns:
        True if successful, False if profile not found
    """
    config = _load_config()
    
    if name not in config["profiles"]:
        return False
    
    config["default"] = name
    _save_config(config)
    return True


def get_default_profile_name() -> Optional[str]:
    """Get the name of the default profile."""
    config = _load_config()
    return config.get("default")
