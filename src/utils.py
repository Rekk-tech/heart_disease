"""
Utility functions for heart disease prediction project.
"""

import logging
import random
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        
    Returns:
        Configured logger
    """
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)  # Tạo thư mục nếu chưa tồn tại
            
    try:
        # Default log format
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        
        # Create logger
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured with level: {log_level}")
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Try to set seed for other libraries if available
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        logging.getLogger(__name__).info(f"Random seed set to: {seed}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error setting seed: {e}")
        raise


def load_config(config_path: str = "conf/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logging.getLogger(__name__).info(f"Configuration loaded from {config_path}")
        return config
        
    except FileNotFoundError:
        logging.getLogger(__name__).error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading configuration: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str = "conf/config.yaml") -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        # Create directory if it doesn't exist
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logging.getLogger(__name__).info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving configuration: {e}")
        raise


def create_directories(directories: list) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logging.getLogger(__name__).info(f"Created directories: {directories}")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating directories: {e}")
        raise


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def print_separator(char: str = "=", length: int = 50) -> None:
    """
    Print a separator line.
    
    Args:
        char: Character to use for separator
        length: Length of separator line
    """
    print(char * length)


def print_section_header(title: str, char: str = "=", length: int = 50) -> None:
    """
    Print a section header.
    
    Args:
        title: Section title
        char: Character to use for separator
        length: Length of separator line
    """
    print_separator(char, length)
    print(f" {title} ".center(length, char))
    print_separator(char, length)


def validate_environment() -> Dict[str, bool]:
    """
    Validate that required packages are installed.
    
    Returns:
        Dictionary with package availability status
    """
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'yaml': 'PyYAML',
        'joblib': 'joblib'
    }
    
    optional_packages = {
        'shap': 'shap',
        'lime': 'lime',
        'plotly': 'plotly'
    }
    
    status = {}
    
    # Check required packages
    for name, package in required_packages.items():
        try:
            __import__(name)
            status[package] = True
        except ImportError:
            status[package] = False
    
    # Check optional packages
    for name, package in optional_packages.items():
        try:
            __import__(name)
            status[package] = True
        except ImportError:
            status[package] = False
    
    return status


def print_environment_info() -> None:
    """
    Print environment information including package versions.
    """
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        
        print_section_header("Environment Information")
        
        print(f"Python version: {pd.__version__}")
        print(f"Pandas version: {pd.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Scikit-learn version: {sklearn.__version__}")
        
        # Check optional packages
        try:
            import matplotlib
            print(f"Matplotlib version: {matplotlib.__version__}")
        except ImportError:
            print("Matplotlib: Not installed")
        
        try:
            import seaborn
            print(f"Seaborn version: {seaborn.__version__}")
        except ImportError:
            print("Seaborn: Not installed")
        
        print_separator()
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error printing environment info: {e}")


def get_memory_usage() -> Dict[str, float]:
    """
    Get memory usage information.
    
    Returns:
        Dictionary with memory usage statistics
    """
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percentage': memory.percent
        }
    except ImportError:
        logging.getLogger(__name__).warning("psutil not available for memory monitoring")
        return {}


def check_disk_space(path: str = ".") -> Dict[str, float]:
    """
    Check available disk space.
    
    Args:
        path: Path to check disk space for
        
    Returns:
        Dictionary with disk space information
    """
    try:
        import shutil
        
        total, used, free = shutil.disk_usage(path)
        
        return {
            'total_gb': total / (1024**3),
            'used_gb': used / (1024**3),
            'free_gb': free / (1024**3),
            'free_percentage': (free / total) * 100
        }
    except Exception as e:
        logging.getLogger(__name__).error(f"Error checking disk space: {e}")
        return {}


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    
    return safe_name


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure that the directory for a file path exists.
    
    Args:
        file_path: Path to file
    """
    directory = Path(file_path).parent
    directory.mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        Human-readable file size
    """
    try:
        size_bytes = Path(file_path).stat().st_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} TB"
    except Exception:
        return "Unknown"


def print_progress_bar(iteration: int, 
                      total: int, 
                      prefix: str = 'Progress', 
                      suffix: str = 'Complete',
                      length: int = 50) -> None:
    """
    Print a progress bar.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix text
        suffix: Suffix text
        length: Length of progress bar
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    # Print new line on complete
    if iteration == total:
        print()


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function call with parameters.
    
    Args:
        func_name: Name of the function
        **kwargs: Function parameters
    """
    logger = logging.getLogger(__name__)
    params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params_str})")
