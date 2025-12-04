"""
Logging configuration module for training.
Provides consistent logging across all training components.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "GreenNetwork",
    level: int = logging.INFO,
    log_file: str = None,
    console: bool = True,
) -> logging.Logger:
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs
        console: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get an existing logger or create a default one.
    
    Args:
        name: Logger name (if None, returns root GreenNetwork logger)
    
    Returns:
        Logger instance
    """
    if name is None:
        name = "GreenNetwork"
    
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        setup_logger(name)
    
    return logger


# Pre-configured loggers for different modules
def get_training_logger(log_dir: str = "logs") -> logging.Logger:
    """Get logger for training module with file output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_{timestamp}.log"
    return setup_logger("GreenNetwork.Train", log_file=log_file)


def get_env_logger() -> logging.Logger:
    """Get logger for environment module."""
    return setup_logger("GreenNetwork.Env", level=logging.INFO)


def get_agent_logger() -> logging.Logger:
    """Get logger for agent module."""
    return setup_logger("GreenNetwork.Agent", level=logging.INFO)
