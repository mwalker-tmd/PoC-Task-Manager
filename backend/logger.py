import logging
import os
from typing import Optional

# Get log level from environment variable, default to INFO
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Create logger instance but don't configure it yet
logger = logging.getLogger("task_agent")

def initialize_logger() -> None:
    """Initialize the logger with configuration from environment variables."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(LOG_LEVELS.get(log_level, logging.INFO))

    # Create console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVELS.get(log_level, logging.INFO))
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)

def set_log_level(level: str) -> None:
    """Set the log level for the task agent logger."""
    level = level.upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"Invalid log level. Must be one of: {', '.join(LOG_LEVELS.keys())}")
    
    logger.setLevel(LOG_LEVELS[level])
    for handler in logger.handlers:
        handler.setLevel(LOG_LEVELS[level])

def get_log_level() -> str:
    """Get the current log level."""
    return logging.getLevelName(logger.level) 