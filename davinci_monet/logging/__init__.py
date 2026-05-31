"""Structured logging for DAVINCI.

This module provides a consistent logging interface for the package,
replacing print() statements with proper structured logging.

Usage:
    from davinci_monet.logging import get_logger, configure_logging

    # Get a logger for your module
    logger = get_logger(__name__)

    # Configure logging at startup
    configure_logging(level="INFO", log_file="output.log")

    # Use the logger
    logger.info("Processing started", extra={"model": "CMAQ", "obs": "AQS"})
"""

from davinci_monet.logging.config import LogLevel, configure_logging, get_logger, set_log_level

__all__ = [
    "configure_logging",
    "get_logger",
    "set_log_level",
    "LogLevel",
]
