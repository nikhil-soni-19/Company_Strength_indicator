"""Centralized logging configuration."""

import logging
import sys
from typing import Optional
from settings import settings


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = getattr(logging, (level or settings.logging.level).upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter(settings.logging.format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if settings.logging.log_file:
        file_handler = logging.FileHandler(settings.logging.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger