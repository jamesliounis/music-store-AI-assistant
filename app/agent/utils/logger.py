# agent/utils/logger.py

import logging
from logging.handlers import RotatingFileHandler
import os


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with both console and file handlers.

    This function sets up a logger that logs messages to both the console
    (with INFO level) and a rotating file (with DEBUG level). It ensures
    that logs are stored in a "logs" directory and rotates log files
    when they reach 5MB, keeping up to two backup logs.

    Logging setup includes:
    - **Console Logging** (`StreamHandler`): Displays logs at `INFO` level or higher.
    - **File Logging** (`RotatingFileHandler`): Stores logs at `DEBUG` level or higher,
      rotating the log file (`app.log`) when it exceeds 5MB, keeping up to two backups.

    Args:
        name (str): The name of the logger, typically the module name.

    Returns:
        logging.Logger: A configured logger instance.

    Notes:
        - If the logger has existing handlers, new handlers will not be added.
        - The "logs" directory is created if it does not exist.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create log directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"), maxBytes=5 * 1024 * 1024, backupCount=2
    )
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger if not already added
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger
