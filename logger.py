import os
import logging
import inspect
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_FILE = "logs/tiny_aristotle.log"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.DEBUG
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Number of backup files to keep


def get_logger(
    log_file=DEFAULT_LOG_FILE,
    log_level=DEFAULT_LOG_LEVEL,
    log_format=DEFAULT_LOG_FORMAT,
):
    """
    Configures and returns a logger with the specified log file, level, and format.

    Args:
        log_file (str): The name of the log file. Defaults to 'tiny_aristotle.log'.
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO). Defaults to DEBUG.
        log_format (str): Format of the log messages. Defaults to a detailed format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Get the filename of the script that called this function
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame.filename
    name = os.path.basename(caller_filename)
    logger_name = f"tiny-aristotle:{name}"

    logger = logging.getLogger(logger_name)
    logger.propagate = False

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=MAX_LOG_FILE_SIZE, backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
