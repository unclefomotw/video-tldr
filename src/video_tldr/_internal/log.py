import logging
from pathlib import Path


def setup_file_logger(log_dir='/tmp/video_tldr/', log_file='app.log', log_level=logging.INFO):
    """
    Sets up a file logger that includes source file and function information.

    Args:
        log_dir (str): Directory where the log files will be stored.
        log_file (str): Path to the log file
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        logging.Logger: Configured logger object
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # Configure logger
    logger = logging.getLogger('file_logger')

    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # Create file handler
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(log_level)

    # Create formatter with file and function information
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger
