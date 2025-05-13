import logging


def set_logger(out_dir):
    # Configure the root logger to log both to a file and the console
    log_format = '%(asctime)s [%(levelname)s] %(module)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    # Create a logger and set its level
    logger = logging.getLogger('example')
    logger.setLevel(logging.DEBUG)
    # Create a console handler and set its level and formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # Create a file handler and set its level and formatter
    file_handler = logging.FileHandler(out_dir / 'log.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger