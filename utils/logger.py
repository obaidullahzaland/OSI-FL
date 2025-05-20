import logging

def setup_logger(name, log_file, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a formatter and set the formatter for the handler.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    return logger
