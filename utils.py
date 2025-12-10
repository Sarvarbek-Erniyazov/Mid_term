

import logging
import os
from datetime import datetime


LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
LOG_FILE = os.path.join(LOG_DIR, datetime.now().strftime('project_%Y_%m_%d_%H_%M_%S.log'))

def setup_logger(name, log_file=LOG_FILE, level=logging.INFO):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    
    if logger.handlers:
        return logger

    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


main_logger = setup_logger('MAIN_PROJECT')