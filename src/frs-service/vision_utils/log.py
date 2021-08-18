import json
import logging
import os
import sys
import errno
from logging.handlers import RotatingFileHandler
from logging.config import dictConfig

# todo: need to take config from yml file
FORMATTER = logging.Formatter("%(asctime)s - %(name)s.%(funcName)s +%(lineno)s: %(levelname)s - %(message)s")
LOG_FILE = "app.log"
LOG_MAX_SIZE = 2 * 1024 * 1024  # 2 MB
LOG_BACKUP_COUNT = 30  # max 30 log backup files
LOG_FILE_NAME = 'logs/app.log'
ERROR_LOG_FILE_NAME = 'logs/app_error.log'


def make_directories(file_path):
    '''
    It'll make directories if not exists!
    Returns
    -------

    '''
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


try:
    with open('log_config.json', 'r+') as f:
        log_config = json.load(f)
        if 'handlers' in log_config and 'file' in log_config['handlers']:
            make_directories(log_config['handlers']['file']['filename'])

        dictConfig(log_config)
except Exception as x:
    raise x


def get_logger_from_config(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger


def get_console_handler(level=logging.DEBUG):
    """
    @param level:
    @return:
    """
    # Stream Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(FORMATTER)

    return handler


def get_file_handler(level=logging.DEBUG):
    """
    @param level:
    @return:
    """
    log_dir = os.path.dirname(LOG_FILE_NAME)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Rotating File Handler
    handler = RotatingFileHandler(LOG_FILE_NAME, maxBytes=LOG_MAX_SIZE, backupCount=LOG_BACKUP_COUNT)
    handler.setLevel(level)
    handler.setFormatter(FORMATTER)

    return handler


def get_logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)

    logger.setLevel(level)  # better to have too much log than not enough

    logger.addHandler(get_console_handler(level))
    logger.addHandler(get_file_handler(level))

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False

    return logger
