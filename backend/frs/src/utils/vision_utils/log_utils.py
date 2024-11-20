import errno
import json
import logging
import os
from logging.config import dictConfig
from vision_utils.singleton_decorator import SingletonDecorator


@SingletonDecorator
class LogUtils:
    def __init__(self, log_config='log_config.json'):
        """
        @param log_config:
        """
        self.__file_path = log_config
        self.set_logger_config()

    def make_directories(self, file_path):
        """
        @param file_path:
        @return:
        """
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def set_logger_config(self):
        """
        @return:
        """
        try:
            with open(self.__file_path, 'r+') as f:
                log_config = json.load(f)
                if 'handlers' in log_config and 'file' in log_config['handlers']:
                    self.make_directories(log_config['handlers']['file']['filename'])

                dictConfig(log_config)
        except Exception as x:
            raise x

    def get_logger(self, logger_name, level=logging.DEBUG):
        """
        @param logger_name:
        @param level:
        @return:
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        return logger
