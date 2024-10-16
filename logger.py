import sys
import logging


class Logger:

    def __init__(self, filename, level=logging.INFO):
        logger = logging.getLogger()
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s %(message)s')

        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.handler = (file_handler, stream_handler)

    def remove_handler(self):
        for handler in self.handler:
            logger = logging.getLogger()
            logger.removeHandler(handler)
