import logging
import inspect
from functools import wraps
from typing import Callable


def log_with_frame_info(log_method: Callable):
    @wraps(log_method)
    def wrapper(message: str):
        frame = inspect.currentframe()
        try:
            if frame and frame.f_back:
                caller = frame.f_back
                enhanced_message = (
                    f"{message} - {caller.f_code.co_filename}:{caller.f_lineno}"
                )
                return log_method(enhanced_message)
            else:
                return log_method(message)
        finally:
            del frame

    return wrapper


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Show debug too
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    @log_with_frame_info
    def debug(self, message: str):
        return self.logger.debug(message)

    @log_with_frame_info
    def info(self, message: str):
        return self.logger.info(message)

    @log_with_frame_info
    def warning(self, message: str):
        return self.logger.warning(message)

    @log_with_frame_info
    def error(self, message: str):
        return self.logger.error(message)

    @log_with_frame_info
    def critical(self, message: str):
        return self.logger.critical(message)
