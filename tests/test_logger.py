import pytest
import logging
from unittest.mock import patch, MagicMock
from Logger import Logger, log_with_frame_info


class TestLogger:
    def test_logger_creation(self):
        logger = Logger("test_logger")
        assert logger.logger.name == "test_logger"
        assert logger.logger.level == logging.DEBUG

    def test_logger_methods_exist(self):
        logger = Logger("test_logger")
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')

    @patch('logging.StreamHandler')
    def test_logger_handler_setup(self, mock_handler):
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        
        logger = Logger("test_logger")
        
        mock_handler.assert_called_once()
        mock_handler_instance.setFormatter.assert_called_once()

    def test_log_with_frame_info_decorator(self):
        mock_log_method = MagicMock()
        decorated_method = log_with_frame_info(mock_log_method)
        
        # Create a mock self object
        mock_self = MagicMock()
        decorated_method(mock_self, "test message")
        
        mock_log_method.assert_called_once()
        call_args = mock_log_method.call_args[0][1]  # Second argument is the message
        assert "test message" in call_args