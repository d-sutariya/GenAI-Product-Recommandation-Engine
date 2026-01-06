"""
Logging utility for MCP Server
Logs to stderr to avoid interfering with JSON-RPC communication on stdout
"""

import sys
from enum import Enum
from datetime import datetime
from typing import Optional


class LogLevel(str, Enum):
    """Log levels for structured logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class MCPLogger:
    """
    Logger for MCP server that writes to stderr
    to avoid interfering with JSON-RPC communication
    """
    
    def __init__(self, name: str = "MCP-Server"):
        self.name = name
        
    def _log(self, level: LogLevel, message: str, context: Optional[dict] = None) -> None:
        """
        Internal logging method that writes to stderr
        
        Args:
            level: Log level
            message: Log message
            context: Optional context dictionary for additional information
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{self.name}] {level.value}: {message}"
        
        if context:
            log_entry += f" | Context: {context}"
            
        sys.stderr.write(f"{log_entry}\n")
        sys.stderr.flush()
    
    def debug(self, message: str, context: Optional[dict] = None) -> None:
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, context)
    
    def info(self, message: str, context: Optional[dict] = None) -> None:
        """Log info message"""
        self._log(LogLevel.INFO, message, context)
    
    def warn(self, message: str, context: Optional[dict] = None) -> None:
        """Log warning message"""
        self._log(LogLevel.WARN, message, context)
    
    def error(self, message: str, context: Optional[dict] = None) -> None:
        """Log error message"""
        self._log(LogLevel.ERROR, message, context)
    
    def success(self, message: str, context: Optional[dict] = None) -> None:
        """Log success message"""
        self._log(LogLevel.SUCCESS, message, context)


# Global logger instance
logger = MCPLogger()
