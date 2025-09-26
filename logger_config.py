import logging
import traceback
import os
from datetime import datetime
from functools import wraps
from typing import Any, Callable

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure main logger
def setup_logger():
    """Set up the main application logger."""
    logger = logging.getLogger('chatbot')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler('logs/chatbot.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for errors only
    error_handler = logging.FileHandler('logs/errors.log', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logger()

def log_error(error: Exception, context: str = "", additional_info: dict = None):
    """
    Log error with full traceback and context.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        additional_info: Dictionary of additional information to log
    """
    error_msg = f"ERROR in {context}: {str(error)}"
    
    if additional_info:
        error_msg += f"\nAdditional Info: {additional_info}"
    
    error_msg += f"\nTraceback:\n{traceback.format_exc()}"
    
    logger.error(error_msg)

def log_detailed_error(error: Exception, context: str = "", local_vars: dict = None, additional_info: dict = None):
    """
    Log error with full traceback, context, and variable contents.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        local_vars: Dictionary of local variables at the time of error
        additional_info: Dictionary of additional information to log
    """
    error_msg = f"DETAILED ERROR in {context}: {str(error)}"
    error_msg += f"\nError Type: {type(error).__name__}"
    
    if additional_info:
        error_msg += f"\nAdditional Info: {additional_info}"
    
    if local_vars:
        error_msg += f"\nLocal Variables at Error:"
        for var_name, var_value in local_vars.items():
            try:
                # Safely convert variable to string, handle large objects
                var_str = str(var_value)
                if len(var_str) > 500:
                    var_str = var_str[:500] + "... (truncated)"
                error_msg += f"\n  {var_name}: {var_str}"
            except Exception as var_error:
                error_msg += f"\n  {var_name}: <Error converting to string: {var_error}>"
    
    error_msg += f"\nFull Traceback:\n{traceback.format_exc()}"
    
    logger.error(error_msg)

def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None):
    """Log function calls for debugging."""
    log_msg = f"FUNCTION CALL: {func_name}"
    if args:
        log_msg += f" - Args: {args}"
    if kwargs:
        log_msg += f" - Kwargs: {kwargs}"
    logger.info(log_msg)

def log_api_call(api_name: str, endpoint: str, status: str, duration: float = None):
    """Log API calls and their results."""
    log_msg = f"API CALL: {api_name} - {endpoint} - Status: {status}"
    if duration:
        log_msg += f" - Duration: {duration:.2f}s"
    logger.info(log_msg)

def error_handler(context: str = ""):
    """
    Decorator to automatically log errors in functions.
    
    Args:
        context: Context description for the error
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                log_error(
                    e, 
                    context=f"{context} - {func.__name__}",
                    additional_info={
                        "function": func.__name__,
                        "args": str(args)[:200],  # Limit length
                        "kwargs": str(kwargs)[:200]
                    }
                )
                raise  # Re-raise the exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                log_error(
                    e,
                    context=f"{context} - {func.__name__}",
                    additional_info={
                        "function": func.__name__,
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200]
                    }
                )
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # Check if coroutine
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def log_user_interaction(session_id: str, user_message: str, bot_response: str, intent: str = None):
    """Log user interactions for analysis."""
    interaction_log = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_message": user_message[:200],  # Limit length
        "bot_response": bot_response[:200],
        "intent": intent
    }
    logger.info(f"USER INTERACTION: {interaction_log}")

def log_search_query(session_id: str, query: str, results_count: int, success: bool):
    """Log search queries and results."""
    search_log = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "query": query,
        "results_count": results_count,
        "success": success
    }
    logger.info(f"SEARCH QUERY: {search_log}")

# Example usage:
# from logger_config import logger, log_error, error_handler

# @error_handler("Data Loading")
# async def load_data():
#     # Your function code here
#     pass