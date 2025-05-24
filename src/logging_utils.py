import functools
import logging
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


def log_calls(func: F) -> F:
    """Decorator to log function entry and exit."""
    logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info("Entering %s with args=%s, kwargs=%s", func.__name__, args, kwargs)
        result = func(*args, **kwargs)
        logger.info("Exiting %s", func.__name__)
        return result

    return wrapper  # type: ignore


def log_all_methods(cls: type) -> type:
    """Class decorator to apply log_calls to all methods."""
    for name, attr in cls.__dict__.items():
        if callable(attr) and not name.startswith("__"):
            setattr(cls, name, log_calls(attr))
    return cls
