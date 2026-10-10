"""Exception hierarchy.

Using specific exception types (rather than bare Exception / generic
ValueError everywhere) lets callers catch precisely what they expect
and lets tests assert on failure modes explicitly.
"""


class UMushroomError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(UMushroomError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class DataError(UMushroomError):
    """Raised when market data cannot be fetched or fails validation."""


class InsufficientHistoryError(DataError):
    """Raised when there is not enough historical data to build features
    or train safely."""


class ModelNotFittedError(UMushroomError):
    """Raised when predict() is called before fit()."""
