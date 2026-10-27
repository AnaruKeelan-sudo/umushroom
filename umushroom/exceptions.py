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


class StaleDataError(DataError):
    """v4.21: the latest close in the data is older than the last trading day.
    Building signals on it would trade on stale prices (the 2026-10-21..27
    GitHub runs did exactly that: Yahoo had not yet published the previous
    day's European closes at 07:00 UTC). The CLI exits with code 3 so the
    website workflow can retry later instead of publishing a stale run."""
