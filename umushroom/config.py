"""Typed, validated configuration.

All tunable parameters live here, loaded from a YAML file (with
environment-variable override for secrets/paths) rather than being
hardcoded inside logic modules. This is the single source of truth
for "what strategy am I running" and makes runs reproducible: you can
diff config.yaml between two runs to see exactly what changed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from umushroom.exceptions import ConfigError
from umushroom.features import TARGET_TYPES
from umushroom.universe import SUPPORTED_BASE_CURRENCIES, resolve_universe

FRENCH_REGIONS = ("US", "Developed", "Europe")
EQUITY_SLEEVE = "equity"


@dataclass(frozen=True)
class SleeveConfig:
    """A separately-modelled slice of the portfolio (v4.2).

    Each sleeve gets its OWN model, backtest and skill verdict, and a fixed
    share of capital. Assets that behave very differently (stocks vs.
    crypto vs. ETFs) are never ranked against each other: crypto's ~5x
    volatility would otherwise dominate every volatility/momentum rank.
    """
    name: str
    tickers: list[str]
    weight: float              # fraction of starting_balance
    top_n: int
    fractional: bool = False   # allow fractional units (e.g. crypto)
    neutralize_by_exchange: bool = False
    # v4.3: "model" = hold the model's top_n; "all" = hold EVERY instrument
    # in the sleeve (use when the sleeve's VERDICT shows no skill -- the
    # model is still run and reported, it just doesn't pick).
    selection: str = "model"
    # per-sleeve override of --weighting ("equal" | "hrp"); None = use CLI
    weighting: str | None = None
    # v4.4 (equity sleeve): rank within sector, and use volume factors
    neutralize_by_sector: bool = False
    use_volume_features: bool = False
    # per-sleeve forecast horizon; None = the global forward_return_days
    forward_return_days: int | None = None


@dataclass(frozen=True)
class StrategyConfig:
    tickers: list[str]
    starting_balance: float
    top_n_holdings: int
    lookback_period: str
    forward_return_days: int
    rebalance_days: int
    min_history_days: int
    model_alpha: float
    random_seed: int
    cache_dir: Path
    cache_max_age_hours: float
    log_level: str
    purged_test_window_days: int
    purged_purge_days: int
    purged_embargo_days: int
    purged_min_train_days: int
    prediction_log_path: Path
    # ---- v4 (optional; defaults keep v3 config files loadable) ----
    base_currency: str = "GBP"
    currency_overrides: dict[str, str] = field(default_factory=dict)
    target: str = "rank"
    neutralize_by_exchange: bool = True
    max_single_exchange_fraction: float = 0.6
    drop_incomplete_bar: bool = True
    fama_french_region: str = "Developed"
    # ---- v4.2: extra sleeves (crypto, ETFs, ...) ----
    sleeves: list[SleeveConfig] = field(default_factory=list)
    # ---- v4.4 ----
    neutralize_by_sector: bool = False     # equity: rank within sector
    sector_file: Path | None = None        # CSV with columns yahoo, sector
    use_volume_features: bool = False      # equity: add volume factors
    ragged_history: bool = False           # keep tickers with partial history
    # v4.7: fetch a ticker's prices from a DIFFERENT Yahoo symbol (e.g. a
    # company whose London line was delisted on Yahoo but which UMushroom
    # still prices from its US listing). Orders keep the UMushroom ticker.
    yahoo_symbols: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def equity_weight(self) -> float:
        return 1.0 - sum(s.weight for s in self.sleeves)

    @property
    def all_sleeves(self) -> list[SleeveConfig]:
        equity = SleeveConfig(
            name=EQUITY_SLEEVE, tickers=list(self.tickers), weight=self.equity_weight,
            top_n=self.top_n_holdings, fractional=False,
            neutralize_by_exchange=self.neutralize_by_exchange,
            neutralize_by_sector=self.neutralize_by_sector,
            use_volume_features=self.use_volume_features)
        return [equity, *self.sleeves]

    @property
    def all_tickers(self) -> list[str]:
        return [t for s in self.all_sleeves for t in s.tickers]


def _validate(cfg: StrategyConfig) -> None:
    errors: list[str] = []

    if not cfg.tickers:
        errors.append("tickers list is empty")
    if len(cfg.tickers) != len(set(cfg.tickers)):
        errors.append("tickers list contains duplicates")

    if cfg.starting_balance <= 0:
        errors.append(f"starting_balance must be > 0, got {cfg.starting_balance}")

    if cfg.top_n_holdings < 1:
        errors.append(f"top_n_holdings must be >= 1, got {cfg.top_n_holdings}")
    if cfg.top_n_holdings > len(cfg.tickers):
        errors.append(
            f"top_n_holdings ({cfg.top_n_holdings}) cannot exceed "
            f"the number of tickers ({len(cfg.tickers)})"
        )

    if cfg.forward_return_days < 1:
        errors.append("forward_return_days must be >= 1")
    if cfg.rebalance_days < 1:
        errors.append("rebalance_days must be >= 1")
    if cfg.min_history_days < 20:
        errors.append("min_history_days should be >= 20 for stable rolling features")

    if cfg.model_alpha < 0:
        errors.append("model_alpha (Ridge regularization) must be >= 0")

    if cfg.cache_max_age_hours < 0:
        errors.append("cache_max_age_hours must be >= 0")

    if cfg.purged_test_window_days < 1:
        errors.append("purged_test_window_days must be >= 1")
    if cfg.purged_purge_days < cfg.forward_return_days:
        errors.append(
            f"purged_purge_days ({cfg.purged_purge_days}) should be >= "
            f"forward_return_days ({cfg.forward_return_days}) to avoid "
            f"training-label leakage across the train/test boundary"
        )
    if cfg.purged_embargo_days < 0:
        errors.append("purged_embargo_days must be >= 0")
    if cfg.purged_min_train_days < cfg.min_history_days:
        errors.append(
            f"purged_min_train_days ({cfg.purged_min_train_days}) should be "
            f">= min_history_days ({cfg.min_history_days})"
        )

    if cfg.base_currency not in SUPPORTED_BASE_CURRENCIES:
        errors.append(f"base_currency must be one of "
                      f"{sorted(SUPPORTED_BASE_CURRENCIES)}, got {cfg.base_currency!r}")
    if cfg.target not in TARGET_TYPES:
        errors.append(f"target must be one of {TARGET_TYPES}, got {cfg.target!r}")
    if not 0 < cfg.max_single_exchange_fraction <= 1:
        errors.append("max_single_exchange_fraction must be in (0, 1]")
    if cfg.fama_french_region not in FRENCH_REGIONS:
        errors.append(f"fama_french_region must be one of {FRENCH_REGIONS}")
    if cfg.neutralize_by_sector:
        if cfg.sector_file is None:
            errors.append("neutralize_by_sector needs sector_file")
        elif not Path(cfg.sector_file).exists():
            errors.append(f"sector_file not found: {cfg.sector_file}")

    names = [sl.name for sl in cfg.sleeves]
    if EQUITY_SLEEVE in names or len(names) != len(set(names)):
        errors.append(f"sleeve names must be unique and not '{EQUITY_SLEEVE}'")
    for sl in cfg.sleeves:
        if not sl.tickers:
            errors.append(f"sleeve '{sl.name}' has no tickers")
        if not 0 < sl.weight < 1:
            errors.append(f"sleeve '{sl.name}' weight must be in (0, 1)")
        if sl.forward_return_days is not None and not (
                1 <= sl.forward_return_days <= cfg.purged_purge_days):
            errors.append(f"sleeve '{sl.name}' forward_return_days must be between "
                          f"1 and purged_purge_days ({cfg.purged_purge_days})")
        if sl.selection not in ("model", "all"):
            errors.append(f"sleeve '{sl.name}' selection must be 'model' or 'all'")
        if sl.weighting not in (None, "equal", "hrp"):
            errors.append(f"sleeve '{sl.name}' weighting must be 'equal' or 'hrp'")
        if not 1 <= sl.top_n <= len(sl.tickers):
            errors.append(f"sleeve '{sl.name}' top_n must be between 1 and its "
                          f"ticker count ({len(sl.tickers)})")
    if cfg.sleeves and cfg.equity_weight <= 0:
        errors.append("sleeve weights must sum to less than 1 "
                      "(the remainder goes to the equity sleeve)")
    every = cfg.tickers + [t for sl in cfg.sleeves for t in sl.tickers]
    dupes = sorted({t for t in every if every.count(t) > 1})
    if dupes and not errors:
        errors.append(f"tickers appear more than once across sleeves: {dupes}")

    unknown_symbols = set(cfg.yahoo_symbols) - set(every)
    if unknown_symbols:
        errors.append(f"yahoo_symbols for tickers not in universe: "
                      f"{sorted(unknown_symbols)}")
    unknown_overrides = set(cfg.currency_overrides) - set(every)
    if unknown_overrides:
        errors.append(f"currency_overrides for tickers not in universe: "
                      f"{sorted(unknown_overrides)}")
    try:
        resolve_universe(every, cfg.currency_overrides)
    except ConfigError as e:
        errors.append(str(e))

    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if cfg.log_level.upper() not in valid_log_levels:
        errors.append(f"log_level must be one of {valid_log_levels}")

    if errors:
        raise ConfigError(
            "Invalid configuration:\n  - " + "\n  - ".join(errors)
        )


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def load_sectors(path: Path) -> dict[str, str]:
    """yahoo ticker -> sector, from the UMushroom universe CSV."""
    import csv
    with open(path, newline="") as fh:
        return {r["yahoo"]: r["sector"] for r in csv.DictReader(fh)
                if r.get("yahoo") and r.get("sector")}


def _parse_sleeves(raw: object) -> list[SleeveConfig]:
    if not raw:
        return []
    if not isinstance(raw, dict):
        raise TypeError("sleeves must be a mapping of name -> settings")
    out = []
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise TypeError(f"sleeve {name!r} must be a mapping")
        out.append(SleeveConfig(
            name=str(name),
            tickers=list(spec.get("tickers") or []),
            weight=float(spec["weight"]),
            top_n=int(spec.get("top_n", len(spec.get("tickers") or []))),
            fractional=_to_bool(spec.get("fractional", False)),
            neutralize_by_exchange=_to_bool(spec.get("neutralize_by_exchange", False)),
            selection=str(spec.get("selection", "model")),
            forward_return_days=(None if spec.get("forward_return_days") is None
                                 else int(spec["forward_return_days"])),
            weighting=(None if spec.get("weighting") is None
                       else str(spec["weighting"])),
        ))
    return out


def load_config(path: str | Path = "config.yaml") -> StrategyConfig:
    """Load and validate strategy config from a YAML file.

    Environment variables prefixed UMUSHROOM_ override individual
    scalar fields, e.g. UMUSHROOM_STARTING_BALANCE=50000, which is
    useful for CI or running multiple contest accounts without
    editing the file.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path.resolve()}")

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping")

    required_keys = {
        "tickers", "starting_balance", "top_n_holdings", "lookback_period",
        "forward_return_days", "rebalance_days", "min_history_days",
        "model_alpha", "random_seed", "cache_dir", "cache_max_age_hours",
        "log_level", "purged_test_window_days", "purged_purge_days",
        "purged_embargo_days", "purged_min_train_days", "prediction_log_path",
    }
    missing = required_keys - raw.keys()
    if missing:
        raise ConfigError(f"Config file {path} is missing keys: {sorted(missing)}")

    for key in raw:
        env_key = f"UMUSHROOM_{key.upper()}"
        if env_key in os.environ:
            raw[key] = os.environ[env_key]

    try:
        return StrategyConfig(
            tickers=list(raw["tickers"]),
            starting_balance=float(raw["starting_balance"]),
            top_n_holdings=int(raw["top_n_holdings"]),
            lookback_period=str(raw["lookback_period"]),
            forward_return_days=int(raw["forward_return_days"]),
            rebalance_days=int(raw["rebalance_days"]),
            min_history_days=int(raw["min_history_days"]),
            model_alpha=float(raw["model_alpha"]),
            random_seed=int(raw["random_seed"]),
            cache_dir=Path(raw["cache_dir"]),
            cache_max_age_hours=float(raw["cache_max_age_hours"]),
            log_level=str(raw["log_level"]),
            purged_test_window_days=int(raw["purged_test_window_days"]),
            purged_purge_days=int(raw["purged_purge_days"]),
            purged_embargo_days=int(raw["purged_embargo_days"]),
            purged_min_train_days=int(raw["purged_min_train_days"]),
            prediction_log_path=Path(raw["prediction_log_path"]),
            base_currency=str(raw.get("base_currency", "GBP")).upper(),
            currency_overrides=dict(raw.get("currency_overrides") or {}),
            target=str(raw.get("target", "rank")),
            neutralize_by_exchange=_to_bool(raw.get("neutralize_by_exchange", True)),
            max_single_exchange_fraction=float(
                raw.get("max_single_exchange_fraction", 0.6)),
            drop_incomplete_bar=_to_bool(raw.get("drop_incomplete_bar", True)),
            fama_french_region=str(raw.get("fama_french_region", "Developed")),
            sleeves=_parse_sleeves(raw.get("sleeves")),
            neutralize_by_sector=_to_bool(raw.get("neutralize_by_sector", False)),
            sector_file=(None if raw.get("sector_file") is None
                         else (path.parent / str(raw["sector_file"]))),
            use_volume_features=_to_bool(raw.get("use_volume_features", False)),
            ragged_history=_to_bool(raw.get("ragged_history", False)),
            yahoo_symbols={str(k): str(v) for k, v in
                           (raw.get("yahoo_symbols") or {}).items()},
        )
    except (TypeError, ValueError, KeyError) as e:
        raise ConfigError(f"Config file {path} has a field of the wrong type: {e}") from e


def sleeve_horizon(cfg: StrategyConfig, sleeve: SleeveConfig) -> int:
    return sleeve.forward_return_days or cfg.forward_return_days
