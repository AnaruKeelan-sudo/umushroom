"""Alpha measurement by factor regression.

    r_strategy - r_f = alpha + b1 * F1 + b2 * F2 + ... + e

ALPHA (the intercept) is the return NOT explained by exposure to known,
cheaply-available risk factors. A strategy that merely loads on the
market or on small caps has a high raw return but zero alpha; judges
(and the literature) care about the intercept and its t-statistic.

Two regressions are provided:

  benchmark_alpha  -- strategy vs. the equal-weight portfolio of its OWN
      universe, using the period returns from the purged walk-forward.
      Needs no extra data, and is the most directly relevant question:
      "did the stock picks beat simply holding everything?"

  fama_french_alpha -- strategy excess return vs. the Fama-French
      factors (Mkt-RF, SMB, HML; optionally RMW, CMA) from Kenneth
      French's data library. Region is configurable; for a mixed
      UK/US/Swiss universe "Developed" is the closest match.
      CAVEAT: French's factors are USD returns; strategy returns are in
      the account's base currency. The currency mismatch adds noise
      (it does not bias the intercept unless FX trended strongly over
      the sample), and is printed in the summary.

Standard errors are Newey-West (HAC). Overlapping or autocorrelated
period returns make plain OLS t-stats too optimistic -- HAC corrects
that. Implemented directly with numpy (no statsmodels dependency, which
keeps the package installable on brand-new Python releases).
"""

from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from umushroom.exceptions import DataError
from umushroom.logging_config import get_logger

logger = get_logger("factor_regression")

FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_DATASETS = {
    "US": "F-F_Research_Data_Factors_daily",
    "Developed": "Developed_3_Factors_Daily",
    "Europe": "Europe_3_Factors_Daily",
}
MIN_REGRESSION_OBS = 12


@dataclass(frozen=True)
class AlphaResult:
    alpha_per_period: float
    alpha_annualized: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: dict[str, float]
    beta_tstats: dict[str, float]
    r_squared: float
    n_obs: int
    nw_lags: int
    label: str
    caveat: str = ""

    def summary(self) -> str:
        lines = [
            f"{self.label}",
            (f"  Alpha (annualised):  {self.alpha_annualized:+.2%}   "
             f"t = {self.alpha_tstat:+.2f}   p = {self.alpha_pvalue:.3f}"),
            f"  Alpha (per period):  {self.alpha_per_period:+.4%}",
        ]
        for name, b in self.betas.items():
            lines.append(f"  beta[{name}]:{'':<{max(1, 12 - len(name))}}{b:+.3f}   "
                         f"t = {self.beta_tstats[name]:+.2f}")
        lines.append(f"  R^2 = {self.r_squared:.3f},  n = {self.n_obs} periods,  "
                     f"Newey-West lags = {self.nw_lags}")
        sig = abs(self.alpha_tstat) >= 2.0
        lines.append("  => alpha is statistically significant (|t| >= 2)" if sig else
                     "  => alpha is NOT statistically distinguishable from zero")
        if self.caveat:
            lines.append(f"  Caveat: {self.caveat}")
        return "\n".join(lines)


def newey_west_default_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS coefficients, Newey-West standard errors, R^2.
    X must already include a constant column."""
    n, k = X.shape
    if n <= k:
        raise ValueError(f"Need more observations ({n}) than regressors ({k})")
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)          # Bartlett kernel
        gamma = u[lag:].T @ u[:-lag]
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv * (n / (n - k))   # small-sample scaling
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return beta, se, r2


def regress_alpha(
    returns: pd.Series,
    factors: pd.DataFrame,
    periods_per_year: float,
    label: str,
    nw_lags: int | None = None,
    caveat: str = "",
) -> AlphaResult:
    df = pd.concat([returns.rename("__y__"), factors], axis=1, join="inner").dropna()
    if len(df) < MIN_REGRESSION_OBS:
        raise ValueError(
            f"Only {len(df)} overlapping observations; need >= {MIN_REGRESSION_OBS} "
            f"for a meaningful alpha regression"
        )
    y = df["__y__"].to_numpy(dtype=float)
    fac_names = [c for c in df.columns if c != "__y__"]
    X = np.column_stack([np.ones(len(df)), df[fac_names].to_numpy(dtype=float)])
    lags = newey_west_default_lags(len(df)) if nw_lags is None else nw_lags
    coef, se, r2 = ols_hac(y, X, lags)
    t = np.divide(coef, se, out=np.full_like(coef, np.nan), where=se > 0)
    dof = len(df) - X.shape[1]
    p_alpha = float(2 * stats.t.sf(abs(t[0]), dof)) if np.isfinite(t[0]) else float("nan")
    return AlphaResult(
        alpha_per_period=float(coef[0]),
        alpha_annualized=float((1 + coef[0]) ** periods_per_year - 1),
        alpha_tstat=float(t[0]),
        alpha_pvalue=p_alpha,
        betas={n: float(b) for n, b in zip(fac_names, coef[1:])},
        beta_tstats={n: float(v) for n, v in zip(fac_names, t[1:])},
        r_squared=r2,
        n_obs=len(df),
        nw_lags=lags,
        label=label,
        caveat=caveat,
    )


def benchmark_alpha(strategy: pd.Series, benchmark: pd.Series,
                    periods_per_year: float) -> AlphaResult:
    return regress_alpha(
        strategy, benchmark.rename("universe").to_frame(), periods_per_year,
        label="Alpha vs. equal-weight universe (out-of-sample periods)",
    )


# ------------------------------------------------------- Fama-French data

def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily-factors CSV (percent units) into a
    decimal-return DataFrame indexed by date. The files carry free-text
    headers/footers; only rows whose first field is an 8-digit date are
    data, and the header is the line immediately before the first one."""
    lines = text.splitlines()
    header: list[str] | None = None
    rows = []
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 1 and parts[0].isdigit() and len(parts[0]) == 8:
            if header is None:
                header = [p.strip() for p in lines[i - 1].split(",")][1:]
            rows.append(parts)
        elif header is not None and rows:
            break   # end of the daily block (annual tables may follow)
    if header is None or not rows:
        raise DataError("Could not find a daily data block in French factor file")
    df = pd.DataFrame([r[1:len(header) + 1] for r in rows], columns=header,
                      index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"))
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    if "RF" not in df.columns or "Mkt-RF" not in df.columns:
        raise DataError(f"Unexpected French factor columns: {list(df.columns)}")
    return df.dropna()


def fetch_french_factors(region: str, cache_dir: Path, max_age_hours: float = 24.0,
                         timeout: float = 30.0) -> pd.DataFrame:
    if region not in FRENCH_DATASETS:
        raise ValueError(f"region must be one of {sorted(FRENCH_DATASETS)}")
    name = FRENCH_DATASETS[region]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"french_{name}.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < max_age_hours:
        return pd.read_parquet(cache)
    url = f"{FRENCH_BASE_URL}{name}_CSV.zip"
    logger.info(f"Downloading Fama-French factors: {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            text = zf.read(zf.namelist()[0]).decode("latin-1")
    except Exception as e:
        raise DataError(f"Failed to download Fama-French factors: {e}") from e
    df = parse_french_csv(text)
    df.to_parquet(cache)
    return df


def compound_factors_over_periods(daily: pd.DataFrame, starts: pd.Series
                                  ) -> pd.DataFrame:
    """Compound daily factor returns over each holding period
    (start, end], where `starts` maps period-entry date -> exit date.
    Periods not fully covered by the factor data are dropped."""
    out = {}
    last_available = daily.index.max()
    for start, end in starts.items():
        if end > last_available:
            continue
        window = daily.loc[(daily.index > start) & (daily.index <= end)]
        if window.empty:
            continue
        out[start] = (1 + window).prod() - 1
    return pd.DataFrame(out).T


def fama_french_alpha(strategy: pd.Series, period_end_dates: pd.Series,
                      daily_factors: pd.DataFrame, periods_per_year: float,
                      base_currency: str, region: str) -> AlphaResult:
    per = compound_factors_over_periods(daily_factors, period_end_dates)
    if per.empty:
        raise ValueError("No strategy periods overlap the Fama-French data "
                         "(French data is published with a 1-2 month lag)")
    excess = strategy.reindex(per.index) - per["RF"]
    fac = per.drop(columns=["RF"])
    caveat = ""
    if base_currency != "USD":
        caveat = (f"factors are USD returns, strategy is in {base_currency}; "
                  f"FX moves add noise to the regression")
    return regress_alpha(excess, fac, periods_per_year,
                         label=f"Fama-French ({region}) alpha", caveat=caveat)
