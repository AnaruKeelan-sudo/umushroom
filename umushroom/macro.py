"""Macro view of the portfolio: what the economy is doing, and how exposed
the holdings are to it. Reporting only -- the model is NOT changed.

For each holding, from the last ~2 years of WEEKLY returns:

  market beta    = how much it moves with global stocks (ACWI)
  macro beta f   = coefficient on factor f in   r = a + b_m*market + b_f*f
                   (so it is the effect of f OVER AND ABOVE the market)

Factors (all free on Yahoo): US 10-year yield, UK gilts, Brent oil, the US
dollar, GBP/USD, gold and the VIX. "Typical weekly shock" = one standard
deviation of the factor's weekly change, which puts every factor on the same
scale ("if oil has a normal-sized week, we move x%").

Scenarios are linear, other-things-equal estimates -- a sketch of the risk,
not a forecast. A GBP move also includes the currency translation of
USD-priced holdings (what the GBP value of a USD asset does when GBP moves).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from umushroom.tracker import FetchFn, yahoo_closes


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    symbol: str
    kind: str          # "ret" = % change, "diff" = change in level (yields, VIX)
    unit: str          # how a change is shown


FACTORS: list[Factor] = [
    Factor("market", "Global stocks (ACWI)", "ACWI", "ret", "%"),
    Factor("us10y", "US 10-year yield", "^TNX", "diff", "pp"),
    Factor("gilts", "UK gilts (IGLT)", "IGLT.L", "ret", "%"),
    Factor("oil", "Brent oil", "BZ=F", "ret", "%"),
    Factor("usd", "US dollar index", "DX-Y.NYB", "ret", "%"),
    Factor("gbp", "GBP/USD", "GBPUSD=X", "ret", "%"),
    Factor("gold", "Gold", "GC=F", "ret", "%"),
    Factor("vix", "VIX (fear index)", "^VIX", "diff", "pts"),
]
FBY = {f.key: f for f in FACTORS}

# (label, factor, shock) -- shock in the factor's own units (0.10 = 10%)
SCENARIOS: list[tuple[str, str, float]] = [
    ("Global stocks fall 10%", "market", -0.10),
    ("US 10-year yield +0.5 pp", "us10y", 0.5),
    ("UK gilts fall 5% (UK yields jump)", "gilts", -0.05),
    ("Oil +20% (supply shock)", "oil", 0.20),
    ("US dollar +5%", "usd", 0.05),
    ("Pound +5% vs dollar", "gbp", 0.05),
    ("Gold +10% (flight to safety)", "gold", 0.10),
    ("VIX +10 points (panic)", "vix", 10.0),
]

LOOKBACK_DAYS = 760          # ~2 years of weekly data
MIN_WEEKS = 40


@dataclass
class HoldingExposure:
    ticker: str
    sleeve: str
    weight: float
    currency: str
    betas: dict[str, float] = field(default_factory=dict)   # empty = no data


@dataclass
class MacroResult:
    as_of: str
    dashboard: list[dict]                   # one row per factor
    holdings: list[HoldingExposure]
    portfolio: dict[str, float]             # value-weighted betas
    shock_sd: dict[str, float]              # weekly SD of each factor
    coverage: float                         # share of value with betas
    currency_mix: dict[str, float]
    scenarios: list[dict]

    def sensitivity(self, key: str) -> float | None:
        """Portfolio move for a typical (1 SD) weekly shock in factor `key`."""
        b, sd = self.portfolio.get(key), self.shock_sd.get(key)
        return None if b is None or sd is None else b * sd


def _weekly(closes: pd.DataFrame) -> pd.DataFrame:
    return closes.sort_index().ffill(limit=5).resample("W-FRI").last()


def factor_changes(weekly: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for f in FACTORS:
        if f.symbol not in weekly.columns:
            continue
        s = weekly[f.symbol]
        if f.key == "us10y" and s.dropna().median() > 20:     # old x10 quoting
            s = s / 10
        out[f.key] = s.pct_change(fill_method=None) if f.kind == "ret" else s.diff()
    return pd.DataFrame(out)


def betas_for(r: pd.Series, fx: pd.DataFrame) -> dict[str, float]:
    """Market beta, then each macro beta controlling for the market."""
    if "market" not in fx.columns:
        return {}
    out: dict[str, float] = {}
    base = pd.concat([r.rename("r"), fx["market"]], axis=1).dropna()
    if len(base) < MIN_WEEKS or base["market"].var() == 0:
        return {}
    out["market"] = float(np.cov(base["r"], base["market"])[0, 1] / base["market"].var())
    for key in fx.columns:
        if key == "market":
            continue
        d = pd.concat([r.rename("r"), fx["market"], fx[key]], axis=1).dropna()
        if len(d) < MIN_WEEKS or d[key].var() == 0:
            continue
        x = np.column_stack([np.ones(len(d)), d["market"], d[key]])
        coef, *_ = np.linalg.lstsq(x, d["r"].to_numpy(), rcond=None)
        out[key] = float(coef[2])
    return out


def _dashboard(closes: pd.DataFrame) -> list[dict]:
    rows = []
    for f in FACTORS:
        if f.symbol not in closes.columns:
            continue
        s = closes[f.symbol].dropna()
        if f.key == "us10y" and len(s) and s.median() > 20:
            s = s / 10
        if len(s) < 70:
            continue
        last = float(s.iloc[-1])

        def chg(n: int, s: pd.Series = s, last: float = last, kind: str = f.kind) -> float:
            prev = float(s.iloc[-1 - n])
            return last / prev - 1 if kind == "ret" else last - prev
        lvl = s.iloc[-252:] if len(s) >= 252 else s
        rows.append({"key": f.key, "label": f.label, "unit": f.unit, "kind": f.kind,
                     "level": last, "chg_1m": chg(21), "chg_3m": chg(63),
                     "pct_1y": float((lvl <= last).mean()),
                     "date": s.index[-1].strftime("%Y-%m-%d")})
    return rows


def _symbol(o: dict) -> str:
    return str(o.get("priced_from") or o["ticker"])


def analyse(payload: dict, fetch: FetchFn = yahoo_closes,
            today: date | None = None) -> MacroResult:
    today = today or date.today()
    orders = payload.get("orders") or []
    syms = sorted({_symbol(o) for o in orders} | {f.symbol for f in FACTORS})
    closes = fetch(syms, today - timedelta(days=LOOKBACK_DAYS))
    closes = closes.loc[:, ~closes.columns.duplicated()]
    weekly = _weekly(closes)
    fx = factor_changes(weekly)

    total = sum(float(o.get("allocated_value") or 0) for o in orders) or 1.0
    holdings: list[HoldingExposure] = []
    for o in orders:
        h = HoldingExposure(str(o["ticker"]), str(o.get("sleeve") or ""),
                            float(o.get("allocated_value") or 0) / total,
                            str(o.get("listing_currency") or ""))
        sym = _symbol(o)
        if sym in weekly.columns:
            r = weekly[sym].pct_change(fill_method=None)
            h.betas = betas_for(r, fx.iloc[-110:])
        holdings.append(h)

    covered = [h for h in holdings if h.betas]
    cov_w = sum(h.weight for h in covered)
    portfolio: dict[str, float] = {}
    for f in FACTORS:
        ws = [(h.weight, h.betas[f.key]) for h in covered if f.key in h.betas]
        w = sum(x for x, _ in ws)
        if w > 0:
            portfolio[f.key] = sum(x * b for x, b in ws) / w      # per unit covered
    sd = {k: float(fx[k].iloc[-110:].std()) for k in fx.columns if fx[k].notna().sum() > 10}

    mix: dict[str, float] = {}
    for h in holdings:
        mix[h.currency or "?"] = mix.get(h.currency or "?", 0.0) + h.weight
    usd_share = sum(h.weight for h in holdings if h.currency.upper() == "USD")

    scen = []
    for label, key, shock in SCENARIOS:
        b = portfolio.get(key)
        if b is None:
            continue
        impact = b * shock
        note = ""
        if key == "gbp":
            # GBP up => a USD asset is worth less in GBP: about -shock/(1+shock)
            fx_effect = usd_share * (-shock / (1 + shock))
            impact += fx_effect
            note = f"includes {fx_effect:+.1%} from USD holdings worth less in GBP"
        scen.append({"label": label, "key": key, "shock": shock, "impact": impact,
                     "note": note})

    return MacroResult(
        as_of=str(payload.get("meta", {}).get("as_of", "")), dashboard=_dashboard(closes),
        holdings=holdings, portfolio=portfolio, shock_sd=sd, coverage=cov_w,
        currency_mix=dict(sorted(mix.items(), key=lambda kv: -kv[1])), scenarios=scen)


def talking_points(res: MacroResult) -> list[str]:
    """Plain-English lines for the macro / risk slides, straight from the numbers."""
    pts = []
    b = res.portfolio.get("market")
    if b is not None:
        pts.append(f"Market beta {b:.2f}: if global stocks fall 10%, the portfolio "
                   f"would be expected to fall about {abs(b) * 10:.1f}%.")
    sens: dict[str, float] = {}
    for k in res.portfolio:
        v = res.sensitivity(k)
        if k != "market" and v is not None:
            sens[k] = v
    if sens:
        ranked = sorted(sens.items(), key=lambda kv: -abs(kv[1]))
        # the runner-up only if it is not trivially small next to the biggest
        top = [kv for i, kv in enumerate(ranked[:2])
               if i == 0 or abs(kv[1]) >= 0.5 * abs(ranked[0][1])]
        for i, (k, v) in enumerate(top):
            f = FBY[k]
            way = "gains" if v > 0 else "loses"
            which = "Biggest" if i == 0 else "Next biggest"
            pts.append(f"{which} macro exposure beyond the market: {f.label}. In a "
                       f"typical week when it rises, the portfolio {way} about "
                       f"{abs(v):.2%} on top of the market move.")
    usd = res.currency_mix.get("USD", 0.0)
    if usd:
        pts.append(f"{usd:.0%} of the money is in USD-priced assets, so a stronger "
                   "pound lowers their value in GBP (currency risk).")
    worst = min(res.scenarios, key=lambda s: s["impact"], default=None)
    if worst:
        pts.append(f"Worst of the stress scenarios: {worst['label'].lower()} "
                   f"({worst['impact']:+.1%}).")
    return pts
