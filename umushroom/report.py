"""Self-contained HTML report: today's buys, the evidence, and Monte Carlo.

`build_payload` gathers everything into plain JSON-safe data;
`render_html` turns it into ONE file with no external libraries or network
access: charts are inline SVG drawn here, styled with the validated
reference palette (light + dark), with a hover readout and a table view
under every chart so no number is only reachable by hovering.

Safety: every piece of text is HTML-escaped when drawn, and the embedded
data block escapes "</" so a hostile company name can't end the script tag.
The in-page script only ever sets text via textContent.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from umushroom.montecarlo import MonteCarloResult, bootstrap_annual_alpha, combine_sleeves
from umushroom.purged_backtest import PurgedWalkForwardResult
from umushroom.rebalance import DEFAULT_REBALANCE_DAYS, next_rebalance, trade_day
from umushroom.risk import MIN_OBS, backtest_var, var_es
from umushroom.window import WindowResult

W, H = 760, 280
M_L, M_R, M_T, M_B = 64, 104, 16, 32          # right margin holds end labels


# ------------------------------------------------------------------- data

@dataclass
class SleeveReport:
    name: str
    weight: float
    selection: str                      # configured rule: model | all | auto
    backtest: PurgedWalkForwardResult | None = None
    alpha_annual: float | None = None
    alpha_t: float | None = None
    validation: list[tuple[str, str, str]] = field(default_factory=list)
    # ---- v4.19 ----
    selection_used: str | None = None   # what this run actually did: model | all | core
    decision: str = ""                  # why (the pre-declared rule, spelled out)
    window: WindowResult | None = None  # contest-window evaluation
    feature_set: str = "equity"
    horizon_days: int | None = None
    n_instruments: int = 0              # usable after data validation
    n_core: int = 0
    universe_status: dict[str, int] = field(default_factory=dict)
    universe_missing: bool = False
    contest_window_days: int | None = None
    # today's model scores over the WHOLE sleeve universe (even when the
    # sleeve ends up holding the core basket), for the ranking tables
    scores: pd.Series | None = None
    # ---- v4.20 ----
    model_type: str = "ridge"               # ridge | mlp | gbm
    weighting: str = "equal"                # equal | hrp | factor
    max_pick_correlation: float | None = None   # near-duplicate guard, None = off
    factor_risk: dict | None = None         # factor_risk_payload(), when sized by it


def _clean(v: Any) -> Any:
    """JSON-safe scalar: NaN/inf -> None, numpy -> python."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def _records(df: pd.DataFrame | None, index_name: str = "ticker") -> list[dict]:
    if df is None or df.empty:
        return []
    out = []
    for idx, row in df.iterrows():
        rec = {index_name: str(idx)}
        rec.update({str(k): _clean(v) for k, v in row.items()})
        out.append(rec)
    return out


def _floats(values: Any) -> list[float | None]:
    return [_clean(v) for v in values]


def stock_snapshot(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Recent-performance facts for the buy list: 1/3/12-month return (21,
    63, 252 trading days) and annualised 60-day volatility, in base currency."""
    cols = [t for t in tickers if t in prices.columns]
    px = prices[cols]
    last = px.iloc[-1]

    def ret(days: int) -> pd.Series:
        if len(px) <= days:
            return pd.Series(np.nan, index=cols)
        return last / px.iloc[-(days + 1)] - 1
    vol = px.pct_change(fill_method=None).iloc[-60:].std() * np.sqrt(252)
    return pd.DataFrame({"ret_1m": ret(21), "ret_3m": ret(63), "ret_12m": ret(252),
                         "vol_60d": vol}).reindex(tickers)


def _hist(values: np.ndarray, bins: int = 40, trim: bool = True) -> dict:
    """Histogram over the central 99% of outcomes. The extreme 0.5% on each
    side is NOT squashed into the edge bars (that drew fake spikes at both
    ends); it is left off the axis and its share reported instead. Shares
    are of ALL values, so the bars honestly sum to less than 100%."""
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"edges": [0.0, 1.0], "share": [0.0], "hidden_low": 0.0, "hidden_high": 0.0}
    lo, hi = (float(x) for x in (np.percentile(v, [0.5, 99.5]) if trim
                                 else (v.min(), v.max())))
    if hi <= lo:
        lo, hi = lo - 0.5 * (abs(lo) or 1.0) * 1e-3, hi + 0.5 * (abs(hi) or 1.0) * 1e-3
    inside = v[(v >= lo) & (v <= hi)]
    counts, edges = np.histogram(inside, bins=bins, range=(lo, hi))
    n = len(v)
    return {"edges": _floats(edges), "share": _floats(counts / n),
            "hidden_low": float((v < lo).sum() / n), "hidden_high": float((v > hi).sum() / n)}


def _uses_model(s: SleeveReport) -> bool:
    """Did this run trade the model's picks? (v4.19: `auto` decides per run.)"""
    if s.selection_used is not None:
        return s.selection_used == "model"
    return s.selection == "model"


def _window_payload(w: WindowResult | None) -> dict | None:
    if w is None or w.n_windows == 0:
        return None
    return {
        "periods": w.window_periods, "days": w.window_days,
        "n_windows": w.n_windows, "n_independent": w.n_independent,
        "win_rate": _clean(w.win_rate), "positive_rate": _clean(w.positive_rate),
        "mean_active": _clean(w.mean_active),
        "median_strategy": _clean(w.median_strategy),
        "median_benchmark": _clean(w.median_benchmark),
        "worst": _clean(w.worst_strategy), "best": _clean(w.best_strategy),
        "active_t": _clean(w.active_tstat), "verdict": w.verdict,
        "dates": [d.strftime("%Y-%m-%d") for d in w.rolling_strategy.index],
        "strategy": _floats(w.rolling_strategy),
        "benchmark": _floats(w.rolling_benchmark),
        "active_hist": _hist(w.rolling_active.to_numpy(dtype=float), bins=30),
    }


def factor_risk_payload(fr: Any) -> dict:
    """JSON-ready view of a factor_risk.FactorRisk (v4.20)."""
    factors = [str(f) for f in fr.budgets.index]
    return {
        "factors": factors,
        "contributions": {f: _clean(fr.contributions.get(f)) for f in factors},
        "budgets": {f: _clean(fr.budgets[f]) for f in factors},
        "idiosyncratic_share": _clean(fr.idiosyncratic_share),
        "weights": {str(t): _clean(w) for t, w in fr.weights.items()},
        "exposures": {str(t): {f: _clean(fr.exposures.loc[t, f]) for f in factors}
                      for t in fr.weights.index},
    }


def _sleeve_payload(s: SleeveReport, periods_per_year: float) -> dict:
    out: dict[str, Any] = {"name": s.name, "weight": s.weight, "selection": s.selection,
                           "alpha": _clean(s.alpha_annual), "alpha_t": _clean(s.alpha_t),
                           "validation": [{"name": n, "status": st, "detail": d}
                                          for n, st, d in s.validation],
                           "selection_used": s.selection_used or (
                               "model" if s.selection == "model" else "all"),
                           "decision": s.decision, "feature_set": s.feature_set,
                           "horizon_days": s.horizon_days,
                           "n_instruments": s.n_instruments, "n_core": s.n_core,
                           "universe_status": dict(s.universe_status),
                           "universe_missing": s.universe_missing,
                           "contest_window_days": s.contest_window_days,
                           "window": _window_payload(s.window),
                           "model_type": s.model_type, "weighting": s.weighting,
                           "max_pick_correlation": _clean(s.max_pick_correlation),
                           "factor_risk": s.factor_risk}
    r = s.backtest
    if r is None or len(r.period_returns) == 0:
        return out
    # a missing week counts as flat (0%) rather than breaking the whole line
    model_curve = (1 + r.period_returns.fillna(0.0)).cumprod()
    bench = (1 + r.benchmark_period_returns.reindex(r.period_returns.index)
             .fillna(0.0)).cumprod()
    active = r.active_returns.reindex(r.period_returns.index).fillna(0.0)
    # a hold-all sleeve EARNS the equal-weight basket; the model's picks are
    # only shown for reference, clearly labelled as not used
    strat = model_curve if _uses_model(s) else bench
    out.update({
        "ic": _clean(r.mean_information_coefficient), "ic_t": _clean(r.ic_tstat),
        "ic_hit": _clean(r.ic_hit_rate), "years": _clean(r.years),
        "sharpe": _clean(r.overall_sharpe),
        "curves": {
            "dates": [d.strftime("%Y-%m-%d") for d in r.period_returns.index],
            "strategy": _floats(strat), "benchmark": _floats(bench),
            "model": _floats(model_curve),
            "cum_alpha": _floats(strat / bench - 1),
            "rolling_alpha": _floats(active.rolling(52, min_periods=26).sum()),
        },
    })
    try:
        draws = bootstrap_annual_alpha(r.active_returns, periods_per_year)
        out["alpha_boot"] = {"prob_positive": float((draws > 0).mean()),
                             "p5": float(np.percentile(draws, 5)),
                             "p50": float(np.median(draws)),
                             "p95": float(np.percentile(draws, 95)),
                             "hist": _hist(draws)}
    except ValueError:
        pass
    return out


def _held_returns(s: SleeveReport) -> pd.Series | None:
    """Weekly returns of what the sleeve actually HOLDS (model picks, or the
    equal-weight basket for hold-all sleeves)."""
    r = s.backtest
    if r is None or len(r.period_returns) == 0:
        return None
    held = (r.period_returns if _uses_model(s)
            else r.benchmark_period_returns.reindex(r.period_returns.index))
    return held.dropna()


def _risk_payload(sleeves: list[SleeveReport], mc: MonteCarloResult | None,
                  value: float | None) -> dict | None:
    out: dict[str, Any] = {"value": _clean(value), "weekly": None, "horizon": None,
                           "sleeves": [], "backtest": None}
    parts = []
    for s in sleeves:
        held = _held_returns(s)
        if held is None or len(held) < MIN_OBS:
            continue
        parts.append((s.weight, held))
        v95, e95 = var_es(held, 0.95)
        v99, e99 = var_es(held, 0.99)
        out["sleeves"].append({"name": s.name, "n": len(held), "var95": v95, "es95": e95,
                               "var99": v99, "es99": e99})
    if parts:
        total = sum(w for w, _ in parts) or 1.0
        port = combine_sleeves([(w / total, r) for w, r in parts])
        if len(port) >= MIN_OBS:
            v95, e95 = var_es(port, 0.95)
            v99, e99 = var_es(port, 0.99)
            out["weekly"] = {"n": len(port), "var95": v95, "es95": e95, "var99": v99,
                             "es99": e99, "worst": float(-port.min()),
                             "hist": _hist(port.to_numpy(), bins=30, trim=False)}
            try:
                bt = backtest_var(port, 0.95, 52)
                out["backtest"] = {
                    "level": bt.level, "window": bt.window, "n": bt.n,
                    "breaches": bt.breaches, "breach_rate": bt.breach_rate,
                    "expected_rate": bt.expected_rate, "lr": bt.lr, "p_value": bt.p_value,
                    "status": bt.status,
                    "dates": [d.strftime("%Y-%m-%d") for d in bt.returns.index],
                    "returns": _floats(bt.returns), "limit": _floats(-bt.var)}
            except ValueError:
                pass
    if mc is not None:
        st, s0 = mc.stats(), mc.start_value
        out["horizon"] = {"weeks": mc.horizon_weeks, "var95": st["var_95"] / s0,
                          "es95": st["es_95"] / s0, "var99": st["var_99"] / s0,
                          "es99": st["es_99"] / s0}
    if out["weekly"] is None and out["horizon"] is None:
        return None
    return out


def build_payload(meta: dict, sleeves: list[SleeveReport], orders: pd.DataFrame,
                  top_ranked: pd.DataFrame | None = None,
                  deciles: pd.Series | None = None,
                  mc: MonteCarloResult | None = None, mc_note: str = "",
                  periods_per_year: float = 252 / 5,
                  top_etfs: dict[str, pd.DataFrame] | None = None) -> dict:
    payload: dict[str, Any] = {
        "meta": {k: _clean(v) for k, v in meta.items()},
        "orders": _records(orders),
        "top": _records(top_ranked),
        "sleeves": [_sleeve_payload(s, periods_per_year) for s in sleeves],
        "deciles": _floats(deciles.tolist()) if deciles is not None else [],
        "mc": None,
        "mc_note": mc_note,
        # v4.19: per-sleeve ETF ranking tables, keyed by sleeve name
        "top_etfs": {k: _records(v) for k, v in (top_etfs or {}).items()},
    }
    if mc is not None:
        bands = mc.percentile_bands()
        payload["mc"] = {
            "horizon_weeks": mc.horizon_weeks, "n_paths": mc.n_paths,
            "weeks": list(range(mc.horizon_weeks + 1)),
            "start_value": mc.start_value, "n_history": mc.n_history,
            "block_size": mc.block_size,
            "bands": {c: _floats(bands[c]) for c in bands.columns},
            "stats": {k: _clean(v) for k, v in mc.stats().items()},
            "final_hist": _hist(mc.final_strategy),
        }
    value = mc.start_value if mc is not None else (
        float(orders["allocated_value"].sum())
        if "allocated_value" in orders.columns and len(orders) else None)
    payload["risk"] = _risk_payload(sleeves, mc, value)
    return payload


# --------------------------------------------------------------- svg helpers

def _esc(x: object) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0, 1.0]
    if hi == lo:
        hi = lo + (abs(lo) or 1.0)
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    t = math.floor(lo / step) * step
    ticks = []
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def _fmt(v: float | None, kind: str) -> str:
    if v is None or not math.isfinite(v):
        return "n/a"
    if kind == "pct":
        return f"{v:+.1%}"
    if kind == "pct2":
        return f"{v:+.2%}"
    if kind == "money":
        return f"{v:,.0f}"
    if kind == "x":
        return f"{v:.2f}x"
    return f"{v:,.2f}"


def _tick(v: float, kind: str, step: float | None = None) -> str:
    if abs(v) < 1e-12:
        v = 0.0                       # never print "-0%"
    if kind in ("pct", "pct2"):
        # as many decimals as the tick STEP needs: 5% / 0.5% / 0.25%
        st = f"{abs(step or 0.01) * 100:.6f}".rstrip("0").rstrip(".")
        dp = min(3, len(st.split(".")[1])) if "." in st else 0
        return f"{v:.{dp}%}"
    if kind == "money":
        mstep = abs(step) if step else 1000.0
        if abs(v) < 1000 or mstep < 100:
            # small steps (8,700 / 8,725 / 8,750): show the whole number
            dp = 0 if mstep >= 1 else 2
            return f"{v:,.{dp}f}"
        # 90k / 95k, or 8.5k / 9k when the step is under 1,000
        dp = 0 if mstep >= 1000 else 1
        return f"{v / 1000:,.{dp}f}k"
    if kind == "x":
        return f"{v:.1f}x"
    return f"{v:g}"


class _Scale:
    def __init__(self, lo: float, hi: float, a: float, b: float):
        self.lo, self.hi, self.a, self.b = lo, hi, a, b

    def __call__(self, v: float) -> float:
        if self.hi == self.lo:
            return (self.a + self.b) / 2
        return self.a + (v - self.lo) / (self.hi - self.lo) * (self.b - self.a)


def _same(a: list[float | None], b: list[float | None]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return False
        if x is not None and y is not None and abs(x - y) > 1e-12 * max(1.0, abs(x)):
            return False
    return True


def _has_data(vs: list[float | None]) -> bool:
    return any(v is not None for v in vs)


def _y_axis(ticks: list[float], ys: _Scale, kind: str) -> str:
    step = ticks[1] - ticks[0] if len(ticks) > 1 else None
    return "".join(
        f'<line class="grid" x1="{M_L}" x2="{W - M_R}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>'
        f'<text class="tick" x="{M_L - 8}" y="{ys(t) + 4:.1f}" text-anchor="end">'
        f'{_esc(_tick(t, kind, step))}</text>' for t in ticks)


def _x_labels(labels: list[str], xs: _Scale, max_labels: int = 8) -> str:
    step = max(1, math.ceil(len(labels) / max_labels))
    return "".join(f'<text class="tick" x="{xs(i):.1f}" y="{H - 10}" '
                   f'text-anchor="middle">{_esc(labels[i])}</text>'
                   for i in range(0, len(labels), step))


def _table(headers: list[str], rows: list[list[str]], summary: str = "Table view") -> str:
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
                  for r in rows)
    return (f'<details class="tv"><summary>{_esc(summary)}</summary><div class="scroll">'
            f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'
            f'</details>')


def _card(title: str, sub: str, inner: str, legend: str = "", chart: dict | None = None,
          table: str = "") -> str:
    data = f' data-chart="{_esc(json.dumps(chart))}"' if chart else ""
    return (f'<figure class="card"><figcaption><h3>{_esc(title)}</h3><p>{_esc(sub)}</p>'
            f'</figcaption>{legend}<div class="plot"{data}><svg viewBox="0 0 {W} {H}" '
            f'role="img" aria-label="{_esc(title)}">{inner}</svg>'
            f'<div class="tip" hidden></div></div>{table}</figure>')


def _legend(items: list[tuple[str, str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i class="key-{shape}" style="background:var({c})"></i>{_esc(n)}</span>'
        for n, c, shape in items) + "</div>"


def line_chart(title: str, sub: str, labels: list[str],
               series: list[tuple[str, list[float | None], str]], kind: str,
               zero_line: bool = False) -> str:
    # a series identical to an earlier one would be drawn exactly underneath
    # it (looks like a missing line, with two end labels on top of each
    # other): merge it into the earlier series' name instead
    merged: list[tuple[str, list[float | None], str]] = []
    for name, vs, color in series:
        dup = next((i for i, (_, ws, _) in enumerate(merged) if _same(ws, vs)), None)
        if dup is None:
            merged.append((name, vs, color))
        else:
            n0, ws, c0 = merged[dup]
            merged[dup] = (f"{n0} = {name}", ws, c0)
    series = merged
    vals = [v for _, vs, _ in series for v in vs if v is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    if zero_line:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    ticks = nice_ticks(lo, hi)
    ys = _Scale(ticks[0], ticks[-1], H - M_B, M_T)
    xs = _Scale(0, max(len(labels) - 1, 1), M_L, W - M_R)
    parts = [_y_axis(ticks, ys, kind), _x_labels(labels, xs)]
    if zero_line and ticks[0] < 0 < ticks[-1]:
        parts.append(f'<line class="zero" x1="{M_L}" x2="{W - M_R}" y1="{ys(0):.1f}" '
                     f'y2="{ys(0):.1f}"/>')
    label_ys: list[float] = []
    # draw the primary (first) series LAST so it sits on top where lines cross
    for _name, vs, color in reversed(series):
        pts = [(xs(i), ys(v)) for i, v in enumerate(vs) if v is not None]
        if not pts:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<path class="ln" d="{d}" style="stroke:var({color})"/>')
        parts.append(f'<circle class="dot" cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" '
                     f'r="4" style="fill:var({color})"/>')
    for _name, vs, _color in series:          # end labels: primary first
        pts = [(xs(i), ys(v)) for i, v in enumerate(vs) if v is not None]
        if not pts:
            continue
        y = pts[-1][1]
        if any(abs(y - other) < 13 for other in label_ys):
            continue          # would collide: the legend + tooltip carry it
        label_ys.append(y)
        last = next(v for v in reversed(vs) if v is not None)
        parts.append(f'<text class="endlab" x="{pts[-1][0] + 8:.1f}" y="{y + 4:.1f}">'
                     f'{_esc(_fmt(last, kind))}</text>')
    parts.append(f'<line class="xhair" y1="{M_T}" y2="{H - M_B}"/>')
    chart = {"labels": labels, "kind": kind, "x0": M_L, "x1": W - M_R,
             "series": [{"name": n, "color": c, "values": vs} for n, vs, c in series]}
    legend = _legend([(n, c, "line") for n, _, c in series]) if len(series) > 1 else ""
    rows = [[labels[i]] + [_fmt(vs[i], kind) for _, vs, _ in series]
            for i in range(len(labels))]
    return _card(title, sub, "".join(parts), legend, chart,
                 _table(["Date"] + [n for n, _, _ in series], rows))


def bar_chart(title: str, sub: str, labels: list[str], values: list[float],
              kind: str, color: str = "--series-1") -> str:
    lo, hi = min(min(values), 0.0), max(max(values), 0.0)
    ticks = nice_ticks(lo, hi)
    ys = _Scale(ticks[0], ticks[-1], H - M_B, M_T)
    band = (W - M_R - M_L) / max(len(values), 1)
    bw = min(24.0, band * 0.6)
    y0 = ys(0)
    parts = [_y_axis(ticks, ys, kind)]
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = M_L + band * (i + 0.5)
        x = cx - bw / 2
        top, bot = (ys(v), y0) if v >= 0 else (y0, ys(v))
        r = min(4.0, max(bot - top, 0.0))
        if v >= 0:      # rounded data end, square at the baseline
            d = (f"M{x:.1f},{bot:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},"
                 f"{top:.1f} H{x + bw - r:.1f} Q{x + bw:.1f},{top:.1f} {x + bw:.1f},"
                 f"{top + r:.1f} V{bot:.1f} Z")
        else:
            d = (f"M{x:.1f},{top:.1f} V{bot - r:.1f} Q{x:.1f},{bot:.1f} {x + r:.1f},"
                 f"{bot:.1f} H{x + bw - r:.1f} Q{x + bw:.1f},{bot:.1f} {x + bw:.1f},"
                 f"{bot - r:.1f} V{top:.1f} Z")
        parts.append(f'<g class="bar" tabindex="0" data-tip="{_esc(lab + ": " + _fmt(v, kind))}">'
                     f'<rect class="hit" x="{cx - band / 2:.1f}" y="{M_T}" width="{band:.1f}" '
                     f'height="{H - M_B - M_T}"/><path d="{d}" style="fill:var({color})"/></g>'
                     f'<text class="tick" x="{cx:.1f}" y="{H - 10}" text-anchor="middle">'
                     f'{_esc(lab)}</text>')
    if ticks[0] < 0 < ticks[-1]:
        parts.append(f'<line class="zero" x1="{M_L}" x2="{W - M_R}" y1="{y0:.1f}" '
                     f'y2="{y0:.1f}"/>')
    return _card(title, sub, "".join(parts),
                 table=_table(["Group", "Value"], [[lab, _fmt(v, kind)]
                                                   for lab, v in zip(labels, values)]))


def histogram(title: str, sub: str, hist: dict, kind: str, marker: float,
              marker_label: str, summary_rows: list[list[str]]) -> str:
    edges, share = hist["edges"], hist["share"]
    hidden = hist.get("hidden_low", 0.0) + hist.get("hidden_high", 0.0)
    if hidden > 0:
        sub = (f"{sub} The most extreme {hidden:.1%} of runs lie beyond the axis "
               f"and aren't drawn (they are included in the table).")
    ticks = nice_ticks(0, max(max(share), 1e-9))
    ys = _Scale(0, ticks[-1], H - M_B, M_T)
    xs = _Scale(edges[0], edges[-1], M_L, W - M_R)
    parts = [_y_axis(ticks, ys, "pct")]
    for i, s in enumerate(share):
        top, bot = ys(s), ys(0)
        if bot - top < 0.5:
            continue
        x0, x1 = xs(edges[i]) + 1, xs(edges[i + 1]) - 1      # 2px surface gap
        tip = f"{_fmt(edges[i], kind)} to {_fmt(edges[i + 1], kind)}: {s:.1%} of runs"
        parts.append(f'<g class="bar" tabindex="0" data-tip="{_esc(tip)}">'
                     f'<rect class="hit" x="{xs(edges[i]):.1f}" y="{M_T}" '
                     f'width="{xs(edges[i + 1]) - xs(edges[i]):.1f}" height="{H - M_B - M_T}"/>'
                     f'<rect x="{x0:.1f}" y="{top:.1f}" width="{max(x1 - x0, 0.5):.1f}" '
                     f'height="{bot - top:.1f}" style="fill:var(--series-1)"/></g>')
    xt = nice_ticks(edges[0], edges[-1], 6)
    xstep = xt[1] - xt[0] if len(xt) > 1 else None
    for t in xt:
        if edges[0] <= t <= edges[-1]:
            parts.append(f'<text class="tick" x="{xs(t):.1f}" y="{H - 10}" '
                         f'text-anchor="middle">{_esc(_tick(t, kind, xstep))}</text>')
    if edges[0] <= marker <= edges[-1]:
        mx = xs(marker)
        parts.append(f'<line class="marker" x1="{mx:.1f}" x2="{mx:.1f}" y1="{M_T}" '
                     f'y2="{H - M_B}"/><text class="endlab" x="{mx + 6:.1f}" '
                     f'y="{M_T + 12}">{_esc(marker_label)}</text>')
    return _card(title, sub, "".join(parts),
                 table=_table(["Percentile", "Value"], summary_rows))


def fan_chart(mc: dict, currency: str) -> str:
    b = mc["bands"]
    n = len(b["p50"])
    labels = [f"wk {i}" for i in range(n)]
    lo = min(min(b["p5"]), min(b["benchmark_p50"]))
    hi = max(max(b["p95"]), max(b["benchmark_p50"]))
    ticks = nice_ticks(lo, hi)
    ys = _Scale(ticks[0], ticks[-1], H - M_B, M_T)
    xs = _Scale(0, n - 1, M_L, W - M_R)
    parts = [_y_axis(ticks, ys, "money"), _x_labels(labels, xs, 10)]
    for a, c, cls in (("p5", "p95", "band-outer"), ("p25", "p75", "band-inner")):
        up = " L".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(b[c]))
        dn = " L".join(f"{xs(i):.1f},{ys(b[a][i]):.1f}" for i in range(n - 1, -1, -1))
        parts.append(f'<path class="{cls}" d="M{up} L{dn} Z"/>')
    for key, color in (("p50", "--series-1"), ("benchmark_p50", "--series-2")):
        d = "M" + " L".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(b[key]))
        parts.append(f'<path class="ln" d="{d}" style="stroke:var({color})"/>'
                     f'<circle class="dot" cx="{xs(n - 1):.1f}" cy="{ys(b[key][-1]):.1f}" '
                     f'r="4" style="fill:var({color})"/>')
    for key, lab in (("p95", "95th"), ("p50", "median"), ("p5", "5th")):
        parts.append(f'<text class="endlab" x="{xs(n - 1) + 8:.1f}" y="{ys(b[key][-1]) + 4:.1f}">'
                     f'{lab} {_esc(_fmt(b[key][-1], "money"))}</text>')
    parts.append(f'<line class="xhair" y1="{M_T}" y2="{H - M_B}"/>')
    chart = {"labels": labels, "kind": "money", "x0": M_L, "x1": W - M_R,
             "series": [{"name": "95th percentile", "color": "--band-key", "values": b["p95"]},
                        {"name": "Our portfolio, median", "color": "--series-1",
                         "values": b["p50"]},
                        {"name": "Benchmark, median", "color": "--series-2",
                         "values": b["benchmark_p50"]},
                        {"name": "5th percentile", "color": "--band-key", "values": b["p5"]}]}
    legend = ('<div class="legend"><span><i class="key-line" style="background:'
              'var(--series-1)"></i>Our portfolio (median)</span><span><i class="key-line" '
              'style="background:var(--series-2)"></i>Benchmark (median)</span><span>'
              '<i class="key-box band-outer-key"></i>90% of outcomes</span><span>'
              '<i class="key-box band-inner-key"></i>middle 50%</span></div>')
    cols = ["p5", "p25", "p50", "p75", "p95", "benchmark_p50"]
    rows = [[labels[i]] + [_fmt(b[c][i], "money") for c in cols] for i in range(n)]
    sub = (f"{mc['n_paths']:,} simulated contests, each built from random blocks of "
           f"{mc['block_size']} consecutive real out-of-sample weeks "
           f"({mc['n_history']} weeks of history). Value in {currency}.")
    return _card(f"Monte Carlo: the next {mc['horizon_weeks']} weeks", sub, "".join(parts),
                 legend, chart,
                 _table(["Week", "5th", "25th", "median", "75th", "95th",
                         "Benchmark median"], rows))


# ------------------------------------------------------------------ page

def _tile(label: str, value: str, note: str = "") -> str:
    return (f'<div class="tile"><div class="t-l">{_esc(label)}</div>'
            f'<div class="t-v">{_esc(value)}</div><div class="t-n">{_esc(note)}</div></div>')


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{v:+.1%}"


def _num(v: Any, dp: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.{dp}f}"


def _price(v: float) -> str:
    """12,345.67 / 98.35 / 0.3352 / 0.08707 -- never scientific notation."""
    if abs(v) >= 1:
        return f"{v:,.2f}"
    return f"{v:.4g}" if abs(v) >= 1e-4 else f"{v:.8f}".rstrip("0")


def _orders_table(orders: list[dict], ccy: str) -> str:
    head = ["Sleeve", "Ticker", "Name", "Sector", "Shares", "Price (local)", "Value " + ccy,
            "Portfolio %", "1M", "3M", "12M", "Volatility", "Priced from"]
    rows = []
    for o in orders:
        lp = o.get("local_price")
        price = ("n/a" if lp is None else
                 f"{_price(lp)} {o.get('listing_currency') or ''}".strip())
        shares = o.get("shares")
        rows.append([o.get("sleeve", ""), o["ticker"], o.get("name") or "",
                     o.get("sector") or "",
                     "" if shares is None else f"{shares:g}", price,
                     _num(o.get("allocated_value"), 0),
                     "n/a" if o.get("portfolio_weight") is None
                     else f"{o['portfolio_weight']:.1%}",
                     _pct(o.get("ret_1m")), _pct(o.get("ret_3m")), _pct(o.get("ret_12m")),
                     "n/a" if o.get("vol_60d") is None else f"{o['vol_60d']:.0%}",
                     o.get("priced_from") or ""])
    return _plain_table(head, rows)


def _top_table(top: list[dict]) -> str:
    head = ["Rank", "Ticker", "Name", "Sector", "Exchange", "Better than", "In portfolio",
            "1M", "3M", "12M", "Volatility"]
    rows = [[str(t.get("rank", "")), t["ticker"], t.get("name") or "",
             t.get("sector") or "", t.get("exchange") or "",
             "n/a" if t.get("rank_pct") is None else f"{t['rank_pct']:.0%} of universe",
             "yes" if t.get("held") else "", _pct(t.get("ret_1m")), _pct(t.get("ret_3m")),
             _pct(t.get("ret_12m")),
             "n/a" if t.get("vol_60d") is None else f"{t['vol_60d']:.0%}"] for t in top]
    return _plain_table(head, rows)


def _plain_table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{_esc(h)}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
                  for r in rows)
    return (f'<div class="scroll"><table class="data"><thead><tr>{th}</tr></thead>'
            f'<tbody>{trs}</tbody></table></div>')


_ACRONYMS = {"etf": "ETF", "etfs": "ETFs"}


def _label(name: str) -> str:
    """Display name for a sleeve: 'etf' -> 'ETF', 'equity' -> 'Equity'."""
    return _ACRONYMS.get(name.lower(), name.capitalize())


def _sleeve_checks(s: dict) -> str:
    if not s.get("validation"):
        return ""
    nm = _label(s["name"])
    items = "".join(f'<li><span class="badge {_esc(v["status"].lower())}">'
                    f'{_esc(v["status"])}</span> <b>{_esc(v["name"])}</b>: '
                    f'{_esc(v["detail"])}</li>' for v in s["validation"])
    return (f'<div class="card"><h3>{_esc(nm)}: Validation checks</h3>'
            f'<ul class="checks">{items}</ul></div>')


def _how_held(s: dict) -> str:
    used = s.get("selection_used") or ("model" if s["selection"] == "model" else "all")
    if used == "model":
        return "model picks"
    if used == "core":
        return f"holds the {s.get('n_core') or ''} core ETFs, equal weight".replace("  ", " ")
    return "holds everything, equal weight"


def _model_used(s: dict) -> bool:
    used = s.get("selection_used") or ("model" if s["selection"] == "model" else "all")
    return used == "model"


def _sleeve_section(s: dict) -> str:
    out = []
    nm = _label(s["name"])
    how = _how_held(s)
    stats = []
    if _model_used(s):
        if s.get("alpha") is not None:
            stats.append(f"alpha {s['alpha']:+.1%}/yr (t = {s['alpha_t']:+.2f})")
        if s.get("ic") is not None:
            stats.append(f"IC {s['ic']:+.3f} (t = {s['ic_t']:+.2f})")
    else:
        stats.append("model reported for reference only, not used")
    out.append(f'<h3 class="sleeve">{_esc(nm)} sleeve <span>{s["weight"]:.0%} of capital, '
               f'{_esc(how)}{"; " + _esc(", ".join(stats)) if stats else ""}</span></h3>')
    c = s.get("curves")
    if c:
        if _model_used(s):
            out.append(line_chart(
                f"{nm}: out-of-sample performance",
                "Growth of 1 when every week was traded by a model trained only on "
                "earlier data (purged walk-forward).", c["dates"],
                [("Our picks", c["strategy"], "--series-1"),
                 ("Equal-weight universe", c["benchmark"], "--series-2")], "x"))
        else:
            out.append(line_chart(
                f"{nm}: out-of-sample performance",
                "What we hold (every instrument, equal weight) against what the model "
                "would have picked. The model failed validation here, so it is not used.",
                c["dates"],
                [("What we hold: all, equal weight", c["strategy"], "--series-1"),
                 ("Model's picks (not used)", c.get("model", c["strategy"]),
                  "--series-2")], "x"))
        if _model_used(s):
            out.append(line_chart(
                f"{nm}: cumulative alpha",
                "How far the picks have pulled ahead of (or behind) simply holding the "
                "whole universe.", c["dates"],
                [("Cumulative alpha", c["cum_alpha"], "--series-1")], "pct", zero_line=True))
            if not _has_data(c["rolling_alpha"]):
                out.append(f'<p class="note">{_esc(nm)}: rolling 1-year alpha needs at '
                           f'least 26 weeks of out-of-sample history.</p>')
            else:
                out.append(line_chart(
                    f"{nm}: rolling 1-year alpha",
                    "Active return summed over the trailing 52 weeks: is the edge steady, "
                    "or does it come and go?", c["dates"],
                    [("Rolling 1-year alpha", c["rolling_alpha"], "--series-1")], "pct",
                    zero_line=True))
    boot = s.get("alpha_boot")
    if boot and _model_used(s):
        out.append(histogram(
            f"{nm}: how sure are we about the alpha?",
            f"5,000 block-bootstrap resamples of the out-of-sample weeks. "
            f"{boot['prob_positive']:.0%} of them show positive alpha.",
            boot["hist"], "pct", 0.0, "no alpha",
            [["5th", _fmt(boot["p5"], "pct")], ["median", _fmt(boot["p50"], "pct")],
             ["95th", _fmt(boot["p95"], "pct")]]))
    return "".join(out)


SECTIONS: list[tuple[str, str]] = [
    ("overview", "Overview"), ("stocks", "Stocks"), ("etfs", "ETFs"), ("graphs", "Graphs"),
    ("montecarlo", "Monte Carlo"), ("risk", "Risk (VaR)"), ("checks", "Checks")]


def _etf_orders_table(orders: list[dict], ccy: str) -> str:
    head = ["Ticker", "UMushroom", "Name", "Asset class", "Shares", "Price (local)",
            "Value " + ccy, "Portfolio %", "1M", "3M", "12M", "Volatility"]
    rows = []
    for o in orders:
        lp = o.get("local_price")
        price = ("n/a" if lp is None else
                 f"{_price(lp)} {o.get('listing_currency') or ''}".strip())
        shares = o.get("shares")
        um = o.get("umushroom_symbol") or ""
        if o.get("umushroom_currency"):
            um = f"{um} ({o['umushroom_currency']})".strip()
        rows.append([o["ticker"], um, o.get("name") or "", o.get("sector") or "",
                     "" if shares is None else f"{shares:g}", price,
                     _num(o.get("allocated_value"), 0),
                     "n/a" if o.get("portfolio_weight") is None
                     else f"{o['portfolio_weight']:.1%}",
                     _pct(o.get("ret_1m")), _pct(o.get("ret_3m")), _pct(o.get("ret_12m")),
                     "n/a" if o.get("vol_60d") is None else f"{o['vol_60d']:.0%}"])
    return _plain_table(head, rows)


def _etf_top_table(top: list[dict]) -> str:
    head = ["Rank", "Ticker", "UMushroom", "Name", "Asset class", "Currency",
            "Better than", "In portfolio", "1M", "3M", "12M", "Volatility"]
    rows = [[str(t.get("rank", "")), t["ticker"], t.get("umushroom_symbol") or "",
             t.get("name") or "", t.get("asset_class") or "", t.get("currency") or "",
             "n/a" if t.get("rank_pct") is None else f"{t['rank_pct']:.0%} of universe",
             "yes" if t.get("held") else "", _pct(t.get("ret_1m")), _pct(t.get("ret_3m")),
             _pct(t.get("ret_12m")),
             "n/a" if t.get("vol_60d") is None else f"{t['vol_60d']:.0%}"] for t in top]
    return _plain_table(head, rows)


def _window_cards(s: dict) -> str:
    """Contest-window evidence for one sleeve: tiles, the per-window chart
    and the spread of active returns."""
    w = s.get("window")
    nm = _label(s["name"])
    days = s.get("contest_window_days") or (w or {}).get("days") or ""
    if not w:
        return (f'<p class="note">{_esc(nm)}: no contest-window evaluation in this run '
                f"(it needs the purged walk-forward backtest).</p>")
    verdict = str(w.get("verdict", ""))
    badge_cls = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}.get(verdict, "")
    tiles = [
        _tile(f"Beat the benchmark over {days} days",
              f"{w['win_rate']:.0%}" if w.get("win_rate") is not None else "n/a",
              f"of {w['n_windows']} rolling windows"),
        _tile("Median window return",
              _fmt(w.get("median_strategy"), "pct"),
              f"benchmark {_fmt(w.get('median_benchmark'), 'pct')}"),
        _tile("Active return per window", _fmt(w.get("mean_active"), "pct"),
              f"t = {w['active_t']:+.2f} on {w['n_independent']} non-overlapping windows"
              if w.get("active_t") is not None else "t = n/a"),
        _tile("Worst window", _fmt(w.get("worst"), "pct"),
              f"best {_fmt(w.get('best'), 'pct')}"),
    ]
    out = [(f'<h3 class="sec">{_esc(nm)}: does the edge arrive inside the contest? '
            f'<span class="badge {badge_cls}">{_esc(verdict)}</span></h3>'),
           f'<section class="tiles">{"".join(tiles)}</section>']
    out.append(line_chart(
        f"{nm}: every {days}-day window, out of sample",
        f"Compounded return of the model's picks against the equal-weight universe over "
        f"each stretch the length of Stage 1 ({w['periods']} rebalances), starting on "
        f"every rebalance date the backtest covers.",
        w["dates"],
        [("Model's picks", w["strategy"], "--series-1"),
         ("Equal-weight universe", w["benchmark"], "--series-2")], "pct", zero_line=True))
    out.append(histogram(
        f"{nm}: active return per {days}-day window",
        f"Picks minus benchmark across all {w['n_windows']} windows; the verdict needs "
        f"more than half of them above zero AND t >= 2 on the independent windows.",
        w["active_hist"], "pct", 0.0, "no edge",
        [["beat benchmark", f"{w['win_rate']:.0%}" if w.get("win_rate") is not None
          else "n/a"],
         ["finished up", f"{w['positive_rate']:.0%}" if w.get("positive_rate") is not None
          else "n/a"]]))
    return "".join(out)


def _etf_section(payload: dict) -> str:
    """The ETFs tab (v4.19): what the ETF sleeve holds today and why."""
    ccy = payload["meta"].get("base_currency", "")
    etf_sleeves = [s for s in payload["sleeves"] if s.get("feature_set") == "etf"
                   or s["name"] == "etf"]
    if not etf_sleeves:
        return ('<h2>ETFs</h2><p class="note">No ETF sleeve in this run. Add an '
                "<b>etf</b> entry under <b>sleeves</b> in config.yaml.</p>")
    out: list[str] = []
    for s in etf_sleeves:
        nm = _label(s["name"])
        used = s.get("selection_used") or "all"
        status = s.get("universe_status") or {}
        n_ok = int(status.get("ok", 0))
        n_cat = int(sum(status.values())) if status else 0
        tiles = [
            _tile("ETF universe",
                  f"{s.get('n_instruments') or 0:,} priced today",
                  (f"{n_ok:,} of {n_cat:,} UMushroom ETFs map to Yahoo; "
                   f"{s.get('n_core') or 0} core") if n_cat else
                  f"{s.get('n_core') or 0} core ETFs"),
            _tile("Holding today",
                  {"model": "model picks", "core": "core basket",
                   "all": "everything"}.get(used, used),
                  f"{s['weight']:.0%} of capital, rule: {s.get('selection')}"),
            _tile("Forecast horizon",
                  f"{s.get('horizon_days') or '?'} trading days",
                  f"inside the {s.get('contest_window_days') or '?'}-day contest window"),
            _tile("Learner and weights",
                  f"{s.get('model_type') or 'ridge'} / {s.get('weighting') or 'equal'}",
                  ("near-duplicate guard: max correlation "
                   f"{s['max_pick_correlation']:.2f}"
                   if s.get("max_pick_correlation") else "near-duplicate guard off")),
        ]
        if s.get("ic") is not None:
            tiles.append(_tile("Ranking skill (out of sample)",
                               f"IC {s['ic']:+.3f}", f"t = {s['ic_t']:+.2f}"
                               + (f" over {s['years']:.1f} years" if s.get("years") else "")))
        if s.get("alpha") is not None:
            tiles.append(_tile("Alpha vs equal-weight ETFs", f"{s['alpha']:+.1%} a year",
                               f"t = {s['alpha_t']:+.2f}"))
        out.append(f"<h2>{_esc(nm)} sleeve</h2>")
        out.append(f'<section class="tiles">{"".join(tiles)}</section>')
        if s.get("decision"):
            out.append(f'<p class="note"><b>Selection rule this run:</b> '
                       f'{_esc(s["decision"])}</p>')
        if s.get("universe_missing"):
            out.append('<p class="note"><b>Universe file missing:</b> only the core ETFs '
                       "were available. Build it with "
                       "<code>python -m umushroom.etf_universe</code> and commit the "
                       "CSV.</p>")
        orders = [o for o in payload["orders"] if o.get("sleeve") == s["name"]]
        out.append("<h3 class=\"sec\">Today's ETF orders</h3>")
        out.append('<p class="note">Enter these on UMushroom. "UMushroom" is the ticker '
                   "and pricing currency shown on the platform; check each local price "
                   "against UMushroom's before buying.</p>")
        out.append(_etf_orders_table(orders, ccy) if orders else
                   '<p class="note">No ETF orders this run.</p>')
        top = (payload.get("top_etfs") or {}).get(s["name"]) or []
        if top:
            out.append('<h3 class="sec">Top-ranked ETFs</h3>')
            out.append('<p class="note">The ETF model\'s 25 highest-ranked ETFs today, '
                       'ranked within their asset class. "Better than" is the share of '
                       "the universe ranked below.</p>")
            out.append(_etf_top_table(top))
        out.append(_window_cards(s))
        c = s.get("curves")
        if c:
            out.append(line_chart(
                f"{nm}: out-of-sample performance",
                "Growth of 1 when every week was traded by a model trained only on earlier "
                "data (purged walk-forward), against holding every ETF equally.",
                c["dates"],
                [("Model's picks", c.get("model", c["strategy"]), "--series-1"),
                 ("Equal-weight universe", c["benchmark"], "--series-2")], "x"))
        if status:
            rows = [[k, str(v)] for k, v in sorted(status.items(),
                                                   key=lambda kv: -kv[1])]
            out.append('<h3 class="sec">Universe build</h3>')
            out.append('<p class="note">Every ETF on UMushroom\'s public catalogue, and '
                       "whether Yahoo Finance can price it. Only <b>ok</b> rows are "
                       "traded; the rest are listed in umushroom_etf_universe.csv with "
                       "the reason.</p>")
            out.append(_plain_table(["Status", "ETFs"], rows))
    out.append('<h3 class="sec">How to read this</h3><ul class="notes">'
               "<li>The ETF model ranks ETFs within their asset class on trend factors "
               "(1-, 3-, 6- and 12-month momentum, 200-day trend, drawdown, volatility, "
               "beta), the same way the stock model ranks stocks.</li>"
               "<li>Every return it is trained to capture is shorter than the contest, "
               "and the contest-window verdict above checks that the picks actually beat "
               "the benchmark over stretches as long as Stage 1.</li>"
               "<li>Under <b>selection: auto</b> the picks are only traded while both the "
               "ranking skill and the window verdict hold; otherwise the sleeve holds the "
               "core basket, equal weight, and this page says so.</li></ul>")
    return "".join(out)

_READ_FIRST = ('<h2>Read this first</h2><ul class="notes">'
               "<li>Backtests use today's listed companies, so firms that failed since "
               "2016 are missing (survivorship bias). Absolute returns look better than "
               "reality; the alpha against the same universe is affected far less.</li>"
               "<li>Monte Carlo resamples real past weeks. It shows a plausible range, "
               "not a forecast, and cannot foresee an unprecedented crash.</li>"
               "<li>Sleeves marked \"holds everything\" failed (or never had) "
               "evidence of skill, so they deliberately don't use the model's picks.</li>"
               "</ul>")


def _tiles(payload: dict) -> str:
    ccy = payload["meta"].get("base_currency", "")
    eq = next((s for s in payload["sleeves"] if s["name"] == "equity"), None)
    mc = payload.get("mc")
    tiles = [_tile("Invested today",
                   f"{sum(o.get('allocated_value') or 0 for o in payload['orders']):,.0f} {ccy}",
                   f"{len(payload['orders'])} orders")]
    if eq and eq.get("alpha") is not None:
        tiles.append(_tile("Stock alpha (out-of-sample)", f"{eq['alpha']:+.1%} a year",
                           f"t = {eq['alpha_t']:+.2f}"
                           + (f" over {eq['years']:.1f} years" if eq.get("years") else "")))
    if eq and eq.get("alpha_boot"):
        tiles.append(_tile("Chance the stock alpha is real",
                           f"{eq['alpha_boot']['prob_positive']:.0%}",
                           "share of bootstrap resamples above zero"))
    etf = next((s for s in payload["sleeves"] if s["name"] == "etf"), None)
    if etf is not None:
        used = etf.get("selection_used") or "all"
        w = etf.get("window") or {}
        tiles.append(_tile("ETF sleeve",
                           {"model": "model picks", "core": "core basket",
                            "all": "all, equal weight"}.get(used, used),
                           (f"{etf.get('n_instruments') or 0:,} ETFs ranked; window "
                            f"verdict {w.get('verdict')}") if w else
                           f"{etf.get('n_instruments') or 0:,} ETFs"))
    as_of = str(payload["meta"].get("as_of") or "")
    if as_of:
        try:
            days = int(payload["meta"].get("rebalance_days") or DEFAULT_REBALANCE_DAYS)
            tiles.append(_tile("Next sell / rebalance check",
                               next_rebalance(as_of, days).strftime("%a %d %b"),
                               f"first run of the next {days}-trading-day period; "
                               f"hold this period's book until then"))
        except ValueError:
            pass
    wk = (payload.get("risk") or {}).get("weekly")
    if wk:
        v = payload["risk"].get("value")
        tiles.append(_tile("Value at Risk (95%, 1 week)",
                           f"-{wk['var95'] * v:,.0f} {ccy}" if v else f"-{wk['var95']:.1%}",
                           f"{wk['var95']:.1%}; average in the worst 5%: {wk['es95']:.1%}"))
    if mc:
        st = mc["stats"]
        tiles.append(_tile(f"Median value after {mc['horizon_weeks']} weeks",
                           f"{st['median_final']:,.0f}", f"{st['median_return']:+.1%}"))
        tiles.append(_tile("Chance of a loss", f"{st['prob_loss']:.0%}",
                           f"5% worst case: {st['p5_final'] - mc['start_value']:+,.0f}"))
        tiles.append(_tile("Chance of beating the benchmark",
                           f"{st['prob_beat_benchmark']:.0%}",
                           f"over {mc['horizon_weeks']} weeks"))
    return f'<section class="tiles">{"".join(tiles)}</section>'


def _money(frac: float, value: float | None, ccy: str) -> str:
    return f"{frac:.1%}" + (f" (≈ {frac * value:,.0f} {ccy})" if value else "")


def _var_backtest_card(bt: dict) -> str:
    detail = (f"Each week, VaR was computed from only the previous {bt['window']} weeks. "
              f"The next week lost more than it {bt['breaches']} times out of {bt['n']} "
              f"({bt['breach_rate']:.1%}; a correct {bt['level']:.0%} VaR is broken about "
              f"{bt['expected_rate']:.0%} of the time). Kupiec test p = {bt['p_value']:.2f}"
              + ((" -- consistent." if bt["n"] >= 100 else
                  f" -- consistent, but only {bt['n']} weeks could be tested, so this "
                  "can only catch large errors.") if bt["status"] == "PASS" else
                 " -- breaks too rarely: VaR is cautious." if bt["status"] == "WARN" else
                 " -- breaks too often: VaR UNDERSTATES the risk."))
    return (f'<div class="card"><h3>Portfolio: VaR backtest</h3><ul class="checks"><li>'
            f'<span class="badge {_esc(bt["status"].lower())}">{_esc(bt["status"])}</span> '
            f"<b>95% VaR held up out of sample</b>: {_esc(detail)}</li></ul></div>")


def _factor_risk_block(payload: dict) -> str:
    """v4.20: the factor-risk decomposition of every sleeve sized by
    `weighting: factor` -- which factor carries how much of the risk, and
    each pick's exposures."""
    parts: list[str] = []
    for s in payload.get("sleeves", []):
        fr = s.get("factor_risk")
        if not fr:
            continue
        factors = list(fr.get("factors") or [])
        parts.append(f'<h3 class="sec">Factor risk: {_esc(_label(s["name"]))} sleeve</h3>')
        parts.append('<p class="note">Weights were set so that each risk factor (the '
                     "average of its group in the sleeve's universe) contributes its "
                     "budgeted share of the picks' systematic risk, using the factors' "
                     "full covariance. Exposures are 1-year betas.</p>")
        idio = fr.get("idiosyncratic_share")
        rows = [[f, f"{(fr['contributions'].get(f) or 0):.1%}",
                 f"{(fr['budgets'].get(f) or 0):.1%}"] for f in factors]
        parts.append(_plain_table(["Factor", "Share of systematic risk", "Budget"], rows))
        if idio is not None:
            parts.append(f'<p class="note">Not explained by the factors (idiosyncratic): '
                         f"{idio:.1%} of total variance.</p>")
        picks = list((fr.get("weights") or {}).items())
        rows = [[t, f"{w:.1%}"] + [f"{(fr['exposures'].get(t) or {}).get(f) or 0:+.2f}"
                                  for f in factors] for t, w in picks]
        parts.append(_plain_table(["Pick", "Weight"] + [f"beta {f}" for f in factors], rows))
    return "".join(parts)


def _risk_section(payload: dict) -> str:
    risk = payload.get("risk")
    ccy = payload["meta"].get("base_currency", "")
    factor_block = _factor_risk_block(payload)
    out = ["<h2>Value at Risk</h2>",
           ('<p class="note"><b>VaR 95%</b>: in 19 weeks out of 20 the portfolio should '
            "lose less than this. <b>Expected shortfall (ES) 95%</b>: in the worst 1 week "
            "in 20, the average loss. ES is the honest one for bad days -- VaR only "
            "says where the bad tail starts, not how deep it goes.</p>")]
    if not risk:
        return "".join(out) + ('<p class="note">Not enough out-of-sample history in this '
                               "run to estimate VaR (needs at least 20 weeks).</p>"
                               + factor_block)
    v, wk, hz = risk.get("value"), risk.get("weekly"), risk.get("horizon")
    tiles = []
    if wk:
        tiles += [_tile("1-week VaR 95%", f"{wk['var95']:.1%}",
                        f"≈ {wk['var95'] * v:,.0f} {ccy}" if v else ""),
                  _tile("1-week ES 95%", f"{wk['es95']:.1%}",
                        f"≈ {wk['es95'] * v:,.0f} {ccy}" if v else ""),
                  _tile("1-week VaR 99%", f"{wk['var99']:.1%}",
                        f"≈ {wk['var99'] * v:,.0f} {ccy}" if v else ""),
                  _tile("Worst week seen", f"{wk['worst']:.1%}",
                        f"in {wk['n']} out-of-sample weeks")]
    if hz:
        tiles += [_tile(f"{hz['weeks']}-week VaR 95%", f"{hz['var95']:.1%}",
                        f"≈ {hz['var95'] * v:,.0f} {ccy} (Monte Carlo)" if v
                        else "Monte Carlo"),
                  _tile(f"{hz['weeks']}-week ES 95%", f"{hz['es95']:.1%}",
                        f"≈ {hz['es95'] * v:,.0f} {ccy}" if v else "")]
    out.append(f'<section class="tiles">{"".join(tiles)}</section>')

    rows = []
    if wk:
        rows += [["1-week VaR (history)", _money(wk["var95"], v, ccy),
                  _money(wk["var99"], v, ccy)],
                 ["1-week ES (history)", _money(wk["es95"], v, ccy),
                  _money(wk["es99"], v, ccy)]]
    if hz:
        rows += [[f"{hz['weeks']}-week VaR (Monte Carlo)", _money(hz["var95"], v, ccy),
                  _money(hz["var99"], v, ccy)],
                 [f"{hz['weeks']}-week ES (Monte Carlo)", _money(hz["es95"], v, ccy),
                  _money(hz["es99"], v, ccy)]]
    out += ['<h3 class="sec">All measures</h3>', _plain_table(["Measure", "95%", "99%"], rows)]
    if wk and wk["n"] < 100:
        out.append(f'<p class="note">The 99% figures rest on only {wk["n"]} weeks '
                   "(about 1 week in 100 is the tail), so treat them as rough.</p>")

    if wk:
        out.append(histogram(
            "Weekly portfolio returns (out of sample)",
            f"Every out-of-sample week, all sleeves at their capital weights. The line "
            f"marks the 95% VaR: {wk['var95']:.1%}.",
            wk["hist"], "pct", -wk["var95"], "95% VaR",
            [["95% VaR", f"-{wk['var95']:.2%}"], ["95% ES", f"-{wk['es95']:.2%}"],
             ["99% VaR", f"-{wk['var99']:.2%}"], ["Worst week", f"-{wk['worst']:.2%}"]]))
    bt = risk.get("backtest")
    if bt:
        out.append(_var_backtest_card(bt))
        out.append(line_chart(
            "Did the VaR hold? Weekly return vs. the VaR limit",
            f"The limit each week uses only the previous {bt['window']} weeks. Weeks "
            f"below the limit are breaches: {bt['breaches']} of {bt['n']}.",
            bt["dates"], [("Weekly return", bt["returns"], "--series-1"),
                          ("95% VaR limit (known in advance)", bt["limit"], "--series-2")],
            "pct", zero_line=True))
    if risk.get("sleeves"):
        out += ['<h3 class="sec">By sleeve (1 week, what each sleeve holds)</h3>', _plain_table(
            ["Sleeve", "Weeks", "VaR 95%", "ES 95%", "VaR 99%", "ES 99%"],
            [[_label(x["name"]), str(x["n"]), f"{x['var95']:.1%}", f"{x['es95']:.1%}",
              f"{x['var99']:.1%}", f"{x['es99']:.1%}"] for x in risk["sleeves"]])]
    out.append(factor_block)
    return "".join(out)


def _period_note(payload: dict) -> str:
    """v4.21: the book is held for a whole rebalance period; say so where
    the orders are, so a daily visitor does not trade daily."""
    as_of = str(payload["meta"].get("as_of") or "")
    if not as_of:
        return ""
    try:
        days = int(payload["meta"].get("rebalance_days") or DEFAULT_REBALANCE_DAYS)
        start, nxt = trade_day(as_of), next_rebalance(as_of, days)
    except ValueError:
        return ""
    return (f'<p class="note"><b>Holding period:</b> these orders are the book for the '
            f"{days}-trading-day period that starts {start.strftime('%a %d %b')} "
            f"(signals of {as_of}); the next rebalance is the first run on or after "
            f"{nxt.strftime('%a %d %b')}. If you already hold this period's book, do "
            "not re-trade it -- see <b>When to sell</b>.</p>")


def section_bodies(payload: dict) -> dict[str, str]:
    """The report split into self-contained sections (one per website tab)."""
    ccy = payload["meta"].get("base_currency", "")
    mc = payload.get("mc")

    stocks = ["<h2>Today's orders</h2>",
              ('<p class="note">Enter these on UMushroom. Check each local price against '
               "UMushroom's before buying; rows with \"Priced from\" use a substitute "
               "Yahoo symbol.</p>"),
              _period_note(payload),
              _orders_table(payload["orders"], ccy)]
    if payload["top"]:
        stocks += ["<h2>Top-ranked stocks</h2>",
                   ('<p class="note">The model\'s 25 highest-ranked stocks today. "Better '
                    'than" is the share of the universe ranked below it.</p>'),
                   _top_table(payload["top"])]

    graphs = ["<h2>Out-of-sample performance</h2>"]
    graphs += [_sleeve_section(s) for s in payload["sleeves"]]
    if payload["deciles"]:
        d = payload["deciles"]
        graphs.append(bar_chart(
            "Stocks: return by prediction decile",
            "Average weekly return of each tenth of the universe, ranked by the model "
            "(1 = lowest). Real ranking skill rises from left to right.",
            [str(i + 1) for i in range(len(d))], [v or 0.0 for v in d], "pct2"))

    mcs: list[str] = []
    if mc:
        st = mc["stats"]
        mcs.append("<h2>Monte Carlo: what the contest could look like</h2>")
        if payload.get("mc_note"):
            mcs.append(f'<p class="note">{_esc(payload["mc_note"])}</p>')
        mcs.append(fan_chart(mc, ccy))
        mcs.append(histogram(
            f"Final value after {mc['horizon_weeks']} weeks",
            f"Spread of outcomes across {mc['n_paths']:,} simulations, starting from "
            f"{mc['start_value']:,.0f} {ccy}.",
            mc["final_hist"], "money", mc["start_value"], "start",
            [["5th", _fmt(st["p5_final"], "money")],
             ["median", _fmt(st["median_final"], "money")],
             ["95th", _fmt(st["p95_final"], "money")]]))
    else:
        mcs.append('<h2>Monte Carlo</h2><p class="note">No simulation in this run '
                   "(it needs at least 20 weeks of out-of-sample history).</p>")

    checks = "".join(_sleeve_checks(s) for s in payload["sleeves"])
    if (payload.get("risk") or {}).get("backtest"):
        checks += _var_backtest_card(payload["risk"]["backtest"])
    checks = "<h2>Validation checks</h2>" + (checks or (
        '<p class="note">No checks in this run. Run with <b>--validate</b> (or start '
        "the website with --validate) to include them.</p>"))

    return {"overview": _tiles(payload) + _READ_FIRST, "stocks": "".join(stocks),
            "etfs": _etf_section(payload),
            "graphs": "".join(graphs), "montecarlo": "".join(mcs),
            "risk": _risk_section(payload), "checks": checks}


def render_page(payload: dict, body: str, title: str = "UMushroom strategy report",
                nav: str = "", sub: str | None = None) -> str:
    """A full page in the report's style (charts, tooltips, dark mode)."""
    meta = payload.get("meta", {})
    ccy = meta.get("base_currency", "")
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return _PAGE.format(
        title=_esc(title), nav=nav,
        sub=_esc(sub if sub is not None else
                 f"Signals as of the {meta.get('as_of', '')} close · generated "
                 f"{meta.get('generated', '')} · prices in {ccy} · v{meta.get('version', '')}"),
        body=body, css=_CSS, js=_JS, data=data)


def render_html(payload: dict) -> str:
    """The whole report on one page."""
    b = section_bodies(payload)
    tiles, _, _ = b["overview"].partition("<h2>Read this first</h2>")
    return render_page(payload, tiles + b["stocks"] + b["etfs"] + b["graphs"]
                       + b["montecarlo"] + b["risk"] + b["checks"] + _READ_FIRST)


def write_report(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.write_text(render_html(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------- chrome

_DARK = """--surface-0:#121211;--surface-1:#1a1a19;--text-primary:#ffffff;
 --text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);
 --series-1:#3987e5;--series-2:#d95926;--band-key:#1c5cab;--good:#0ca30c;
 --warn:#fab219;--bad:#e66767"""

_CSS = """
:root{color-scheme:light;--surface-0:#f4f3f0;--surface-1:#fcfcfb;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--border:rgba(11,11,11,.10);
 --series-1:#2a78d6;--series-2:#eb6834;--band-key:#86b6ef;--good:#006300;
 --warn:#8a5a00;--bad:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;
 """ + _DARK + """}}
:root[data-theme="dark"]{color-scheme:dark;""" + _DARK + """}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
 font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 12px}h3{font-size:16px;margin:0}h3.sec{margin:24px 0 8px}.scroll+.note{margin-top:8px}
.sub,.note{color:var(--text-secondary);margin:0 0 12px;font-size:14px}
h3.sleeve{margin:28px 0 4px}h3.sleeve span{font-weight:400;color:var(--text-secondary);
 font-size:14px;margin-left:8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile,.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px}
.tile{padding:14px}.card{padding:16px;margin:16px 0}
.t-l{font-size:13px;color:var(--text-secondary)}
.t-v{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums;margin:4px 0}
.t-n{font-size:12px;color:var(--muted)}
figure{margin:16px 0}figcaption p{margin:2px 0 8px;color:var(--text-secondary);font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--text-secondary);
 margin:4px 0 8px}
.key-line{display:inline-block;width:16px;height:2px;vertical-align:middle;margin-right:6px}
.key-box{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;
 margin-right:6px}
.band-outer-key{background:color-mix(in srgb,var(--series-1) 12%,transparent)}
.band-inner-key{background:color-mix(in srgb,var(--series-1) 30%,transparent)}
.plot{position:relative}svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}.zero{stroke:var(--muted);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}.endlab{fill:var(--text-secondary);font-size:11px}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface-1);stroke-width:2}
.band-outer{fill:var(--series-1);opacity:.10}.band-inner{fill:var(--series-1);opacity:.20}
.marker{stroke:var(--text-secondary);stroke-width:1}
.xhair{stroke:var(--muted);stroke-width:1;opacity:0}.plot:hover .xhair{opacity:1}
.bar .hit{fill:transparent}.bar:hover>:not(.hit),.bar:focus>:not(.hit){opacity:.75}
.bar:focus{outline:none}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);
 border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;
 box-shadow:0 4px 16px rgba(0,0,0,.14);white-space:nowrap;z-index:2}
.tip .row{display:flex;align-items:center;gap:6px}.tip b{font-variant-numeric:tabular-nums}
.tip .nm{color:var(--text-secondary)}
details.tv{margin-top:8px;font-size:13px}details.tv summary{color:var(--text-secondary);
 cursor:pointer}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--text-secondary);font-weight:600}
table.data{background:var(--surface-1);border:1px solid var(--border)}
ul.checks{list-style:none;padding:0;margin:8px 0 0;font-size:13px}ul.checks li{margin:6px 0}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:1px 6px;border-radius:4px;
 border:1px solid currentColor}
.badge.pass{color:var(--good)}.badge.warn{color:var(--warn)}.badge.fail{color:var(--bad)}
.notes li{margin:6px 0;color:var(--text-secondary)}
"""

_JS = r"""
(function(){
function fmt(v,k){if(v===null||v===undefined||isNaN(v))return'n/a';
 if(k==='pct'||k==='pct2')return(v>=0?'+':'')+(v*100).toFixed(k==='pct2'?2:1)+'%';
 if(k==='money')return Math.round(v).toLocaleString();if(k==='x')return v.toFixed(2)+'x';
 return v.toFixed(2);}
document.querySelectorAll('.plot').forEach(function(plot){
 var tip=plot.querySelector('.tip'),svg=plot.querySelector('svg'),raw=plot.getAttribute('data-chart');
 function place(x,y){var r=plot.getBoundingClientRect();
  tip.style.left=Math.max(0,Math.min(x+14,r.width-tip.offsetWidth-4))+'px';
  tip.style.top=Math.max(0,y-tip.offsetHeight-10)+'px';}
 if(raw){var d=JSON.parse(raw),xh=svg.querySelector('.xhair'),n=d.labels.length;
  plot.addEventListener('pointermove',function(e){
   var r=svg.getBoundingClientRect(),vx=(e.clientX-r.left)*svg.viewBox.baseVal.width/r.width;
   var i=Math.max(0,Math.min(n-1,Math.round((vx-d.x0)/(d.x1-d.x0)*(n-1))));
   var px=d.x0+(d.x1-d.x0)*i/Math.max(n-1,1);xh.setAttribute('x1',px);xh.setAttribute('x2',px);
   tip.textContent='';var h=document.createElement('div');h.className='nm';
   h.textContent=d.labels[i];tip.appendChild(h);
   d.series.forEach(function(s){var row=document.createElement('div');row.className='row';
    var k=document.createElement('i');k.className='key-line';k.style.background='var('+s.color+')';
    var b=document.createElement('b');b.textContent=fmt(s.values[i],d.kind);
    var nm=document.createElement('span');nm.className='nm';nm.textContent=s.name;
    row.appendChild(k);row.appendChild(b);row.appendChild(nm);tip.appendChild(row);});
   tip.hidden=false;place(e.clientX-r.left,e.clientY-r.top);});
  plot.addEventListener('pointerleave',function(){tip.hidden=true;});}
 plot.querySelectorAll('.bar').forEach(function(g){
  function show(x,y){tip.textContent=g.getAttribute('data-tip');tip.hidden=false;place(x,y);}
  g.addEventListener('pointermove',function(e){var r=plot.getBoundingClientRect();
   show(e.clientX-r.left,e.clientY-r.top);});
  g.addEventListener('pointerleave',function(){tip.hidden=true;});
  g.addEventListener('focus',function(){var r=plot.getBoundingClientRect(),
   b=g.getBoundingClientRect();show(b.left-r.left,b.top-r.top);});
  g.addEventListener('blur',function(){tip.hidden=true;});});
});})();
"""

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body>{nav}<main><h1>{title}</h1><p class="sub">{sub}</p>{body}</main>
<script id="report-data" type="application/json">{data}</script>
<script>{js}</script></body></html>
"""
