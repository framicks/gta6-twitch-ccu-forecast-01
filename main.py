"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers.

Model: learn the intraday shape of a big console launch from the case's
comparables (past launches' hourly Twitch category viewers).

For every comparable launch we take its release day (UTC, hours 1..24) and
the log-ratio of each hour's viewers to that launch's hour-1 print:

    r_h = ln(viewers_h / viewers_1)

Only launches whose hour 1 was already "live" are used for training: hour-1
viewers at least 15% of that launch's release-day peak. (Launches that unlock
later in the UTC day have a near-zero hour 1; their ratios are pre-launch
noise, not a shape.) The Grand Theft Auto VI category's own pre-release
history is not a launch and is excluded.

For each hour h = 2..24 the forecast is

    q_tau(h) = p1 * exp(mu_h + z_tau * sigma_h)

where p1 is the case's hour-1 print, mu_h is the mean of r_h across the
training launches, sigma_h is the interquartile range of r_h divided by
1.349 (the normal IQR), clamped to [0.10, 1.0], and z_tau is the standard
normal quantile at the scored levels. The learned curve dips in the deep
night (hours 3-11, mu ~ -0.3) and rises to the evening peak (hours 19-21,
mu ~ +0.5): the flat house baseline ignores this shape entirely, which is
the edge this model exploits.

Validation (leave-one-out over the 26 usable comparables, and the 4 practice
cases scored against SullyGnome's hourly averages fetched once each):
  practice: this model 0.308 vs house baseline 0.387
  LOO (h1/peak >= 0.15): this model 0.238 vs house baseline 0.284
  LOO (h1/peak >= 0.30): this model 0.207 vs house baseline 0.251
Lower pinball loss is better.

Guardrails: quantiles are non-decreasing, > 0 and <= 1e9; any hour with
fewer than 5 training launches falls back to the house baseline's widening
spread around a flat median; if the comparables are unusable at all, the
whole forecast falls back to the baseline formula; a tiny hour-1 print
(< 1) counts as 1. Deterministic: comparables are processed in name order,
no randomness, no network, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Exact z values for the scored levels, so the validated numbers are the
# ones the program ships; any other level is computed, not guessed.
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Used only for levels outside the scored five; accurate to ~1e-9, more
    than enough for a quantile that is then rounded to 2 decimals.
    """
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= 1 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def z_values(levels: list) -> list[float]:
    """Standard normal quantiles for the levels a case asks for."""
    out = []
    for level in levels:
        level = float(level)
        if level in Z_TABLE:
            out.append(Z_TABLE[level])
        elif 0 < level < 1:
            out.append(norm_ppf(level))
        else:
            raise ValueError(f"quantile level out of (0, 1): {level!r}")
    return out

# The house baseline's spread, in log units per square-root hour (fallback).
BASE_SIGMA = 0.15

# Training-set rules (validated by leave-one-out; see module docstring).
MIN_HOURS = 20        # a comparable needs at least this many release-day hours
MIN_RATIO = 0.15      # its hour-1 print must be >= this share of its day peak
MIN_LAUNCHES = 5      # an hour needs at least this many training launches
SIGMA_FLOOR = 0.10
SIGMA_CAP = 1.0


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as the starter rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp, tolerating a missing 'Z'."""
    text = text.strip()
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return datetime.fromisoformat(text.replace("Z", "+00:00").replace("+00:00", ""))


def quantile(sorted_xs: list[float], p: float) -> float:
    """Linear-interpolation quantile of an already-sorted list."""
    n = len(sorted_xs)
    if n == 1:
        return sorted_xs[0]
    i = p * (n - 1)
    lo = int(i)
    frac = i - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[min(lo + 1, n - 1)] * frac


def day_series(comp: dict) -> dict[int, float]:
    """A comparable's release-day hourly viewers, {hour 1..24: viewers}.

    The comparable's `hourly` points run from 14 days before its release
    day to 3 days after it; only the 24 hours of `day_start`'s UTC day
    count, keyed by their `from` timestamp.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out: dict[int, float] = {}
    for row in comp.get("hourly") or []:
        try:
            t = parse_ts(row["from"])
            viewers = row.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        delta = (t - day_start).total_seconds() / 3600
        if 0 <= delta < 24 and delta == int(delta):
            out[int(delta) + 1] = float(viewers)
    return out


def select_training(case: dict) -> list[dict[int, float]]:
    """The comparables usable as launch-shape training data, in name order."""
    training = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: c.get("name", "")):
        name = (comp.get("name") or "").lower()
        if "before release" in name:
            # The target category's own pre-release history is not a launch.
            continue
        series = day_series(comp)
        if 1 not in series or len(series) < MIN_HOURS:
            continue
        peak = max(series.values())
        if peak <= 0 or series[1] / peak < MIN_RATIO:
            # Hour 1 was not yet live (unlock later in the UTC day).
            continue
        training.append(series)
    return training


def fit_curve(training: list[dict[int, float]]) -> dict[int, tuple[float, float]]:
    """Per-hour (mu, sigma) of log(viewers_h / viewers_1) across training."""
    curve: dict[int, tuple[float, float]] = {}
    for h in range(2, 25):
        ratios = sorted(
            math.log(s[h] / s[1]) for s in training if h in s and 1 in s
        )
        if len(ratios) < MIN_LAUNCHES:
            continue
        mu = sum(ratios) / len(ratios)
        iqr = quantile(ratios, 0.75) - quantile(ratios, 0.25)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, iqr / 1.349))
        curve[h] = (mu, sigma)
    return curve


def hour_quantiles(p1: float, h: int, curve: dict[int, tuple[float, float]],
                   zlist: list[float]) -> list:
    """The quantiles for hour h: learned shape, baseline fallback."""
    if h in curve:
        mu, sigma = curve[h]
        raw = [p1 * math.exp(mu + z * sigma) for z in zlist]
    else:
        raw = [p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)) for z in zlist]
    out: list = []
    prev = 0.0
    for value in raw:
        q = round2(min(1e9, max(0.01, value)))
        # Rounding must not break the non-decreasing order.
        if q < prev:
            q = prev
        out.append(q)
        prev = float(q)
    return out


def hour1_print(case: dict) -> float:
    """The case's hour-1 print; a print below 1 counts as 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            try:
                return max(1.0, float(row["viewers"]))
            except (KeyError, TypeError, ValueError):
                break
    # Defensive fallback: last pre-checkpoint hourly point, else 1.
    history = case.get("history_hourly") or case.get("history") or []
    for row in reversed(history):
        try:
            return max(1.0, float(row["viewers"]))
        except (KeyError, TypeError, ValueError):
            continue
    return 1.0


def forecast_case(case: dict) -> dict:
    training = select_training(case)
    curve = fit_curve(training)
    p1 = hour1_print(case)
    zlist = z_values(case.get("levels") or LEVELS)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], curve, zlist)}
            for row in rows
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
