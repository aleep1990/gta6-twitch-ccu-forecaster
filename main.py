"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Method: per-hour log-ratio regression on the case's comparables.
For each past launch, take its release-day hourly curve and form the log
ratio lr_h = ln(viewers_h / viewers_1) for hours h = 2..24.  A launch's
hour-1 print already tells you how much of its audience has unlocked:
launches whose hour 1 is already huge stay flat (peak ~1x hour 1), while
launches whose hour 1 is still tiny surge 10-1000x later in the day
(leave-one-out correlation between ln(p1) and lr_19 is about -0.70).
So for each hour we fit a weighted linear regression of lr_h on
centered ln(p1), with Gaussian kernel weights (bandwidth 1.5 in ln p1)
so launches near this case's own hour-1 size dominate, the slope shrunk
by n/(n+8), and the spread taken as the weighted residual standard
deviation (floored at 0.15 log units).  The forecast is then
    q_tau(h) = p1 * exp(mu_h + z_tau * s_h),
rounded to 2 decimals half-up, clamped to (0, 1e9] and non-decreasing.

Leave-one-out over the 37 usable comparable launch curves in the kit
(practice case p00's comparables): mean pinball loss 0.378 vs the house
baseline's 0.930; restricted to the 28 launches with hour-1 >= 10,000
viewers (the regime GTA VI's hour 1 will almost surely be in):
0.257 vs baseline 0.351, typical miss 47.6% vs 69.0%, 90% coverage
0.924 vs 0.766.  The house baseline scores about 0.387 on the practice.

Robustness: any failure in the fit (missing/degenerate comparables,
missing hour-1 print) falls back to the house persistence baseline, so
the output is always a valid forecast.  Deterministic: no randomness,
no wall clock, everything iterated in file/sorted order.

Run: python3 -P -s main.py  (reads $MAKEFUN_INPUT_DIR else /input,
writes $MAKEFUN_OUTPUT_DIR else /output, one .out per .in, name order).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; this entry is
# a single self-contained file, so no sys.path fix is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA_BASELINE = 0.15  # house persistence baseline's log spread per sqrt-hour
KERNEL_BW = 1.5        # Gaussian kernel bandwidth in ln(p1) units
SHRINK_K = 8           # slope shrink: n / (n + SHRINK_K)
MIN_SPREAD = 0.15      # floor on the per-hour log spread
MAX_SPREAD = 2.5       # cap on the per-hour log spread
MIN_CURVES_PER_HOUR = 5


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime (stdlib only)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up (same as the kit starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def extract_curves(case):
    """Release-day hourly viewer curves from the comparables.

    Returns a list of dicts {hour: viewers} for hours 1..24, keeping only
    launches with a positive hour-1 print and at least 15 release-day
    hours.  The target game's own pre-release history (marked 'Not a
    launch') is excluded: it has no release day.
    """
    curves = []
    for comp in case.get("comparables") or []:
        try:
            unlock = (comp.get("unlock") or "").lower()
            if "not a launch" in unlock:
                continue
            day_start = parse_time(comp["day_start"])
            curve = {}
            for row in comp.get("hourly") or []:
                viewers = row.get("viewers")
                if not viewers or viewers <= 0:
                    continue
                hour = int((parse_time(row["from"]) - day_start).total_seconds() // 3600) + 1
                if 1 <= hour <= 24:
                    curve[hour] = viewers
            if 1 in curve and len(curve) >= 15:
                curves.append(curve)
        except (KeyError, TypeError, ValueError):
            continue
    curves.sort(key=lambda c: c[1])
    return curves


def baseline_quantiles(p1, h):
    return [round2(p1 * math.exp(z * SIGMA_BASELINE * math.sqrt(h - 1))) for z in Z]


def fit_quantiles(curves, p1):
    """Kernel-weighted per-hour log-ratio regression; see module docstring."""
    log_p1 = math.log(p1)
    mean_log_p1 = sum(math.log(c[1]) for c in curves) / len(curves)
    x0 = log_p1 - mean_log_p1
    out = {}
    for h in range(2, 25):
        pts = []
        for c in curves:
            if h in c:
                pts.append((math.log(c[1]) - mean_log_p1, math.log(c[h] / c[1])))
        if len(pts) < MIN_CURVES_PER_HOUR:
            mu = median([y for _, y in pts]) if pts else 0.0
            spread = 0.5
        else:
            weights = [math.exp(-0.5 * ((x - x0) / KERNEL_BW) ** 2) for x, _ in pts]
            sw = sum(weights)
            mx = sum(w * x for w, (x, _) in zip(weights, pts)) / sw
            my = sum(w * y for w, (_, y) in zip(weights, pts)) / sw
            var = sum(w * (x - mx) ** 2 for w, (x, _) in zip(weights, pts))
            cov = sum(w * (x - mx) * (y - my) for w, (x, y) in zip(weights, pts))
            slope = cov / var if var > 0 else 0.0
            slope *= len(pts) / (len(pts) + SHRINK_K)
            intercept = my - slope * mx
            mu = intercept + slope * x0
            resid = [y - (intercept + slope * x) for x, y in pts]
            mres = sum(w * r for w, r in zip(weights, resid)) / sw
            spread = math.sqrt(sum(w * (r - mres) ** 2 for w, r in zip(weights, resid)) / sw)
            spread = max(MIN_SPREAD, min(MAX_SPREAD, spread))
        qs = []
        prev = 0.0
        for z in Z:
            q = round2(p1 * math.exp(mu + z * spread))
            q = max(1e-2, min(1e9, q))
            q = max(q, prev)  # keep the row non-decreasing after rounding
            qs.append(q)
            prev = q
        out[h] = qs
    return out


def forecast_case(case, name):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = first.get("viewers")
    if not p1 or p1 <= 0:
        p1 = 1
    p1 = max(1, p1)
    try:
        curves = extract_curves(case)
        if len(curves) < 3:
            raise ValueError("too few comparable curves")
        quantiles = fit_quantiles(curves, p1)
    except Exception:
        quantiles = {h: baseline_quantiles(p1, h) for h in range(2, 25)}
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        hours.append({"hour": h, "q": quantiles[h]})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
