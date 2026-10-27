"""Time-series forecasting with Google's TimesFM 2.5 (200M, Apache-2.0), run locally.

TimesFM only predicts numbers; the language model still finds the history
with SQL. If TimesFM isn't installed, a simple seasonal baseline is used so
the app keeps working (and says so).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
MAX_HORIZON = 60

FORECAST_WORDS = re.compile(
    r"\b(forecast|predict|projection|next|coming|future)\b"
    r"|توقع|تنبؤ|تنبأ|القادم|القادمة|المقبل|المقبلة|المستقبل",
    re.IGNORECASE,
)
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}


def is_forecast_question(question: str) -> bool:
    return bool(FORECAST_WORDS.search(question))


BY_GROUP = re.compile(r"\b(by|each|per|every|across)\b|لكل|حسب|كل\s", re.IGNORECASE)


def asks_by_group(question: str) -> bool:
    """'forecast households by region' -> True; 'forecast households' -> False.
    'per month' alone doesn't count: it names the period, not a group."""
    q = re.sub(r"\b(per|each|every)\s+(day|week|month|quarter|year)\b", "", question, flags=re.I)
    return bool(BY_GROUP.search(q))


def parse_horizon(question: str, default: int = 3) -> int:
    """'next 3 months' -> 3; Arabic/Western digits; falls back to default."""
    q = question.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")).lower()
    m = re.search(r"(\d+)\s*(day|week|month|quarter|year|يوم|أيام|اسبوع|أسبوع|أسابيع|شهر|أشهر|شهور)", q)
    if m:
        return max(1, min(int(m.group(1)), MAX_HORIZON))
    for word, n in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(day|week|month|quarter|year)s?\b", q):
            return n
    return default


def infer_freq(dates: pd.Series) -> str:
    """Pandas offset alias from the median gap between dates."""
    gaps = pd.Series(sorted(dates.unique())).diff().dropna()
    if gaps.empty:
        return "MS"
    days = gaps.median().days
    if days <= 1:
        return "D"
    if days <= 8:
        return "W"
    if days <= 31:
        return "MS"
    if days <= 92:
        return "QS"
    return "YS"


class Forecaster:
    """Lazy wrapper: loads TimesFM on first use, falls back to a baseline."""

    def __init__(self):
        self._model = None
        self.backend = None  # "timesfm" or "baseline"

    def _load(self):
        if self.backend:
            return
        try:
            import timesfm  # noqa: WPS433 (optional dependency)
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_CHECKPOINT)
            model.compile(timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            ))
            self._model, self.backend = model, "timesfm"
        except Exception:
            self.backend = "baseline"

    def predict(self, series: list[np.ndarray], horizon: int):
        """Returns (point, low, high) arrays of shape (n_series, horizon)."""
        self._load()
        if self.backend == "timesfm":
            point, quant = self._model.forecast(horizon=horizon, inputs=series)
            # quant[..., 0] is the mean, then the 10th..90th percentiles
            return np.asarray(point), np.asarray(quant[..., 1]), np.asarray(quant[..., 9])
        return baseline_forecast(series, horizon)


def baseline_forecast(series, horizon):
    """Mean of recent values plus a damped linear trend, with a ±1.28σ band (~80%)."""
    points, lows, highs = [], [], []
    for s in series:
        s = np.asarray(s, dtype=float)
        recent = s[-min(len(s), 6):]
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent) > 1 else 0.0
        steps = np.arange(1, horizon + 1)
        p = recent.mean() + 0.5 * slope * (steps + (len(recent) - 1) / 2)
        sd = np.std(np.diff(s)) if len(s) > 2 else abs(recent.mean()) * 0.1
        band = 1.28 * sd * np.sqrt(steps)
        p = np.maximum(p, 0) if (s >= 0).all() else p
        points.append(p)
        lows.append(np.maximum(p - band, 0) if (s >= 0).all() else p - band)
        highs.append(p + band)
    return np.array(points), np.array(lows), np.array(highs)


def forecast_frame(df: pd.DataFrame, horizon: int, forecaster: Forecaster,
                   keep_groups: bool = True) -> pd.DataFrame:
    """Turn an SQL result (date, [group], value) into history + forecast rows.
    keep_groups=False sums the groups per period (the question didn't ask for them).

    Output columns: period, [group], value, low, high, kind ('actual'|'forecast').
    """
    date_col = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None)
    if date_col is None:  # try to parse the first column as dates
        date_col = df.columns[0]
        df = df.assign(**{date_col: pd.to_datetime(df[date_col], errors="raise")})
    num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("The query result has no numeric column to forecast.")
    value_col = num_cols[-1]
    group_cols = [c for c in df.columns if c not in (date_col, value_col) and c not in num_cols]
    group_col = group_cols[0] if group_cols and keep_groups else None

    df = df.sort_values(date_col)
    freq = infer_freq(df[date_col])
    groups = list(df.groupby(group_col, sort=True)) if group_col else [(None, df)]
    series, futures = [], []
    for _, g in groups:
        g = g.groupby(date_col, as_index=False)[value_col].sum()
        if len(g) < 3:
            raise ValueError("At least 3 past periods are needed to forecast.")
        series.append(g[value_col].to_numpy(dtype=float))
        futures.append(pd.date_range(g[date_col].iloc[-1], periods=horizon + 1, freq=freq)[1:])

    point, low, high = forecaster.predict(series, horizon)

    rows = []
    for i, (key, g) in enumerate(groups):
        g = g.groupby(date_col, as_index=False)[value_col].sum()
        for d, v in zip(g[date_col], g[value_col]):
            rows.append({"period": d, **({group_col: key} if group_col else {}),
                         "value": v, "low": None, "high": None, "kind": "actual"})
        for j, d in enumerate(futures[i]):
            rows.append({"period": d, **({group_col: key} if group_col else {}),
                         "value": round(float(point[i][j]), 2),
                         "low": round(float(low[i][j]), 2), "high": round(float(high[i][j]), 2),
                         "kind": "forecast"})
    return pd.DataFrame(rows)
