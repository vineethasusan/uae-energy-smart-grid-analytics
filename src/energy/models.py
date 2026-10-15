"""Day-ahead load and solar forecasting, and revenue-protection (meter tampering) detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y) / y))


# ---------- day-ahead network load ----------

def load_features(df: pd.DataFrame, weather_cols=True) -> pd.DataFrame:
    t = df["time"]
    X = pd.DataFrame({"hour": t.dt.hour, "dow": t.dt.dayofweek, "month": t.dt.month, "ramadan": df["ramadan"],
                      "lag_24": df["load_kw"].shift(24), "lag_168": df["load_kw"].shift(168),
                      "mean_prev_day": df["load_kw"].shift(24).rolling(24).mean()}, index=df.index)
    if weather_cols:
        X["temp_fc"] = df["temp_fc"]
        X["cdh_fc"] = np.clip(df["temp_fc"] - 24, 0, None)
        X["temp_change"] = df["temp_fc"] - df["temp"].shift(24)        # tomorrow's forecast vs today's actual
        X["humidity"] = df["humidity"].shift(24)
    return X


def fit_load(train: pd.DataFrame, weather_cols=True):
    X = load_features(train, weather_cols).dropna()
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0).fit(X, train.loc[X.index, "load_kw"])


# ---------- day-ahead solar ----------

def solar_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["time"]
    return pd.DataFrame({"hour": t.dt.hour, "doy": t.dt.dayofyear, "clear_sky_ghi": df["clear_sky_ghi"], "ghi_fc": df["ghi_fc"],
                         "dust_fc": df["dust_fc"], "cloud_fc": df["cloud_fc"], "temp_fc": df["temp_fc"]}, index=df.index)


def fit_solar(train: pd.DataFrame):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(solar_features(train), train["solar_kw"])


# ---------- meter tampering ----------

def tamper_scores(daily: pd.DataFrame, w: pd.DataFrame, baseline_days: int = 120, recent_days: int = 60) -> pd.DataFrame:
    """For each meter: learn its normal consumption vs cooling demand on the first months, then measure how far
    the last `recent_days` fall short of that expectation and whether the shortfall is sustained (tampering)
    or temporary (vacation, renovation). Unsupervised: no labelled theft cases are needed."""
    days = [c for c in daily.columns if isinstance(c, pd.Timestamp)]
    cdd = w.assign(d=w["time"].dt.normalize()).groupby("d")["temp"].apply(lambda s: np.clip(s - 24, 0, None).sum()).reindex(days).to_numpy()
    X = np.column_stack([cdd, pd.DatetimeIndex(days).dayofweek >= 5])
    out = []
    for _, r in daily.iterrows():
        y = r[days].to_numpy(float)
        m = LinearRegression().fit(X[:baseline_days], y[:baseline_days])
        exp = np.clip(m.predict(X), 0.05, None)
        ratio = y / exp
        recent = ratio[-recent_days:]
        weekly = pd.Series(ratio).rolling(14).median().to_numpy()
        drops = np.diff(weekly[~np.isnan(weekly)][::7])
        out.append({"meter_id": r["meter_id"], "type": r["type"], "recent_ratio": float(np.median(recent)),
                    "share_days_below_50pct": float(np.mean(recent < 0.5)), "zero_days": int(np.sum(y[-recent_days:] < 0.5)),
                    "largest_step_drop": float(-drops.min()) if len(drops) else 0.0,
                    "recovered_after_dip": float(np.median(ratio[-21:]) > 0.8 and np.min(weekly[-120:]) < 0.5),
                    "kwh_shortfall_recent": float(np.sum(np.clip(exp[-recent_days:] - y[-recent_days:], 0, None)))})
    s = pd.DataFrame(out)
    # sustained, large shortfall ranks highest; recovered dips are discounted
    s["score"] = (1 - s["recent_ratio"].clip(0, 1)) * 0.5 + s["share_days_below_50pct"] * 0.3 + s["largest_step_drop"].clip(0, 1) * 0.2
    s.loc[s["recovered_after_dip"] == 1, "score"] *= 0.3
    return s.merge(daily[["meter_id", "tamper"]], on="meter_id")


def yoy_rule(daily: pd.DataFrame) -> pd.Series:
    """Common utility rule: flag if the last 60 days are less than half of the same 60 days a year earlier."""
    days = [c for c in daily.columns if isinstance(c, pd.Timestamp)]
    cur = daily[days[-60:]].sum(axis=1)
    prev = daily[days[-425:-365]].sum(axis=1)
    return 1 - (cur / prev).clip(0, 2)
