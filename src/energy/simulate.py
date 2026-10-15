"""Simulated Dubai distribution network: weather, smart-meter consumption and rooftop solar.

Weather: hourly temperature and humidity with the UAE's hot summer (45 C+ afternoons), mild winter,
and occasional dust events that cut solar output. A day-ahead weather forecast is the actual weather plus
forecast error, because a real day-ahead model only has the forecast.
Meters: villas, apartments, commercial and industrial customers. Air-conditioning drives load
(cooling degree hours), with Ramadan shifting activity into the night and weekends changing the daily shape.
A share of meters are tampered with partway through the year (bypass, partial bypass, meter stopped).
Solar: Shams Dubai-style rooftop systems; output follows irradiance, panel heating losses and dust soiling.
All values are illustrative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAMADAN = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30")]
CUSTOMER_TYPES = {"villa": (0.35, 2.2), "apartment": (0.45, 0.9), "commercial": (0.15, 6.0), "industrial": (0.05, 18.0)}  # share, base kW


def weather(start="2024-01-01", end="2025-12-31 23:00", seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="h")
    doy, hour = t.dayofyear.to_numpy(), t.hour.to_numpy()
    seasonal = 28.5 - 8.5 * np.cos(2 * np.pi * (doy - 20) / 365.25)          # ~20 C in Jan, ~37 C mean in Aug
    daily = 6.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather_noise = np.repeat(rng.normal(0, 1.6, len(t) // 24 + 1), 24)[:len(t)]
    temp = seasonal + daily + weather_noise + rng.normal(0, 0.6, len(t))
    humidity = np.clip(55 + 15 * np.cos(2 * np.pi * (hour - 4) / 24) + 10 * np.sin(2 * np.pi * (doy - 200) / 365.25) + rng.normal(0, 6, len(t)), 10, 98)
    # dust events: a few days each spring and summer with heavy haze
    dust = np.zeros(len(t))
    for _ in range(40):
        s = rng.integers(0, len(t) - 72)
        if t[s].month in (3, 4, 5, 6, 7, 8, 9):
            dust[s:s + rng.integers(24, 72)] = rng.uniform(0.3, 0.7)
    sun = np.clip(np.sin(np.pi * (hour - 6) / 13), 0, None) * (0.85 + 0.15 * np.sin(2 * np.pi * (doy - 80) / 365.25))
    cloud = np.where(rng.random(len(t) // 24 + 1) < 0.06, rng.uniform(0.2, 0.6), 0.0).repeat(24)[:len(t)]
    ghi = 1000 * sun * (1 - dust) * (1 - cloud)                               # W/m2
    w = pd.DataFrame({"time": t, "temp": temp, "humidity": humidity, "dust": dust, "cloud": cloud, "ghi": ghi, "clear_sky_ghi": 1000 * sun})
    # day-ahead forecast = actual + error (larger for dust and cloud, which are harder to predict)
    w["temp_fc"] = temp + rng.normal(0, 1.2, len(t))
    w["dust_fc"] = np.clip(dust + rng.normal(0, 0.12, len(t)) * (dust > 0) - 0.1 * (rng.random(len(t)) < 0.2) * dust, 0, 1)
    w["cloud_fc"] = np.clip(cloud + rng.normal(0, 0.1, len(t)) * (cloud > 0), 0, 1)
    w["ghi_fc"] = 1000 * sun * (1 - w["dust_fc"]) * (1 - w["cloud_fc"])
    w["ramadan"] = 0
    for a, b in RAMADAN:
        w.loc[(w["time"] >= a) & (w["time"] < pd.Timestamp(b) + pd.Timedelta(days=1)), "ramadan"] = 1
    return w


def meters(w: pd.DataFrame, n: int = 400, seed: int = 2, tamper_share: float = 0.06) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (hourly aggregate network load, daily per-meter consumption with tampering labels)."""
    rng = np.random.default_rng(seed)
    types = rng.choice(list(CUSTOMER_TYPES), n, p=[v[0] for v in CUSTOMER_TYPES.values()])
    base = np.array([CUSTOMER_TYPES[x][1] for x in types]) * rng.lognormal(0, 0.3, n)
    cooling_sens = base * rng.uniform(0.05, 0.09, n)                          # kW per degree above 24 C
    hour = w["time"].dt.hour.to_numpy()
    dow = w["time"].dt.dayofweek.to_numpy()
    weekend = dow >= 5
    cdh = np.clip(w["temp"].to_numpy() - 24, 0, None)
    ram = w["ramadan"].to_numpy()

    shape = {
        "villa": np.where((hour >= 17) | (hour <= 1), 1.35, np.where((hour >= 8) & (hour <= 14), 0.75, 1.0)),
        "apartment": np.where((hour >= 18) | (hour <= 0), 1.4, np.where((hour >= 9) & (hour <= 16), 0.7, 0.95)),
        "commercial": np.where((hour >= 9) & (hour <= 22), 1.45, 0.45) * np.where(weekend, 1.1, 1.0),
        "industrial": np.where((hour >= 7) & (hour <= 18), 1.2, 0.85) * np.where(weekend, 0.6, 1.0),
    }
    ram_shift = np.where(ram == 1, np.where((hour >= 20) | (hour <= 3), 1.25, np.where((hour >= 9) & (hour <= 15), 0.85, 1.0)), 1.0)

    days = w["time"].dt.normalize().to_numpy()
    uniq_days, day_idx = np.unique(days, return_inverse=True)
    total = np.zeros(len(w))
    daily = np.zeros((n, len(uniq_days)))
    tamper_type = np.array([""] * n, dtype=object)
    tamper_start = np.full(n, -1)
    bad = rng.choice(n, int(n * tamper_share), replace=False)
    for i in bad:
        tamper_type[i] = rng.choice(["bypass", "partial_bypass", "stopped_meter"])
        tamper_start[i] = rng.integers(150, len(uniq_days) - 120)
    for i in range(n):
        kw = (base[i] * shape[types[i]] + cooling_sens[i] * cdh) * ram_shift * rng.lognormal(0, 0.12, len(w))
        total += kw                                                           # the network still carries the true load
        d = np.bincount(day_idx, weights=kw)
        if tamper_type[i]:
            s = tamper_start[i]
            factor = {"bypass": 0.15, "partial_bypass": 0.55, "stopped_meter": 0.0}[tamper_type[i]]
            d[s:] = d[s:] * factor + (rng.normal(0, 0.3, len(d) - s).clip(0) if factor == 0 else 0)
        daily[i] = d
    # legitimate look-alikes: vacation homes empty for weeks in summer, shops closing for renovation
    for k, i in enumerate(rng.choice([j for j in range(n) if not tamper_type[j]], int(n * 0.08), replace=False)):
        s = rng.integers(160, 600) if k % 2 else rng.integers(len(uniq_days) - 70, len(uniq_days) - 25)   # some overlap the review window
        daily[i, s:s + rng.integers(20, 50)] *= rng.uniform(0.15, 0.35)
    load = pd.DataFrame({"time": w["time"], "load_kw": total})
    meta = pd.DataFrame({"meter_id": [f"MTR{i:04d}" for i in range(n)], "type": types, "tamper": tamper_type,
                         "tamper_start_day": tamper_start})
    daily_df = pd.DataFrame(daily, columns=pd.DatetimeIndex(uniq_days))
    daily_df.insert(0, "meter_id", meta["meter_id"])
    return load, meta.merge(daily_df, on="meter_id")


def solar(w: pd.DataFrame, kwp: float = 5000.0, seed: int = 4) -> pd.Series:
    """Fleet of rooftop systems (total kWp). Output falls ~0.4%/C above 25 C cell temperature and with soiling."""
    rng = np.random.default_rng(seed)
    cell_t = w["temp"] + w["ghi"] / 800 * 25
    temp_loss = 1 - 0.004 * np.clip(cell_t - 25, 0, None)
    soiling = 1 - 0.0015 * (np.arange(len(w)) // 24 % 30)                     # panels cleaned monthly
    return kwp * w["ghi"] / 1000 * 0.82 * temp_loss * soiling * rng.normal(1, 0.02, len(w))
