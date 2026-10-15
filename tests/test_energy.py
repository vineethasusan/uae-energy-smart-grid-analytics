import numpy as np

from energy import models


def test_summer_load_driven_by_cooling(sim):
    _, df, _ = sim
    m = df.groupby(df["time"].dt.month)["load_kw"].mean()
    assert m[8] > 1.3 * m[1]
    assert df["solar_kw"].min() >= 0 and df.loc[df["time"].dt.hour == 0, "solar_kw"].max() < 1


def test_weather_aware_load_forecast_beats_last_week(sim):
    _, df, _ = sim
    train, test = df[df["time"] < "2025-07-01"], df[df["time"] >= "2025-07-01"]
    m = models.fit_load(train)
    X = models.load_features(df).loc[test.index]
    assert models.mape(test["load_kw"], m.predict(X)) < models.mape(test["load_kw"], X["lag_168"])


def test_solar_model_beats_persistence(sim):
    _, df, _ = sim
    train, test = df[df["time"] < "2025-07-01"], df[df["time"] >= "2025-07-01"]
    test = test[test["clear_sky_ghi"] > 50]
    m = models.fit_solar(train)
    err_model = np.mean(np.abs(m.predict(models.solar_features(test)) - test["solar_kw"]))
    err_persist = np.mean(np.abs(df["solar_kw"].shift(24).loc[test.index] - test["solar_kw"]))
    assert err_model < err_persist


def test_tamper_ranking_finds_tampered_meters(sim):
    w, _, daily = sim
    s = models.tamper_scores(daily, w)
    top = s.nlargest(10, "score")
    assert (top["tamper"] != "").mean() >= 0.5
    recovered = s[(s["tamper"] == "") & (s["recovered_after_dip"] == 1)]
    assert recovered["score"].median() < s.loc[s["tamper"] != "", "score"].median()
