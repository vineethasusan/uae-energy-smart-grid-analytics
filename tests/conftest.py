import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def sim():
    from energy import simulate
    w = simulate.weather()
    load, daily = simulate.meters(w, n=200, seed=5)
    df = w.merge(load, on="time")
    df["solar_kw"] = simulate.solar(w).to_numpy()
    return w, df, daily
