"""Shared test helpers.

``create_mock_solar_wind_frame`` exists so that offline tests never need the
network and never need to hand raw numbers to a physics formula: they build a
real solar-wind frame, store it in a :class:`agent_core.FrameStore`, and pass
the returned handle exactly the way the agent does. The contract under test is
therefore the production one (a handle in, an aligned frame out), not a
test-only shortcut.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

_counter = itertools.count(1)


def make_solar_wind_frame(
    V: float = 450.0,
    Bz: float = -5.0,
    Np: float = 5.0,
    Bx: float = 0.0,
    By: float = 0.0,
    *,
    periods: int = 6,
    freq: str = "min",
    start: str = "2024-09-12 00:00:00",
) -> pd.DataFrame:
    """A solar-wind frame (constant by default) on a regular UTC clock."""
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": index,
            "V": [float(V)] * periods,
            "Bz": [float(Bz)] * periods,
            "Np": [float(Np)] * periods,
            "Bx": [float(Bx)] * periods,
            "By": [float(By)] * periods,
        }
    )


def create_mock_solar_wind_frame(
    V: float = 450.0,
    Bz: float = -5.0,
    Np: float = 5.0,
    Bx: float = 0.0,
    By: float = 0.0,
    *,
    periods: int = 6,
    freq: str = "min",
    start: str = "2024-09-12 00:00:00",
    store: Any = None,
) -> str:
    """Build a mock solar-wind frame, store it, and return its handle.

    ``store`` is an :class:`agent_core.FrameStore`; when omitted a fresh one is
    created, which is all a single test needs. The handle is unique per call
    (``sw:mock:1``, ``sw:mock:2``, ...) so several mocks can coexist in one store.
    """
    from agent_core import FrameStore

    if store is None:
        store = FrameStore()
    frame = make_solar_wind_frame(
        V=V, Bz=Bz, Np=Np, Bx=Bx, By=By, periods=periods, freq=freq, start=start
    )
    return store.put(f"sw:mock:{next(_counter)}", frame)
