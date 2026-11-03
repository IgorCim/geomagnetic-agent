"""Inspect current figure layout without changing the plotter or writing HTML."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import pandas as pd
    import geomag_plotter as gp

    before = hashlib.sha256((ROOT / "geomag_plotter.py").read_bytes()).hexdigest()
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-09-10", periods=4, freq="h", tz="UTC"),
                          "X": [20490., 20500., 20510., 20500.], "Y": [2850., 2860., 2870., 2860.],
                          "Z": [46900., 46901., 46902., 46901.], "H": [20690., 20700., 20710., 20700.],
                          "F": [51260., 51265., 51270., 51265.], "D": [7.9, 7.91, 7.92, 7.91],
                          "I": [66.1, 66.2, 66.3, 66.2], "Dst": [0., 5., 10., 5.], "Kp": [1., 2., 3., 2.],
                          "Bz": [-1., -2., -3., -2.], "V": [400., 410., 420., 410.]})
    cases = []
    def capture(fig, path, *args):
        refs = [trace.yaxis or "y" for trace in fig.data]
        cases.append({"traces": [trace.name for trace in fig.data], "y_axis_references": refs,
                      "independent_y_axes": len(set(refs))})
        return str(path)
    with patch.object(gp, "_write", capture):
        for components in (["X", "Y", "Z"], ["H", "F"], ["H", "D", "I"], ["D", "I"], ["Dst", "Kp"]):
            gp.plot_components(frame, components=components)
            cases[-1]["call"] = f"plot_components({components!r})"
        gp.plot_space_weather_comparison(frame, ground_component="H", solar_wind_components=["Bz", "V"])
        cases[-1]["call"] = "plot_space_weather_comparison(H, Bz/V)"
    after = hashlib.sha256((ROOT / "geomag_plotter.py").read_bytes()).hexdigest()
    assert before == after
    report = {"scope": "Synthetic figure-object inspection; no model, no remote data, no HTML rendered or written",
              "plotter_sha256": before, "plotter_unchanged": True, "cases": cases}
    output = ROOT / "verification" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_plot_layout")
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Evidence:", output)


if __name__ == "__main__":
    main()
