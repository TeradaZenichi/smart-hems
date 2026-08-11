# -*- coding: utf-8 -*-
"""Supervised windows for the forecaster, honoring the project's data split.

A sample is a forecast ORIGIN t in one series (case x scenario):
    hist:    [t-672, t)  demand, pv, temperature, radiation (672, 4)
    static:  calendar + family + house/PV configuration     (18,)
    target:  [t, t+96)   demand & pv, normalized            (96, 2)

Weather channels are observations from the historical window. Future weather
from the CSV is not exposed because it is realized weather, not an archived
forecast available at origin t.

Split rules (already agreed, implemented here so they cannot drift):
  - train origins: hourly stride; the TARGET may not touch a validation
    week (relaxed rule — history may read into one, it is only context).
  - val origins:   4/day (00,06,12,18h); target fully inside a val week.
"""

import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from environment.dataset import CSV_DIR, SPLITS, list_cases, validation_weeks
from model.spec import (
    FAMILIES,
    HIST_STEPS,
    HIST_WEATHER,
    HORIZON,
    HOUSE_ORIENTATIONS,
    PV_ORIENTATIONS,
    TARGETS,
)

PARAMS_PATH = os.path.join(CSV_DIR, "parameters.json")
VAL_HOURS = (0, 6, 12, 18)
TRAIN_STRIDE = 4  # hourly origins on the 15-min grid


class Normalizer:
    """Normalization shared by forecaster and (later) feature caches.

    Power (demand/pv): W -> kW -> /Pnorm      (same convention as the env)
    Weather:           min-max from parameters.json (fit on train scenarios)
    """

    def __init__(self, params_path: str = PARAMS_PATH):
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
        self.pnorm = float(params["general"]["Pnorm"])
        self.weather_bounds = params["general"]["state_normalization"]

    def transform_power(self, watts: torch.Tensor) -> torch.Tensor:
        return watts / 1000.0 / self.pnorm

    def inverse_power(self, normalized: torch.Tensor) -> torch.Tensor:
        """Back to kW (for reporting MAE in physical units), not W."""
        return normalized * self.pnorm

    def transform_weather(self, values: torch.Tensor, column: str) -> torch.Tensor:
        bounds = self.weather_bounds[column]
        scaled = (values - bounds["min"]) / (bounds["max"] - bounds["min"])
        return torch.clamp(scaled, min=0.0, max=1.0)


def _col(df: pd.DataFrame, name: str) -> torch.Tensor:
    # .copy() avoids PyTorch's non-writable-tensor warning (pandas array is read-only)
    return torch.from_numpy(df[name].to_numpy().copy()).float()


def calendar_features(ts: pd.Timestamp) -> torch.Tensor:
    """Origin timestamp as 6 cyclic features: hour-of-day, weekday, day-of-year,
    each a sin/cos pair (so 23h and 0h sit next to each other on the circle)."""
    hour = ts.hour + ts.minute / 60.0
    dow = ts.weekday()
    doy = ts.dayofyear
    return torch.tensor([
        math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * dow / 7.0), math.cos(2 * math.pi * dow / 7.0),
        math.sin(2 * math.pi * doy / 365.0), math.cos(2 * math.pi * doy / 365.0),
    ], dtype=torch.float32)


def _one_hot(value: str, choices: list[str]) -> torch.Tensor:
    """One-hot with a fixed order defined in model/spec.json."""
    if value not in choices:
        raise ValueError(f"unknown category {value!r}; expected one of {choices}")
    out = torch.zeros(len(choices), dtype=torch.float32)
    out[choices.index(value)] = 1.0
    return out


def case_features(case: str) -> torch.Tensor:
    """Family and house/PV configuration encoded as 12 model features.

    Case format:
        family_house_envelope_pv-tilt(angle)_capacity

    The envelope and PV capacity are omitted because the balanced dataset has
    only poor and PV1000. Hyphens in filenames become underscores in model
    categories. Tilt is divided by 90 to keep it in [0, 1].
    """
    try:
        family_text, house, _, pv_tilt, _ = case.split("_")
        pv_text, tilt_text = pv_tilt.split("-tilt(")
        tilt = float(tilt_text.removesuffix(")"))
        family = family_text.replace("-", "_")
        pv = pv_text.replace("-", "_")
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid case id: {case!r}") from exc

    return torch.cat([
        _one_hot(family, FAMILIES),
        _one_hot(house, HOUSE_ORIENTATIONS),
        _one_hot(pv, PV_ORIENTATIONS),
        torch.tensor([tilt / 90.0], dtype=torch.float32),
    ])


class ForecastWindows(Dataset):
    """All windows of a split ('train' | 'val') over a list of scenarios
    (default: forecast_train).

    Series are preloaded into RAM tensors (~20 MB); __getitem__ only slices,
    no I/O, which is what keeps the GPU fed.
    """

    def __init__(self, split: str, scenarios=None, csv_dir: str = CSV_DIR,
                 params_path: str = PARAMS_PATH):
        assert split in ("train", "val")
        self.split = split
        self.norm = Normalizer(params_path)
        scenarios = scenarios or SPLITS["forecast_train"]

        self.series = []       # one dict per series: tensors + timestamps
        self.origins = []      # global list of (series_index, origin)
        for case, paths in sorted(list_cases(csv_dir).items()):
            for scenario in scenarios:
                df = pd.read_csv(paths[scenario], sep=";")
                index = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M")
                power = torch.stack([
                    self.norm.transform_power(_col(df, c)) for c in TARGETS
                ], dim=1)                                        # (35040, 2)
                weather = torch.stack([
                    self.norm.transform_weather(_col(df, c), c) for c in HIST_WEATHER
                ], dim=1)                                        # (35040, 2)
                s = len(self.series)
                self.series.append({"power": power, "weather": weather,
                                    "index": pd.DatetimeIndex(index),
                                    "case": case, "scenario": scenario})
                self.origins += [(s, int(o)) for o in
                                 self._origins(pd.DatetimeIndex(index))]

    def _origins(self, index: pd.DatetimeIndex) -> np.ndarray:
        n = len(index)
        weeks = validation_weeks(index)
        step = index[1] - index[0]
        pos = {ts: i for i, ts in enumerate(index)}
        if self.split == "val":
            out = []
            for a, b in weeks:
                t = a
                while t + HORIZON * step <= b:
                    if t.hour in VAL_HOURS and t.minute == 0 and pos[t] >= HIST_STEPS:
                        out.append(pos[t])
                    t += step
            return np.array(sorted(out))
        ok = np.zeros(n, dtype=bool)
        ok[HIST_STEPS:n - HORIZON + 1:TRAIN_STRIDE] = True
        for a, b in weeks:                       # target must not touch a val week
            ia, ib = pos.get(a), pos.get(b - step)
            lo = max(0, (ia if ia is not None else 0) - HORIZON + 1)
            hi = (ib + 1) if ib is not None else n
            ok[lo:hi] = False
        return np.flatnonzero(ok)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, k: int):
        s, i = self.origins[k]
        serie = self.series[s]
        power_hist = serie["power"][i - HIST_STEPS:i]
        weather_hist = serie["weather"][i - HIST_STEPS:i]
        hist = torch.cat([power_hist, weather_hist], dim=1)
        target = serie["power"][i:i + HORIZON]
        static = torch.cat([
            calendar_features(serie["index"][i]),
            case_features(serie["case"]),
        ])
        return hist, static, target
