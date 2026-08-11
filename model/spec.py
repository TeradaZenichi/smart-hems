# -*- coding: utf-8 -*-
"""Forecaster I/O contract — architecture-agnostic.

Every forecast network (GRU, future MLP/TCN...) and the data pipeline
(forecasting/dataset.py, forecasting/predictor.py) share these shapes. They
belong to no single architecture (an MLP would reuse the same contract), so
they live here, not inside model/{architecture}/.

Two JSON inputs, and the rule for which value goes where:
  - model/spec.json ............ free choices of the contract (how much
                                 history, what horizon, which weather columns).
  - data/csvs/parameters.json .. temporal resolution (general.timestep).

Everything else this module exposes is DERIVED from those at import time,
never a literal. Numbers like 672/96 can't go into JSON: they are results of a
computation (e.g. 672 = 7 x 24 x 60/timestep), and storing them as
constants would desync them from the source if the timestep ever changed
(the repo already has data/generated_5min/ next to data/csvs/).

Shape conventions used across the project (B = batch):
    hist:    (B, 672, 4)   7 days of [demand, pv, temperature, radiation]
    static:  (B, 18)       calendar + family + house/PV configuration
    target:  (B, 96, 2)    next 24 h of [demand, pv], normalized

Realized future weather is deliberately absent: the CSVs contain observations,
not archived weather forecasts. It may later be added as an explicit oracle or
as a separate experiment with genuine forecast data.
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ----- free contract choices (model/spec.json) -----------------------------
with open(os.path.join(_HERE, "spec.json"), encoding="utf-8") as _f:
    _SPEC = json.load(_f)
HIST_DAYS = _SPEC["hist_days"]          # how far back to look
HORIZON_HOURS = _SPEC["horizon_hours"]  # how far ahead to predict
TARGETS = _SPEC["targets"]              # predicted series: demand, pv
HIST_WEATHER = _SPEC["hist_weather"]    # observed weather entering the GRU
FAMILIES = _SPEC["families"]
HOUSE_ORIENTATIONS = _SPEC["house_orientations"]
PV_ORIENTATIONS = _SPEC["pv_orientations"]

# ----- temporal resolution (data/csvs/parameters.json) ---------------------
# Sole place that knows the data timestep; every step/hour below derives from it.
with open(os.path.join(_ROOT, "data", "csvs", "parameters.json"), encoding="utf-8") as _f:
    _TIMESTEP_MIN = json.load(_f)["general"]["timestep"]  # minutes per step
STEPS_PER_HOUR = 60 // _TIMESTEP_MIN

# ----- derived (never literals; never in JSON) -----------------------------
HIST_STEPS = HIST_DAYS * 24 * STEPS_PER_HOUR
HORIZON = HORIZON_HOURS * STEPS_PER_HOUR
N_TARGETS = len(TARGETS)
HIST_DIM = N_TARGETS + len(HIST_WEATHER)

# N_CALENDAR is not a knob: it mirrors forecasting.dataset.calendar_features,
# which emits exactly these 6 values (sin/cos of hour, weekday, day-of-year).
# Changing it here without changing there would break STATIC_DIM, so it stays
# in code next to the computation, not in JSON.
N_CALENDAR = 6
META_DIM = (len(FAMILIES) + len(HOUSE_ORIENTATIONS)
            + len(PV_ORIENTATIONS) + 1)  # + normalized PV tilt
STATIC_DIM = N_CALENDAR + META_DIM
