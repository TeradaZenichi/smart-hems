"""Data access for the 15-min scenario CSVs in data/csvs.

Keeps every dataset concern (scenario splits, balanced-case listing, PV
scaling, battery-only masking, normalization bounds) out of the simulator:
SmartHomeEnv consumes the DataFrame returned by ``load_simulation_csv``
unchanged.

Climate-transfer split (all tariffs present in every subset). The forecaster
and the SAC train on disjoint scenario-years, so the forecast errors the SAC
sees are always out-of-sample for the forecaster:

    forecast_train  REF, TDYC       (typical current years)
    sac_train       CC, WC          (extreme current years, minus val weeks)
    validation      12 held-out weeks per sac_train scenario (1 per month)
    test            CF, TDYF, WF    (future climate, untouched by both)
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(HERE, "data", "csvs")

SPLITS = {
    "forecast_train": ["REF", "TDYC"],
    "sac_train": ["CC", "WC"],
    "test": ["CF", "TDYF", "WF"],
}
SCENARIOS = [s for split in SPLITS.values() for s in split]

EPISODE_DAYS = 7
HISTORY_DAYS = 1  # encoder look-back (L=96 at 15 min = 24 h)

WEATHER_COLUMNS = [
    "drybulb_C", "relhum_percent", "Global Horizontal Radiation",
    "dni_Wm2", "dhi_Wm2", "Wind Speed (m/s)", "wdir_deg",
]

_NAME_RE = re.compile(r"Simulation_(\w+)_RAD_(.+)\.csv$")


def list_cases(csv_dir=CSV_DIR, balanced_only=True):
    """Map case id (household_orientation_..._layout) -> {scenario: path}.

    With ``balanced_only`` (default) only cases present in all scenarios are
    returned, dropping the six REF-only extras so the design stays balanced.
    """
    cases = {}
    for path in glob.glob(os.path.join(csv_dir, "Simulation_*.csv")):
        m = _NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        scenario, case = m.groups()
        cases.setdefault(case, {})[scenario] = path
    if balanced_only:
        cases = {c: v for c, v in cases.items() if len(v) == len(SCENARIOS)}
    return cases


def split_files(split, csv_dir=CSV_DIR, balanced_only=True):
    """All (case, scenario, path) entries belonging to a split."""
    return [
        (case, scenario, paths[scenario])
        for case, paths in sorted(list_cases(csv_dir, balanced_only).items())
        for scenario in SPLITS[split]
    ]


def load_simulation_csv(path, pv_scale=1.0, battery_only=False):
    """Load one scenario CSV ready for SmartHomeEnv.

    pv_scale multiplies the PV1000 (1 kW reference) generation column, e.g.
    2.5 / 5.0 / 7.5 for the system sizes considered. battery_only zeroes the
    EV schedule, which makes EVEnv inert (P = 0, no costs) without any
    environment change.
    """
    df = pd.read_csv(path, sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M")
    df = df.set_index("timestamp")
    df["produced_electricity_rate_W"] = df["produced_electricity_rate_W"] * pv_scale
    if battery_only:
        df[["ev_status", "ev_conn", "ev_arrival", "ev_departure"]] = 0
    return df


def validation_weeks(index):
    """The 12 held-out validation weeks of a sac_train year: one calendar week
    per month, starting on the first Monday on or after the 8th. Deterministic
    by construction so the split is reproducible and versioned with the code.

    Returns a list of (start, end) timestamps, end exclusive.
    """
    year = index[0].year
    weeks = []
    for month in range(1, 13):
        d = pd.Timestamp(year=year, month=month, day=8)
        d = d + pd.Timedelta(days=(7 - d.weekday()) % 7)  # first Monday >= 8th
        weeks.append((d, d + pd.Timedelta(days=7)))
    return weeks


def training_start_mask(index, episode_days=EPISODE_DAYS, history_days=HISTORY_DAYS):
    """Boolean mask over ``index``: True where a training episode may start.

    Only the episode span [s, s + episode_days) — where training reward is
    collected — must stay clear of the validation weeks. The encoder history
    is allowed to read into a validation week: it is conditioning input, not
    a training target (accepted mild exposure, documented in the protocol).
    Starts that would run past the end of the data are excluded.
    """
    mask = np.ones(len(index), dtype=bool)
    ep = pd.Timedelta(days=episode_days)
    hist = pd.Timedelta(days=history_days)
    for a, b in validation_weeks(index):
        mask &= ~((index > a - ep) & (index < b))
    mask &= index + ep <= index[-1] + (index[1] - index[0])
    mask &= index - hist >= index[0]
    return mask


def compute_state_normalization(csv_dir=CSV_DIR):
    """Weather min/max over the training scenarios only (resumo §22): every
    scenario that feeds either module (forecast_train + sac_train), never test.

    Weather is identical across cases within a scenario, so one file per
    training scenario suffices.
    """
    cases = list_cases(csv_dir)
    first = next(iter(sorted(cases)))
    bounds = None
    for scenario in SPLITS["forecast_train"] + SPLITS["sac_train"]:
        w = pd.read_csv(cases[first][scenario], sep=";", usecols=WEATHER_COLUMNS)
        lo, hi = w.min(), w.max()
        if bounds is None:
            bounds = (lo, hi)
        else:
            bounds = (pd.concat([bounds[0], lo], axis=1).min(axis=1),
                      pd.concat([bounds[1], hi], axis=1).max(axis=1))
    return {
        c: {"min": round(float(bounds[0][c]), 2), "max": round(float(bounds[1][c]), 2)}
        for c in WEATHER_COLUMNS
    }


def build_parameters(reference_json, out_json, csv_dir=CSV_DIR):
    """Write the 15-min parameters file: reference params with timestep=15
    and state normalization recomputed from the training scenarios."""
    with open(reference_json, encoding="utf-8") as f:
        params = json.load(f)
    params["general"]["timestep"] = 15
    params["general"]["tariff_horizon_h"] = 24  # day-ahead tariff window in obs
    params["general"]["state_normalization"] = compute_state_normalization(csv_dir)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    return params


if __name__ == "__main__":
    ref = os.path.join(HERE, "data", "reference", "parameters.json")
    out = os.path.join(CSV_DIR, "parameters.json")
    params = build_parameters(ref, out)
    print(f"wrote {out}")
    print(json.dumps(params["general"]["state_normalization"], indent=2))
    cases = list_cases()
    print(f"{len(cases)} balanced cases x {len(SCENARIOS)} scenarios")
