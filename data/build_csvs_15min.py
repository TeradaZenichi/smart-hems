"""Build 15-min Simulation_*.csv files in data/csvs from the raw data in data/new.

Same union as build_reference_csvs.py (RAD demand/PV + EPW weather + tariffs),
but at the RAD files' native 15-min resolution (35040 rows, 365-day year 2000)
and with the tariff columns replaced:

  tar_flat  0.17 EUR/kWh constant (price_factor 1 at the 0.17 base price)
  tar_tou   BEL_REF_*_tou.csv hourly price, repeated 4x; calendar-only tariff
            (peak 0.2266 Mon-Fri 07-22h except BE holidays, off-peak 0.1259),
            identical for every weather scenario
  tar_rtp   the scenario's *_weather_driven.csv hourly price, repeated 4x

Demand/PV are taken as-is (EnergyPlus end-of-interval labels realigned to
interval start). Weather is the scenario EPW stretched onto the 15-min grid
with the same np.interp scheme validated against the 5-min reference files.
EV columns use data/EV/ev_schedule_current_15min.csv in every scenario, so
climate is the only distribution shift in the main test. The high-demand EV
profile remains separate and is applied by environment.dataset only in the
mobility-generalization test.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, "new")
OUT = os.path.join(HERE, "csvs")

SCENARIO_DIR = {
    "REF": "reference",
    "CC": os.path.join("current", "cold"),
    "WC": os.path.join("current", "warm"),
    "CF": os.path.join("future", "cold"),
    "WF": os.path.join("future", "warm"),
    "TDYC": os.path.join("typical", "current"),
    "TDYF": os.path.join("typical", "future"),
}

N_15MIN = 365 * 24 * 4          # 35040
N_HOUR = 365 * 24               # 8760

BASE_PRICE = 0.17               # EUR/kWh, "vast" tariff = price_factor 1

EPW_FIELDS = {
    6: "drybulb_C",
    8: "relhum_percent",
    13: "Global Horizontal Radiation",
    14: "dni_Wm2",
    15: "dhi_Wm2",
    21: "Wind Speed (m/s)",
    20: "wdir_deg",
}

SCENARIO_BASENAMES = {
    "REF": "BEL_REF_Brussels.064510_IWEC 1(1)",
    "CC": "Extreme_CC_Year_Uccle(1)",
    "CF": "Extreme_CF_Year_Uccle(1)",
    "WC": "Extreme_WC_Year_Uccle(1)",
    "WF": "Extreme_WF_Year_Uccle(1)",
    "TDYC": "TDYC_Uccle_Current(1)",
    "TDYF": "TDYF_Uccle_Future(1)",
}

WD_DIR = os.path.join(
    NEW, "wetransfer_extreme_cc_year_uccle-1-_weather_driven-1-csv_2026-06-27_0744 (1)"
)
TOU_PATH = os.path.join(NEW, "BEL_REF_Brussels.064510_IWEC 1(1)_tou.csv")

COLUMNS = [
    "timestamp", "electricity_demand_rate_W", "produced_electricity_rate_W",
    "drybulb_C", "relhum_percent", "Global Horizontal Radiation", "dni_Wm2",
    "dhi_Wm2", "Wind Speed (m/s)", "wdir_deg", "ev_status", "tar_flat",
    "tar_tou", "tar_rtp", "ev_conn", "ev_arrival", "ev_departure",
]

FAMILY_NAMES = {
    "2jong": "young-couple",
    "2oud": "older-couple",
    "2vol2kids": "family-with-children",
}
HOUSE_NAMES = {
    "noord": "north",
    "oost": "east",
    "west": "west",
    "zuid": "south",
}
PV_NAMES = {
    "O": "east",
    "OW": "east-west",
    "W": "west",
    "Z": "south",
}


def hourly_price_15min(path):
    wd = pd.read_csv(path)
    assert len(wd) >= N_HOUR, f"{path}: {len(wd)} rows"
    price = wd["price_eur_per_kWh_at_base_price"].to_numpy(dtype=float)[:N_HOUR]
    return np.repeat(price, 4)


def load_epw_15min(scenario):
    path = os.path.join(NEW, SCENARIO_BASENAMES[scenario] + ".epw")
    raw = pd.read_csv(path, skiprows=8, header=None)
    # WC is a spliced 366-day file (8784 h); keep the first 8760 h
    assert len(raw) >= N_HOUR, f"{path}: {len(raw)} rows"
    raw = raw.iloc[:N_HOUR]
    anchors = np.linspace(0, N_15MIN - 1, N_HOUR)
    grid = np.arange(N_15MIN)
    return {
        col: np.interp(grid, anchors, raw[idx].to_numpy(dtype=float))
        for idx, col in EPW_FIELDS.items()
    }


def collect_rad_files():
    """Map output name -> RAD csv path; the 06-27 batch wins over 06-26 duplicates."""
    folders = [  # oldest first so newer overwrite
        os.path.join(NEW, "wetransfer_braziliaans-mapje_2026-06-26_1740 (1)"),
        os.path.join(NEW, "wetransfer_braziliaans-mapje_2026-06-27_0933 (1)"),
    ]
    files = {}
    for folder in folders:
        for path in glob.glob(os.path.join(folder, "**", "RAD_*.csv"), recursive=True):
            files[os.path.basename(path)] = path
    return files


def parse_rad_name(name):
    m = re.match(r"RAD_(.+)_\((\w+)\)_(.+)\.csv$", name)
    if not m:
        raise ValueError(f"unexpected RAD file name: {name}")
    building, scenario, layout = m.groups()
    family_raw, house_raw, envelope_raw = building.split("_", maxsplit=2)
    pv_raw, tilt = layout.split("-tilt(", maxsplit=1)
    family = FAMILY_NAMES[family_raw]
    house = HOUSE_NAMES[house_raw]
    envelope = "poor" if envelope_raw == "SLECHT" else envelope_raw.lower()
    pv = PV_NAMES[pv_raw]
    case = f"{family}_{house}_{envelope}_{pv}-tilt({tilt}"
    return scenario, f"Simulation_{scenario}_RAD_{case}.csv"


def main():
    os.makedirs(OUT, exist_ok=True)
    ev = pd.read_csv(os.path.join(HERE, "EV", "ev_schedule_current_15min.csv"))
    # 2001 starts on a Monday and is not a leap year, so the timestamps' weekday
    # matches the TOU/EV calendar (day 0 = Monday) and the year ends on 31/12
    timestamps = pd.date_range("2001-01-01", periods=N_15MIN, freq="15min").strftime(
        "%d/%m/%Y %H:%M"
    )
    tar_flat = np.full(N_15MIN, BASE_PRICE)
    tar_tou = hourly_price_15min(TOU_PATH)

    weather_cache, rtp_cache = {}, {}
    rad_files = collect_rad_files()
    print(f"{len(rad_files)} RAD files found")

    for i, (name, path) in enumerate(sorted(rad_files.items()), 1):
        scenario, out_name = parse_rad_name(name)
        if scenario not in weather_cache:
            weather_cache[scenario] = load_epw_15min(scenario)
            rtp_cache[scenario] = hourly_price_15min(
                os.path.join(WD_DIR, SCENARIO_BASENAMES[scenario] + "_weather_driven (1).csv")
            )

        rad = pd.read_csv(path)
        assert len(rad) == N_15MIN, f"{path}: {len(rad)} rows"
        df = pd.DataFrame({
            "timestamp": timestamps,
            "electricity_demand_rate_W": rad.iloc[:, 1].to_numpy(dtype=float),
            "produced_electricity_rate_W": rad.iloc[:, 2].to_numpy(dtype=float),
            **weather_cache[scenario],
            "ev_status": ev["ev_status"].to_numpy(),
            "tar_flat": tar_flat,
            "tar_tou": tar_tou,
            "tar_rtp": rtp_cache[scenario],
            "ev_conn": ev["ev_conn"].to_numpy(),
            "ev_arrival": ev["ev_arrival"].to_numpy(),
            "ev_departure": ev["ev_departure"].to_numpy(),
        })[COLUMNS]
        out_dir = os.path.join(OUT, SCENARIO_DIR[scenario])
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, out_name), sep=";", index=False)
        print(f"[{i}/{len(rad_files)}] {out_name}")


if __name__ == "__main__":
    main()
