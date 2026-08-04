"""Build Simulation_*.csv files (reference format) from the raw data in data/new.

Combines, per weather scenario:
  - EnergyPlus RAD_*.csv        -> demand / PV production (15-min, end-of-interval
                                   timestamps; each value covers the preceding 15 min)
  - <scenario>.epw              -> weather columns (hourly)
  - <scenario>_weather_driven.csv -> dynamic tariff (hourly, EUR/kWh)

Output matches the existing reference contract:
  105120 rows (365-day year at 5-min steps, timestamps in year 2000),
  ';'-separated, same 19 columns as Simulation_CY_Cur_HP__PV5000-HB5000.csv.

Column construction:
  - demand/PV: each 15-min value repeated 3x (covers its interval)
  - weather:   8760 hourly EPW values stretched over the 105120 grid with
               np.interp anchors at linspace(0, N-1, 8760) — this reproduces the
               original reference weather columns to numerical precision
  - tar_s/tar_w/tar_sw: all three filled with the weather-driven hourly price
               (single dynamic tariff in the new scheme), repeated 12x
  - tar_flat/tar_tou/ev_*: copied from Simulation_CY_Cur (identical schedule
               across reference files)
"""

import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, "new")
OUT = os.path.join(HERE, "generated_5min")

N_5MIN = 365 * 24 * 12          # 105120
N_15MIN = 365 * 24 * 4          # 35040
N_HOUR = 365 * 24               # 8760

# EPW data fields (0-indexed) -> reference column names
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

COLUMNS = [
    "timestamp", "electricity_demand_rate_W", "produced_electricity_rate_W",
    "drybulb_C", "relhum_percent", "Global Horizontal Radiation", "dni_Wm2",
    "dhi_Wm2", "Wind Speed (m/s)", "wdir_deg", "ev_status", "tar_s", "tar_w",
    "tar_sw", "tar_flat", "tar_tou", "ev_conn", "ev_arrival", "ev_departure",
]


def load_epw_5min(scenario):
    path = os.path.join(NEW, SCENARIO_BASENAMES[scenario] + ".epw")
    raw = pd.read_csv(path, skiprows=8, header=None)
    # WC is a spliced 366-day file (8784 h); the reference kept the first 8760 h
    assert len(raw) >= N_HOUR, f"{path}: {len(raw)} rows"
    raw = raw.iloc[:N_HOUR]
    anchors = np.linspace(0, N_5MIN - 1, N_HOUR)
    grid = np.arange(N_5MIN)
    return {
        col: np.interp(grid, anchors, raw[idx].to_numpy(dtype=float))
        for idx, col in EPW_FIELDS.items()
    }


def load_tariff_5min(scenario):
    path = os.path.join(WD_DIR, SCENARIO_BASENAMES[scenario] + "_weather_driven (1).csv")
    wd = pd.read_csv(path)
    assert len(wd) >= N_HOUR, f"{path}: {len(wd)} rows"
    price = wd["price_eur_per_kWh_at_base_price"].to_numpy(dtype=float)[:N_HOUR]
    return np.repeat(price, 12)


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
    return scenario, f"Simulation_{scenario}_RAD_{building}_{layout}.csv"


def main():
    os.makedirs(OUT, exist_ok=True)
    base = pd.read_csv(
        os.path.join(HERE, "reference", "Simulation_CY_Cur_HP__PV5000-HB5000.csv"),
        sep=";",
    )
    shared = {
        c: base[c].to_numpy()
        for c in ["ev_status", "tar_flat", "tar_tou", "ev_conn", "ev_arrival", "ev_departure"]
    }
    # 2001 starts on a Monday and is not a leap year, so the timestamps' weekday
    # matches the TOU/EV calendar (day 0 = Monday) and the year ends on 31/12
    timestamps = pd.date_range("2001-01-01", periods=N_5MIN, freq="5min").strftime(
        "%d/%m/%Y %H:%M"
    )

    weather_cache, tariff_cache = {}, {}
    rad_files = collect_rad_files()
    print(f"{len(rad_files)} RAD files found")

    for i, (name, path) in enumerate(sorted(rad_files.items()), 1):
        scenario, out_name = parse_rad_name(name)
        if scenario not in weather_cache:
            weather_cache[scenario] = load_epw_5min(scenario)
            tariff_cache[scenario] = load_tariff_5min(scenario)

        rad = pd.read_csv(path)
        assert len(rad) == N_15MIN, f"{path}: {len(rad)} rows"
        demand = np.repeat(rad.iloc[:, 1].to_numpy(dtype=float), 3)
        produced = np.repeat(rad.iloc[:, 2].to_numpy(dtype=float), 3)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "electricity_demand_rate_W": demand,
            "produced_electricity_rate_W": produced,
            **weather_cache[scenario],
            "ev_status": shared["ev_status"],
            "tar_s": tariff_cache[scenario],
            "tar_w": tariff_cache[scenario],
            "tar_sw": tariff_cache[scenario],
            "tar_flat": shared["tar_flat"],
            "tar_tou": shared["tar_tou"],
            "ev_conn": shared["ev_conn"],
            "ev_arrival": shared["ev_arrival"],
            "ev_departure": shared["ev_departure"],
        })[COLUMNS]
        df.to_csv(os.path.join(OUT, out_name), sep=";", index=False)
        print(f"[{i}/{len(rad_files)}] {out_name}")


if __name__ == "__main__":
    main()
