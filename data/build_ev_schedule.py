"""Build EV schedules (ev_status/ev_conn/ev_arrival/ev_departure) from data/EV.

Source: Dataset 1 (Norwegian residential charging sessions, Dec 2018 - Jan 2020).
Two profiles, each from one private user with ~1 year of overnight home charging:

  current -> AsO2-1  (used by scenarios REF, TDYC, CC, WC)
  future  -> AdO1-3  (used by scenarios TDYF, CF, WF)

Column semantics (matching the reference Simulation_*.csv files):
  ev_conn      0 away, 1 connected, 2 last step before departure
  ev_arrival   battery fraction consumed by the trip that preceded the arrival
               (El_kWh charged in the session x charger efficiency / Emax);
               constant over the session's rows. May exceed 1 for long trips -
               the environment models those as mid-trip fast-charge stops.
  ev_status    max(0, 1 - ev_arrival), informational
  ev_departure the next session's ev_arrival, constant over the session's rows

Calendar: sessions are mapped onto the synthetic 365-day year by day offset from
the Monday of the user's first week, so weekdays line up with the TOU tariff
calendar (day 0 = Monday). Days outside the user's observation window are filled
by copying the nearest observed day with the same weekday (multiples of 7 days).

Outputs: data/EV/ev_schedule_{current,future}_{5min,15min}.csv
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS_CSV = os.path.join(HERE, "EV", "Dataset 1_EV charging reports.csv")
OUT_DIR = os.path.join(HERE, "EV")

PROFILES = {"current": "AsO2-1", "future": "AdO1-3"}
RESOLUTIONS = {"5min": 288, "15min": 96}

EMAX_KWH = 40.0
CHARGER_EFF = 0.92
TRIP_FRACTION_CAP = 2.0
MERGE_GAP = pd.Timedelta(minutes=30)
MIN_DURATION = pd.Timedelta(minutes=15)
DAYS = 365


def load_sessions(user_id):
    df = pd.read_csv(SESSIONS_CSV, sep=";", decimal=",")
    df = df[df["User_ID"] == user_id].copy()
    df["start"] = pd.to_datetime(df["Start_plugin"], format="%d.%m.%Y %H:%M")
    df["end"] = pd.to_datetime(df["End_plugout"], format="%d.%m.%Y %H:%M")
    df = df.sort_values("start").reset_index(drop=True)

    # merge re-plugs: gap to previous session below MERGE_GAP
    merged = []
    for _, row in df.iterrows():
        if merged and row["start"] - merged[-1]["end"] < MERGE_GAP:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
            merged[-1]["El_kWh"] += row["El_kWh"]
        else:
            merged.append({"start": row["start"], "end": row["end"], "El_kWh": row["El_kWh"]})
    out = pd.DataFrame(merged)
    return out[out["end"] - out["start"] >= MIN_DURATION].reset_index(drop=True)


def build_arrays(sessions, steps_per_day):
    """Rasterize sessions onto the synthetic year; returns conn and trip arrays."""
    n = DAYS * steps_per_day
    step_min = 24 * 60 // steps_per_day
    conn = np.zeros(n, dtype=np.int8)
    trip = np.zeros(n, dtype=np.float64)  # ev_arrival value, constant per session

    first = sessions["start"].iloc[0]
    anchor = (first - pd.Timedelta(days=first.weekday())).normalize()  # Monday, day 0

    for _, s in sessions.iterrows():
        a = int((s["start"] - anchor).total_seconds() // (step_min * 60))
        b = int(np.ceil((s["end"] - anchor).total_seconds() / (step_min * 60)))
        a, b = max(a, 0), min(b, n)
        if b - a < 1:
            continue
        frac = min(s["El_kWh"] * CHARGER_EFF / EMAX_KWH, TRIP_FRACTION_CAP)
        conn[a:b] = 1
        trip[a:b] = frac

    # fill days outside the observation window with the nearest same-weekday day
    last_day = int((sessions["end"].max() - anchor).days)
    covered = np.zeros(DAYS, dtype=bool)
    covered[: min(last_day + 1, DAYS)] = True
    for d in np.where(~covered)[0]:
        for off in range(7, DAYS, 7):
            for src in (d - off, d + off):
                if 0 <= src < DAYS and covered[src]:
                    sl = slice(d * steps_per_day, (d + 1) * steps_per_day)
                    sr = slice(src * steps_per_day, (src + 1) * steps_per_day)
                    conn[sl] = conn[sr]
                    trip[sl] = trip[sr]
                    break
            else:
                continue
            break
    return conn, trip


def finalize(conn, trip):
    """Enforce run structure and derive the four ev_* columns."""
    n = len(conn)
    ev_conn = np.zeros(n, dtype=np.int64)
    ev_arrival = np.zeros(n)
    ev_status = np.zeros(n)
    ev_departure = np.zeros(n)

    padded = np.concatenate([[0], (conn != 0).astype(int), [0]])
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]  # exclusive

    # day-copy filling may create runs touching each other; runs list is already
    # maximal, so just derive per-run values
    run_trip = [trip[a:b].max() for a, b in zip(starts, ends)]
    for i, (a, b) in enumerate(zip(starts, ends)):
        ev_conn[a:b] = 1
        ev_conn[b - 1] = 2
        ev_arrival[a:b] = run_trip[i]
        ev_status[a:b] = max(0.0, 1.0 - run_trip[i])
        ev_departure[a:b] = run_trip[i + 1] if i + 1 < len(run_trip) else 0.0
    return pd.DataFrame({
        "ev_status": ev_status,
        "ev_conn": ev_conn,
        "ev_arrival": ev_arrival,
        "ev_departure": ev_departure,
    })


def main():
    for profile, user in PROFILES.items():
        sessions = load_sessions(user)
        print(f"{profile} ({user}): {len(sessions)} sessions after cleaning")
        for res, steps_per_day in RESOLUTIONS.items():
            conn, trip = build_arrays(sessions, steps_per_day)
            df = finalize(conn, trip)
            out = os.path.join(OUT_DIR, f"ev_schedule_{profile}_{res}.csv")
            df.to_csv(out, index=False)
            runs = int((np.diff(np.concatenate([[0], (df['ev_conn'] != 0).astype(int)])) == 1).sum())
            arr = df.loc[(df["ev_conn"] != 0) & (df["ev_conn"].shift(fill_value=0) == 0), "ev_arrival"]
            print(f"  {res}: {runs} sessions/yr, connected {(df['ev_conn'] != 0).mean():.1%}, "
                  f"arrival p50 {arr.median():.3f} (={arr.median() * EMAX_KWH:.1f} kWh), "
                  f"max {arr.max():.3f}")


if __name__ == "__main__":
    main()
