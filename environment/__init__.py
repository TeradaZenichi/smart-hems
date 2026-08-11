from gymnasium import spaces
import gymnasium as gym
import pandas as pd
import numpy as np


def _storage_capex(parameters):
    if "capex_per_kwh" in parameters:
        return float(parameters["capex_per_kwh"]) * float(parameters["Emax"])
    return float(parameters["capex"])


def _battery_step(E, cmd, Emax, Emin, Pmax_c, Pmax_d, η, curve, Δt):
    """Shared storage physics for BESS and EV.

    cmd (kW) >= 0 charges, < 0 discharges. The realized power is limited by the
    SoC-power curve (linearly interpolated) and by the energy headroom; η is
    applied one-way in each direction. Returns (P, E_next).
    """
    soc = min(max(E / Emax, 0.0), 1.0)
    if cmd >= 0:
        cap = float(np.interp(soc, curve["soc"], curve["charge_pu"]))
        P = min(cmd, cap * Pmax_c, (Emax - E) / (Δt * η))
        E = min(E + P * Δt * η, Emax)
    else:
        cap = float(np.interp(soc, curve["soc"], curve["discharge_pu"]))
        P = max(cmd, -cap * Pmax_d, (Emin - E) * η / Δt)
        E = max(E + P * Δt / η, Emin)
    return P, E


class Simulation:
    def __init__(self, start, days, parameters):
        self.timestep = parameters["timestep"]
        self.Δt = self.timestep / 60
        self.start = start
        self.step = start
        self.days = days


class Weather:
    def __init__(self, parameters, df, simulation):
        self.sim = simulation
        mins = np.array([parameters[c]["min"] for c in df.columns], dtype=np.float32)
        maxs = np.array([parameters[c]["max"] for c in df.columns], dtype=np.float32)
        self.norm = np.clip(
            (df.to_numpy(dtype=np.float32) - mins) / (maxs - mins), 0.0, 1.0
        )

    def _get_obs(self):
        return self.norm[self.sim.t_idx].tolist()


class Lags:
    def __init__(self, simulation, grid, pv, load, Pnorm):
        self.sim = simulation
        self.grid, self.pv, self.load, self.Pnorm = grid, pv, load, Pnorm

    def _get_obs(self):
        spd = self.sim.steps_per_day
        i, n = self.sim.t_idx, 1000.0 * self.Pnorm
        i24, i7d = max(0, i - spd), max(0, i - 7 * spd)
        return [
            float(self.grid[i24]), float(self.grid[i7d]),
            float(self.pv[i24]) / n, float(self.pv[i7d]) / n,
            float(self.load[i24]) / n, float(self.load[i7d]) / n,
        ]


class Grid:
    def __init__(self, parameters, df, simulation):
        self.sim = simulation
        self.df = df
        self.Pmax = parameters["Pmax_import"]
        self.Pmin = -parameters["Pmax_export"]
        self.export_tariff_factor = parameters.get("export_tariff_factor", 1.0)
        self.penalty = parameters["net_penalty"]

    def step(self, power):
        tariff_t = self.df[self.sim.t_idx]
        import_power = max(power, 0.0)
        export_power = max(-power, 0.0)
        cost = (import_power - self.export_tariff_factor * export_power) * tariff_t * self.sim.Δt
        penalty = 0.0
        if power > self.Pmax:
            penalty = self.penalty * (power - self.Pmax) * self.sim.Δt
        if power < self.Pmin:
            penalty = self.penalty * (self.Pmin - power) * self.sim.Δt
        return power, cost, penalty


class LoadEnv():
    def __init__(self, df, simulation):
        self.sim = simulation
        self.df = df

    def step(self):
        Pload = self.df[self.sim.t_idx] / 1000  # kW
        return Pload, 0

class PVEnv():
    def __init__(self, df, simulation):
        self.sim = simulation
        self.df = df

    def step(self, action):
        Ppv = self.df[self.sim.t_idx] / 1000  # kW
        PPV = Ppv * (1 - action)

        # Teacher-like small curtailment penalty: 0.01 * χPV * Ppv * Δt
        pv_cost = 0.01 * action * Ppv * self.sim.Δt

        return PPV, pv_cost

class BatteryEnv():
    def __init__(self, parameters, SoC, simulation):
        self.sim = simulation
        self.Pmax = parameters["Pmax"]
        self.Emax = parameters["Emax"]
        self.DoD = parameters["DoD"]
        self.η = parameters["η"]
        self.capex = _storage_capex(parameters)
        self.soc0 = SoC
        self.soc = SoC
        self.E = self.soc * self.Emax
        self.ncycles = parameters["ncycles"]
        self.soc_power = parameters["soc_power_curve_pu"]

    def reset(self):
        self.soc = self.soc0
        self.E = self.soc * self.Emax

    def step(self, action):
        Emin = self.Emax * (1 - self.DoD)
        P, self.E = _battery_step(
            self.E, action * self.Pmax, self.Emax, Emin,
            self.Pmax, self.Pmax, self.η, self.soc_power, self.sim.Δt,
        )
        self.soc = np.clip(self.E / self.Emax, 0.0, 1.0)

        # degradation on discharged throughput only (capex_references.md rates)
        cost = (self.capex / (self.Emax * self.ncycles)) * max(0.0, -P) * self.sim.Δt
        return P, cost


class EVEnv():
    def __init__(self, parameters, df, simulation):
        self.sim = simulation
        self.ev_conn_arr = df["ev_conn"].to_numpy(dtype=np.int8)
        self.ev_arrival_arr = df["ev_arrival"].to_numpy(dtype=np.float32)
        self.arrival_idx_arr = np.full(len(self.ev_conn_arr), -1, dtype=np.int32)
        current_arrival = -1
        prev_conn = 0
        for idx, conn_raw in enumerate(self.ev_conn_arr):
            conn = int(conn_raw)
            if conn in (1, 2) and prev_conn == 0:
                current_arrival = idx
            if conn in (1, 2):
                self.arrival_idx_arr[idx] = current_arrival
            else:
                current_arrival = -1
            prev_conn = conn
        self.departure_idx_arr = np.full(len(self.ev_conn_arr), -1, dtype=np.int32)
        next_departure = -1
        for idx in range(len(self.ev_conn_arr) - 1, -1, -1):
            conn = int(self.ev_conn_arr[idx])
            if conn == 2:
                next_departure = idx
            if conn in (1, 2):
                self.departure_idx_arr[idx] = next_departure
            else:
                next_departure = -1
        ev = self.ev_conn_arr.astype(np.int32)
        prev = np.concatenate([[0], ev[:-1]])
        is_arrival = (prev == 0) & ((ev == 1) | (ev == 2))
        arrivals = np.where(is_arrival)[0]
        n = len(ev)
        self.next_arrival_idx_arr = np.full(n, -1, dtype=np.int32)
        if len(arrivals) > 0:
            j = np.searchsorted(arrivals, np.arange(n), side="left")
            valid = j < len(arrivals)
            self.next_arrival_idx_arr[valid] = arrivals[j[valid]]

        self.Pmax_c = parameters["Pmax_c"]
        self.Pmax_d = parameters["Pmax_d"]
        self.Emax = parameters["Emax"]
        self.DoD = parameters["DoD"]
        self.η = parameters["η"]
        self.capex = _storage_capex(parameters)
        self.ncycles = parameters["ncycles"]

        self.penalty = parameters["penalty"]
        self.soc_min = parameters["soc_min"]
        self.soc_critical = parameters["soc_critical"]
        self.fast_price = parameters["fast_price"]  # EUR/kWh, fixed public fast-charge price
        self.soc_power = parameters["soc_power_curve_pu"]

        self.prev_conn = 0

        self.soc = 1.0
        self.E = self.soc * self.Emax
        self.status = "disconnected"

    def reset(self):
        self.soc = 1.0
        self.E = self.Emax
        self.status = "disconnected"
        self.prev_conn = 0

    def step(self, action):
        if action >= 0:
            command = action * self.Pmax_c
        else:
            command = action * self.Pmax_d

        cost = 0.0

        t = self.sim.step
        idx_t = self.sim.t_idx
        conn_t = int(self.ev_conn_arr[idx_t])

        connected_t = conn_t in (1, 2)
        connected_prev = int(self.prev_conn) in (1, 2)

        is_start = (t == self.sim.start)
        is_arrival = (not is_start) and connected_t and (not connected_prev)

        Emin = self.Emax * (1 - self.DoD)

        if conn_t == 0:
            P = 0.0
            self.status = "disconnected"

        elif is_start and connected_t:
            P = 0.0
            self.status = "arriving"

        elif is_arrival:
            P = 0.0
            trip = float(self.ev_arrival_arr[idx_t])
            E_trip = self.Emax * trip
            E_dep = float(self.E)  # HOLD: energia do departure anterior
            Ecrit = float(self.soc_critical) * self.Emax

            if E_trip <= E_dep:
                self.E = E_dep - E_trip

            else:
                E_leg = self.Emax - Ecrit
                if E_leg <= 1e-9:
                    # fallback
                    deficit = E_trip - E_dep
                    cost += float(self.fast_price) * deficit
                    self.E = 0.0
                else:
                    E_pre = max(0.0, E_dep - Ecrit)
                    R = E_trip - E_pre  # > 0
                    n_fast = int(np.ceil(R / E_leg))
                    cost += float(n_fast) * float(self.fast_price) * E_leg
                    rem_last = R - float(n_fast - 1) * E_leg
                    self.E = self.Emax - rem_last

            self.E = float(np.clip(self.E, 0.0, self.Emax))

            self.status = "arriving"


        elif conn_t == 2:
            P = 0.0
            self.status = "departing"

        else:
            P, self.E = _battery_step(
                self.E, command, self.Emax, Emin,
                self.Pmax_c, self.Pmax_d, self.η, self.soc_power, self.sim.Δt,
            )
            self.status = "connected"

        self.soc = float(np.clip(self.E / self.Emax, 0.0, 1.0))

        if self.status == "connected":
            sev = max(0.0, self.Emax * self.soc_min - self.E)
            cost += self.penalty * sev * self.sim.Δt

        # degradation on discharged throughput only (capex_references.md rates)
        cost += (self.capex / (self.Emax * self.ncycles)) * max(0.0, -P) * self.sim.Δt

        self.prev_conn = conn_t

        return P, cost


class SmartHomeEnv(gym.Env):
    def __init__(self, df, parameters, start, days, BESS_SoC, tariff,
                 track_operation=True, context=None, extra_obs=None):
        super().__init__()

        self.track_operation = bool(track_operation)
        # static conditioning features (e.g. pv_scale, tariff one-hot) appended
        # at the end of every observation; empty tuple keeps the space unchanged
        self.context = tuple(float(c) for c in (context or ()))
        # per-step exogenous features, one row per timestep (e.g. the forecast
        # feature cache with perfect future or model forecast); row t_idx is
        # appended to the observation
        if extra_obs is not None:
            extra_obs = np.asarray(extra_obs, dtype=np.float32)
            assert len(extra_obs) == len(df), \
                f"extra_obs has {len(extra_obs)} rows for {len(df)} timesteps"
        self.extra_obs = extra_obs
        self.timestamps = list(df.index)
        self.ts_to_idx = {ts: i for i, ts in enumerate(self.timestamps)}

        self.sim = Simulation(start, days, parameters["general"])
        self.sim.t_idx = self.ts_to_idx[pd.Timestamp(self.sim.step)]
        self.sim.steps_per_day = int(24 * 60 // self.sim.timestep)
        self.sim.end_idx = min(len(self.timestamps), self.sim.t_idx + self.sim.steps_per_day * self.sim.days)

        self.bess = BatteryEnv(parameters["BESS"], BESS_SoC, simulation=self.sim)
        self.load = LoadEnv(df["electricity_demand_rate_W"].to_numpy(dtype=np.float32), simulation=self.sim)
        self.pv = PVEnv(df["produced_electricity_rate_W"].to_numpy(dtype=np.float32), simulation=self.sim)

        self.ev = EVEnv(parameters["EV"], df[["ev_conn", "ev_arrival"]], simulation=self.sim)

        self.grid = Grid(parameters["Grid"], df[tariff].to_numpy(dtype=np.float32), simulation=self.sim)

        # slim weather block: with the day-ahead tariff window in the
        # observation, weather no longer needs to proxy the (old) weather-
        # driven tariffs; keep only the direct drivers of demand and PV
        weather_df = df[[
            "drybulb_C", "Global Horizontal Radiation", "Wind Speed (m/s)"
        ]]
        self.weather = Weather(parameters["general"]["state_normalization"], weather_df, simulation=self.sim)

        self.Pnorm = parameters["general"]["Pnorm"]
        self.lags = Lags(self.sim, self.grid.df, self.pv.df, self.load.df, self.Pnorm)

        # day-ahead tariff window: hours ahead exposed in the observation
        self.tariff_horizon_h = int(parameters["general"].get("tariff_horizon_h", 0))

        self._operation_columns = [
            "bess_cmd", "ev_cmd", "pv_cmd",
            "PLoad", "PPV", "PBESS", "PEV", "PGrid",
            "EBESS", "EEV", "SoCBESS", "SoCEV",
            "χPV",
            "tariff", "reward",
            "energy_cost", "bess_cost", "ev_cost",
            "pv_cost",
            "grid_penalty"
        ]
        self._op_rows = []
        self._op_index = []

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_sample = np.asarray(self._get_observation(), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_sample.shape,
            dtype=np.float32,
        )
        self.state = obs_sample.copy()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        PD, _ = self.load.step()
        PBESS, bess_cost = self.bess.step(action[0])
        PEV, ev_cost = self.ev.step(action[1])
        PPV, pv_cost = self.pv.step(action[2])

        PGrid, energy_cost, penalty = self.grid.step(PD + PBESS + PEV - PPV)

        reward = - (energy_cost + penalty + bess_cost + ev_cost + pv_cost)

        if self.track_operation:
            row = {
                "bess_cmd": float(action[0]),
                "ev_cmd": float(action[1]),
                "pv_cmd": float(action[2]),
                "PLoad": float(PD),
                "PPV": float(PPV),
                "PBESS": float(PBESS),
                "PEV": float(PEV),
                "PGrid": float(PGrid),
                "EBESS": float(self.bess.E),
                "EEV": float(self.ev.E),
                "SoCBESS": float(self.bess.soc),
                "SoCEV": float(self.ev.soc),
                "χPV": float(action[2]),
                "tariff": float(self.grid.df[self.sim.t_idx]),
                "reward": float(reward),
                "energy_cost": float(energy_cost),
                "bess_cost": float(bess_cost),
                "ev_cost": float(ev_cost),
                "pv_cost": float(pv_cost),
                "grid_penalty": float(penalty),
            }
            self._op_rows.append(row)
            self._op_index.append(self.sim.step)

        info = {
            "energy_cost": energy_cost,
            "bess_cost": bess_cost,
            "ev_cost": ev_cost,
            "penalty": penalty,
            "pv_cost": pv_cost,
            "pgrid": PGrid,
            "pbess": PBESS,
            "pev": PEV,
            "ppv": PPV,
            "timestep": self.sim.step,
        }

        next_idx = self.sim.t_idx + 1
        terminated = next_idx >= self.sim.end_idx
        truncated = False

        if not terminated:
            self.sim.t_idx = next_idx
            self.sim.step = self.timestamps[self.sim.t_idx]
            self.state = self._get_observation()

        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "start" in options:
            self.sim.start = pd.Timestamp(options["start"])
            self.sim.step = self.sim.start
        else:
            self.sim.step = self.sim.start

        if "days" in options:
            self.sim.days = options["days"]

        self.sim.t_idx = self.ts_to_idx[pd.Timestamp(self.sim.step)]
        self.sim.end_idx = min(len(self.timestamps), self.sim.t_idx + self.sim.steps_per_day * self.sim.days)

        if "bess_soc" in options:
            self.bess.soc0 = float(options["bess_soc"])

        self.bess.reset()
        self.ev.reset()
        self._op_rows.clear()
        self._op_index.clear()
        self.state = self._get_observation()
        return self.state, {}

    @property
    def operation(self):
        """Per-step operation log of the current episode (built on access)."""
        return pd.DataFrame(
            self._op_rows, index=self._op_index, columns=self._operation_columns
        )

    def close(self):
        self.state = None

    def _get_observation(self):
        # day-of-month intentionally absent: no energetic meaning beyond
        # what month + weekday + hour already encode
        observation = [
            np.sin(2 * np.pi * (self.sim.step.minute / 60.0)),
            np.cos(2 * np.pi * (self.sim.step.minute / 60.0)),
            np.sin(2 * np.pi * (self.sim.step.hour / 24.0)),
            np.cos(2 * np.pi * (self.sim.step.hour / 24.0)),
            np.sin(2 * np.pi * ((self.sim.step.month - 1) / 12.0)),
            np.cos(2 * np.pi * ((self.sim.step.month - 1) / 12.0)),
            np.sin(2 * np.pi * (self.sim.step.weekday() / 7.0)),
            np.cos(2 * np.pi * (self.sim.step.weekday() / 7.0)),
        ]

        conn_t = int(self.ev.ev_conn_arr[self.sim.t_idx])
        ev_connected = conn_t in (1, 2)
        is_start_obs = (self.sim.step == self.sim.start)
        is_arrival_obs = ev_connected and (int(self.ev.prev_conn) == 0) and (not is_start_obs)
        ev_controllable = (conn_t == 1) and (not is_start_obs) and (not is_arrival_obs)

        # Expose persisted EV SoC during disconnection (last-known value); the
        # ev_controllable flag tells the agent whether it can act on it.
        ev_soc_obs = float(self.ev.soc)

        power_obs = [
            (self.load.df[self.sim.t_idx] / 1000) / self.Pnorm,
            (self.pv.df[self.sim.t_idx] / 1000) / self.Pnorm,
            self.bess.soc,
            ev_soc_obs,
            1.0 if ev_controllable else 0.0,
        ]

        tariff_obs = [self.grid.df[self.sim.t_idx]]
        weather_obs = self.weather._get_obs()
        observations = observation + power_obs + tariff_obs + weather_obs
        observations.extend(self._extra_features(ev_connected))
        observations.extend(self.lags._get_obs())
        # appended last so earlier observation indices stay stable (shaping.py);
        # order: tariff window, per-step extras (forecast block), static context
        observations.extend(self._future_tariffs())
        if self.extra_obs is not None:
            observations.extend(self.extra_obs[self.sim.t_idx])
        observations.extend(self.context)

        return np.array(observations, dtype=np.float32)

    def _future_tariffs(self) -> list[float]:
        """Day-ahead tariff window: the tariff 1..H hours ahead of now.

        Tariffs are hourly step functions, so hourly samples represent the
        window exactly; the last value is held at the end of the data.
        """
        if self.tariff_horizon_h <= 0:
            return []
        sph = int(60 // self.sim.timestep)
        idx = self.sim.t_idx + sph * np.arange(1, self.tariff_horizon_h + 1)
        idx = np.minimum(idx, len(self.grid.df) - 1)
        return [float(v) for v in self.grid.df[idx]]

    def _extra_features(self, ev_connected: bool) -> list[float]:
        """Extra timing features without exposing future energy demand.
        """
        spd = int(self.sim.steps_per_day)
        i = int(self.sim.t_idx)
        lo = max(0, i - spd)
        past_tar = self.grid.df[lo:i + 1]
        if past_tar.size > 1:
            t_now = float(self.grid.df[i])
            t_min, t_max = float(past_tar.min()), float(past_tar.max())
            tar_pct = (t_now - t_min) / (t_max - t_min) if t_max > t_min else 0.5
        else:
            tar_pct = 0.5

        win = max(1, int(60 // self.sim.timestep))
        lo2 = max(0, i - win + 1)
        past_pv = self.pv.df[lo2:i + 1]
        pv_ema = float(past_pv.mean()) / 1000.0 / self.Pnorm if past_pv.size else 0.0

        scale = 12.0 * 60.0 / self.sim.timestep
        if ev_connected:
            arr_idx = int(self.ev.arrival_idx_arr[i])
            dep_idx = int(self.ev.departure_idx_arr[i])
            t_since = min(1.0, max(0, i - arr_idx) / scale) if arr_idx >= 0 else 0.0
            t_until_dep = min(1.0, max(0, dep_idx - i) / scale) if dep_idx >= 0 else 0.0
            t_until_arr = 0.0
        else:
            nxt = int(self.ev.next_arrival_idx_arr[i])
            t_since = 0.0
            t_until_dep = 0.0
            t_until_arr = min(1.0, max(0, nxt - i) / scale) if nxt >= 0 else 1.0

        return [float(tar_pct), float(pv_ema), float(t_since), float(t_until_dep), float(t_until_arr)]
