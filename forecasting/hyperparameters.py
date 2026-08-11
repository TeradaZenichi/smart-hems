# -*- coding: utf-8 -*-
"""Hyperparameters as a dataclass: one object, three roles.

  1. Typed config with defaults, instead of scattered loose arguments.
  2. Serializable (JSON next to the checkpoint) for reproducible runs.
  3. Optuna-integrated: the SEARCH SPACE lives in each field's metadata, next
     to its default. No metadata -> fixed; with metadata -> tunable. Search
     space and config never drift into separate files.

Optuna is OPTIONAL (this module does not import it). Plain use:

    hp = ForecasterHP()                      # defaults
    hp = ForecasterHP(lr=1e-3, hidden=256)   # point overrides
    hp = ForecasterHP.load("run_x_hp.json")  # reproduce an old run

With Optuna: ForecasterHP.suggest(trial) samples the tunable fields; see the
__main__ block below for a runnable example.
"""

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# Bootstrap so "import model.gru" works both when imported as
# forecasting.hyperparameters and when run directly (VSCode play button).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.gru import GRU_DEFAULTS  # single source for the architecture defaults


def _space(kind, *args):
    """Mark a field tunable: kind in {'log_float','float','int','cat'}."""
    return {"space": (kind, args)}


@dataclass
class Hyperparameters:
    """Base: generic serialization and Optuna sampling for any subclass."""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict):
        """Ignore unknown keys, so a JSON saved by an older version of the
        class (renamed/removed field) still loads."""
        known = {f_.name for f_ in fields(cls)}
        extra = set(d) - known
        if extra:
            print(f"[{cls.__name__}] ignoring unknown keys: {sorted(extra)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def suggest(cls, trial):
        """Instantiate by sampling the _space-annotated fields from the trial;
        the rest keep their defaults. Works for any subclass."""
        values = {}
        for f_ in fields(cls):
            spec = f_.metadata.get("space")
            if spec is None:
                continue
            kind, args = spec
            if kind == "log_float":
                values[f_.name] = trial.suggest_float(f_.name, *args, log=True)
            elif kind == "float":
                values[f_.name] = trial.suggest_float(f_.name, *args)
            elif kind == "int":
                values[f_.name] = trial.suggest_int(f_.name, *args)
            elif kind == "cat":
                values[f_.name] = trial.suggest_categorical(f_.name, list(args))
            else:
                raise ValueError(f"unknown space: {kind}")
        return cls(**values)


@dataclass
class ForecasterHP(Hyperparameters):
    """GRUForecaster + Trainer config. Fields with _space enter the HPO."""

    # architecture — default from model/model.json (GRU_DEFAULTS); only the
    # Optuna search space is defined here, the factory value lives there
    hidden: int = field(default=GRU_DEFAULTS["hidden"], metadata=_space("cat", 64, 128, 256))
    mlp_hidden: int = field(default=GRU_DEFAULTS["mlp_hidden"], metadata=_space("cat", 128, 256, 512))

    # optimization (forecasting/trainer.py)
    lr: float = field(default=3e-4, metadata=_space("log_float", 1e-4, 3e-3))
    batch_size: int = field(default=256, metadata=_space("cat", 128, 256, 512))

    # fixed (no metadata -> Optuna leaves them alone)
    patience: int = 8
    max_epochs: int = 100
    seed: int = 42


if __name__ == "__main__":
    # class smoke test: defaults, disk round-trip, old-JSON tolerance, and
    # Optuna sampling (skipped if optuna is not installed).
    hp = ForecasterHP()
    tmp = os.path.join(tempfile.gettempdir(), "smart_hems_demo_hp.json")
    hp.save(tmp)
    assert ForecasterHP.load(tmp) == hp                # round-trip (compares by value)
    os.remove(tmp)
    assert ForecasterHP(lr=1e-3).lr == 1e-3            # point override
    assert ForecasterHP.from_dict({**hp.to_dict(), "dead_key": 1}) == hp  # ignores dead key
    print("defaults + round-trip + overrides: ok")

    try:
        import optuna
    except ImportError:
        print("optuna not installed — HPO sampling skipped (the class does not need it)")
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: (ForecasterHP.suggest(t).lr - 1e-3) ** 2, n_trials=3)
        assert set(study.best_params) <= {"hidden", "mlp_hidden", "lr", "batch_size"}, \
            "Optuna may only touch _space fields; patience/max_epochs/seed are fixed"
        print("Optuna sampling (only _space fields): ok")
