"""Neural architectures for the smart-hems project.

Each architecture is a subpackage model/{architecture}/ (e.g. model/gru/); the
shared I/O contract (shapes) lives in model/spec.py; weightless baselines live
in model/baselines/. Training loops, datasets and inference utilities live in
forecasting/ (and later sac/).
"""

from model.baselines import PersistenceForecaster, SeasonalPersistenceForecaster
from model.gru import GRUForecaster

__all__ = ["GRUForecaster", "PersistenceForecaster", "SeasonalPersistenceForecaster"]
