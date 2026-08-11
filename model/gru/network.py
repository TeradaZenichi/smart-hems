# -*- coding: utf-8 -*-
"""Arquitetura GRU do forecaster.

Encoder GRU compartilhado + duas cabeças MLP: uma para demanda e outra para
PV. Os shapes vêm de model/spec.py e os defaults de model/model.json.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from model.spec import HIST_DIM, HORIZON, STATIC_DIM

MODEL_FILE = Path(__file__).resolve().parents[1] / "model.json"
with MODEL_FILE.open(encoding="utf-8") as file:
    GRU_DEFAULTS = json.load(file)["gru"]


class GRUForecaster(nn.Module):
    """Codifica o histórico e prevê demanda e PV para todo o horizonte."""

    def __init__(
        self,
        hidden: int = GRU_DEFAULTS["hidden"],
        mlp_hidden: int = GRU_DEFAULTS["mlp_hidden"],
    ):
        super().__init__()
        self.gru = nn.GRU(input_size=HIST_DIM, hidden_size=hidden, batch_first=True)
        self.load_head = nn.Sequential(
            nn.Linear(hidden + STATIC_DIM, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, HORIZON),
        )
        self.pv_head = nn.Sequential(
            nn.Linear(hidden + STATIC_DIM, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, HORIZON),
        )
      
    def forward(self, hist: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """(B, 672, 4), (B, 18) -> (B, 96, 2)."""
        _, h_n = self.gru(hist)
        z = torch.cat([h_n[-1], static], dim=1)
        load = self.load_head(z)
        pv = self.pv_head(z)
        return torch.stack([load, pv], dim=2)
