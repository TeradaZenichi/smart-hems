# -*- coding: utf-8 -*-
"""Treinamento da previsão com referência e anos típicos atual e futuro."""

import time
from pathlib import Path

import torch

from forecasting.dataset import ForecastWindows
from forecasting.hyperparameters import ForecasterHP
from forecasting.trainer import Trainer
from model.gru import GRUForecaster

ROOT = Path(__file__).resolve().parents[1]


class Forecaster:
    def __init__(self, hp: ForecasterHP | None = None):
        self.hp = hp or ForecasterHP()
        self.model_file = ROOT / "results" / "forecaster_gru.pt"
        self.hp_file = ROOT / "results" / "forecaster_gru_hp.json"

    def train(self):
        torch.manual_seed(self.hp.seed)

        start = time.time()
        train_data = ForecastWindows("train")
        val_data = ForecastWindows("val")
        print(
            f"windows: {len(train_data)} train, {len(val_data)} val "
            f"({time.time() - start:.0f}s to load)"
        )

        model = GRUForecaster(
            hidden=self.hp.hidden,
            mlp_hidden=self.hp.mlp_hidden,
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"GRUForecaster: {params / 1e3:.0f}k params | hp: {self.hp.to_dict()}")

        trainer = Trainer(
            model,
            lr=self.hp.lr,
            batch_size=self.hp.batch_size,
            patience=self.hp.patience,
        )
        trainer.fit(train_data, val_data, max_epochs=self.hp.max_epochs)
        trainer.save(str(self.model_file))
        self.hp.save(str(self.hp_file))
        print(f"best val MSE: {trainer.best_val:.5f} | saved to {self.model_file}")
        return trainer.best_val
