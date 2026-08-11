# -*- coding: utf-8 -*-
"""Agente SAC para controle energético."""

from pathlib import Path


class SAC:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]
        self.cache_dir = root / "results" / "forecast_cache"
        self.model_file = root / "results" / "sac.pt"

    def train(self):
        # TODO: treinar os testes sem previsão, com futuro perfeito e com
        # previsão do modelo.
        raise NotImplementedError("TODO: SAC.train")
