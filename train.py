# -*- coding: utf-8 -*-
"""Pipeline completo: forecaster -> features -> SAC.

Cada etapa vive em sua própria classe. O pipeline apenas organiza a ordem.

Run:
    .venv\Scripts\python.exe train.py
"""


class Pipeline:
    def __init__(self):
        from forecasting.forecaster import Forecaster
        from sac import SAC

        self.forecaster = Forecaster()
        self.agent = SAC()

    def forecast(self):
        return self.forecaster.train()

    def features(self):
        """Gera features com futuro perfeito e com previsão do modelo."""
        from forecasting.predictor import (
            ForecastFeatureCache as ForecastCache,
            GroundTruthFeatureCache as TruthCache,
            cache_path,
            save_cache,
        )
        from forecasting.trainer import Trainer
        from model.gru import GRUForecaster

        raise NotImplementedError("TODO: Pipeline.features")

    def sac(self):
        return self.agent.train()

    def run(self):
        self.forecast()
        self.features()
        self.sac()


if __name__ == "__main__":
    Pipeline().run()
