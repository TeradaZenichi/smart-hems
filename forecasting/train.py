# -*- coding: utf-8 -*-
"""Train the forecaster on the typical current scenarios (REF+TDYC). Just
glue, no TODOs: once the smoke test passes end to end, this simply works.

Run directly (VSCode play button) or as a module:
    .venv\\Scripts\\python.exe -m forecasting.train
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.dataset import ForecastWindows
from forecasting.hyperparameters import ForecasterHP
from forecasting.trainer import Trainer
from model.gru import GRUForecaster

CKPT = str(PROJECT_ROOT / "data" / "checkpoints" / "forecaster_gru.pt")
HP_JSON = str(PROJECT_ROOT / "data" / "checkpoints" / "forecaster_gru_hp.json")


def main(hp: ForecasterHP | None = None):
    hp = hp or ForecasterHP()  # defaults; the HPO passes candidates through here
    import torch
    torch.manual_seed(hp.seed)

    t0 = time.time()
    train_ds = ForecastWindows("train")
    val_ds = ForecastWindows("val")
    print(f"windows: {len(train_ds)} train, {len(val_ds)} val "
          f"({time.time() - t0:.0f}s to load)")

    model = GRUForecaster(hidden=hp.hidden, mlp_hidden=hp.mlp_hidden)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"GRUForecaster: {n_par/1e3:.0f}k params | hp: {hp.to_dict()}")

    trainer = Trainer(model, lr=hp.lr, batch_size=hp.batch_size,
                      patience=hp.patience)
    trainer.fit(train_ds, val_ds, max_epochs=hp.max_epochs)
    trainer.save(CKPT)
    hp.save(HP_JSON)  # config saved next to the checkpoint: reproducible
    print(f"best val MSE: {trainer.best_val:.5f} | saved to {CKPT}")
    return trainer.best_val


if __name__ == "__main__":
    main()
