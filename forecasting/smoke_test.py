# -*- coding: utf-8 -*-
"""Verificador dos TODOs do forecasting, em ordem de dependência.

Para no primeiro TODO pendente ou assert que falhar. Roda rápido: nada aqui
depende de época cheia de treino em dados reais — só a etapa 3 lê CSVs.

    .venv\\Scripts\\python.exe forecasting\\smoke_test.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def stage(n, nome):
    print(f"\n--- etapa {n}: {nome} ---")


def etapa_1_normalizer():
    from forecasting.dataset import Normalizer
    norm = Normalizer()
    p = norm.transform_power(torch.tensor([0.0, 1435.4, 10000.0]))
    assert abs(p[1].item() - 0.14354) < 1e-4, "transform_power: W -> kW -> /Pnorm"
    assert abs(norm.inverse_power(p)[1].item() - 1.4354) < 1e-4, "inverse_power deve parar em kW"
    t = norm.transform_weather(torch.tensor([-14.0, 36.59]), "drybulb_C")
    assert abs(t[0]) < 1e-3 and abs(t[1] - 1.0) < 1e-3, "transform_weather: min-max com clamp"
    print("ok")


def etapa_2_calendario():
    import pandas as pd
    from forecasting.dataset import calendar_features, case_features
    from model.spec import META_DIM
    c = calendar_features(pd.Timestamp("2001-01-01 00:00"))
    assert c.shape == (6,), f"esperava (6,), veio {tuple(c.shape)}"
    assert abs(c[0]) < 1e-6 and abs(c[1] - 1.0) < 1e-6, "meia-noite: sin=0, cos=1"
    meta = case_features("young-couple_east_poor_east-west-tilt(30)_PV1000")
    assert meta.shape == (META_DIM,), f"esperava metadata ({META_DIM},), veio {tuple(meta.shape)}"
    assert meta[:-1].sum().item() == 3.0, "familia, casa e PV devem ter um one-hot cada"
    assert abs(meta[-1].item() - 30.0 / 90.0) < 1e-6, "tilt deve ser normalizado por 90"
    print("ok")


def etapa_3_dataset():
    from forecasting.dataset import ForecastWindows
    from model.spec import HIST_DIM, HIST_STEPS, HORIZON, N_TARGETS, STATIC_DIM
    print("  (carregando 18 CSVs do cenario REF...)")
    ds = ForecastWindows("val", scenarios=["REF"])
    hist, static, target = ds[0]
    assert hist.shape == (HIST_STEPS, HIST_DIM) and static.shape == (STATIC_DIM,) \
        and target.shape == (HORIZON, N_TARGETS), \
        f"shapes: hist {tuple(hist.shape)} static {tuple(static.shape)} target {tuple(target.shape)}"
    s, i = ds.origins[0]
    assert torch.equal(hist[-1, :N_TARGETS], ds.series[s]["power"][i - 1]), \
        "historico de potencia deve terminar em i-1"
    assert torch.equal(hist[-1, N_TARGETS:], ds.series[s]["weather"][i - 1]), \
        "historico meteorologico deve terminar em i-1"
    assert torch.equal(target[0], ds.series[s]["power"][i]), "target deve comecar em i"
    print(f"ok ({len(ds)} janelas)")


def etapa_4_baselines():
    from model.baselines import PersistenceForecaster, SeasonalPersistenceForecaster
    from model.spec import HIST_DIM, HIST_STEPS, HORIZON, N_TARGETS
    fake = torch.arange(HIST_STEPS * HIST_DIM, dtype=torch.float32).reshape(1, HIST_STEPS, HIST_DIM)
    assert torch.equal(PersistenceForecaster()(fake, None), fake[:, -HORIZON:, :N_TARGETS]), \
        "persistencia = alvos das ultimas 24h"
    assert torch.equal(SeasonalPersistenceForecaster()(fake, None), fake[:, :HORIZON, :N_TARGETS]), \
        "sazonal = alvos de D-7"
    print("ok")


def etapa_5_gru():
    from model.gru import GRUForecaster
    from model.spec import HIST_DIM, HIST_STEPS, HORIZON, N_TARGETS, STATIC_DIM
    model = GRUForecaster(hidden=32, mlp_hidden=64)
    assert model.load_head[-1].out_features == HORIZON, "cabeca de demanda deve prever 96 passos"
    assert model.pv_head[-1].out_features == HORIZON, "cabeca de PV deve prever 96 passos"
    hist = torch.randn(4, HIST_STEPS, HIST_DIM)
    static = torch.randn(4, STATIC_DIM)
    out = model(hist, static)
    assert out.shape == (4, HORIZON, N_TARGETS), \
        f"forward deve sair (4, {HORIZON}, {N_TARGETS}), veio {tuple(out.shape)}"
    print("ok")


def etapa_6_trainer():
    from torch.utils.data import DataLoader, TensorDataset
    from model.gru import GRUForecaster
    from model.spec import HIST_DIM, HIST_STEPS, HORIZON, N_TARGETS, STATIC_DIM
    from forecasting.trainer import Trainer

    torch.manual_seed(0)
    ds = TensorDataset(
        torch.randn(32, HIST_STEPS, HIST_DIM),
        torch.randn(32, STATIC_DIM),
        torch.randn(32, HORIZON, N_TARGETS),
    )
    model = GRUForecaster(hidden=16, mlp_hidden=32)
    trainer = Trainer(model, lr=1e-2, batch_size=8, patience=5, device="cpu")

    before = next(trainer.model.parameters()).clone()
    trainer._evaluate(DataLoader(ds, batch_size=8))
    assert torch.equal(before, next(trainer.model.parameters())), \
        "_evaluate alterou os pesos — falta um torch.no_grad() (ou tem zero_grad/backward/step la dentro)"

    history = trainer.fit(ds, ds, max_epochs=5)
    assert history[-1]["train"] < history[0]["train"], \
        "loss de treino nao caiu — confira zero_grad/backward/step em _train_epoch"
    print(f"ok (loss {history[0]['train']:.3f} -> {history[-1]['train']:.3f} em {len(history)} epocas)")


ETAPAS = [
    ("Normalizer", etapa_1_normalizer),
    ("calendar_features", etapa_2_calendario),
    ("ForecastWindows", etapa_3_dataset),
    ("baselines de persistencia", etapa_4_baselines),
    ("GRUForecaster", etapa_5_gru),
    ("Trainer", etapa_6_trainer),
]


def main():
    for n, (nome, fn) in enumerate(ETAPAS, start=1):
        stage(n, nome)
        try:
            fn()
        except NotImplementedError as e:
            print(f"  >>> falta implementar: {e}")
            return
        except AssertionError as e:
            print(f"  >>> implementado, mas errado: {e}")
            return
    print("\nTUDO OK.")


if __name__ == "__main__":
    main()
