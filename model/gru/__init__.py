# -*- coding: utf-8 -*-
"""Arquitetura GRU do forecaster.

Encoder GRU sobre o histórico + MLP que funde o resumo com as features
estáticas. Shapes de entrada/saída vêm de model/spec.py (contrato de I/O
agnóstico de arquitetura); os defaults de largura vêm de model/model.json.
"""

import json
import os

import torch
import torch.nn as nn

from model.spec import HORIZON, N_SERIES, STATIC_DIM

# Defaults de arquitetura desta rede. forecasting/hyperparameters.py importa
# GRU_DEFAULTS daqui em vez de hardcodar os mesmos números — o valor "de
# fábrica" mora só em model/model.json, sob a chave "gru". O ganhador de uma
# busca Optuna, ou de um treino específico, NÃO volta pra cá: fica no
# *_hp.json salvo ao lado do checkpoint (forecasting/train.py), que é o
# registro reprodutível daquele run.
_MODEL_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.json"
)
with open(_MODEL_JSON_PATH, encoding="utf-8") as _f:
    GRU_DEFAULTS = json.load(_f)["gru"]


class GRUForecaster(nn.Module):
    """Encoder GRU sobre o histórico + MLP que funde o resumo com as
    features estáticas e emite o horizonte inteiro de uma vez (direto,
    não autoregressivo — sem acúmulo de erro passo a passo).

    Fluxo do forward:
        hist (B, 672, 2) ──GRU──> h_T (B, hidden)          [último estado]
        concat([h_T, static])  (B, hidden+54)
        ──MLP──> (B, 96*2) ──reshape──> (B, 96, 2)
    """

    def __init__(self, hidden: int = GRU_DEFAULTS["hidden"],
                 mlp_hidden: int = GRU_DEFAULTS["mlp_hidden"]):
        super().__init__()
        # TODO(lição 4): crie as camadas.
        #   self.gru  — nn.GRU(input_size=N_SERIES, hidden_size=hidden,
        #               batch_first=True)  <- batch_first pois nossos tensores
        #               são (B, T, C); sem ele a GRU espera (T, B, C).
        #   self.head — nn.Sequential(camada1, ativação, camada2):
        #               nn.Linear(hidden + STATIC_DIM, mlp_hidden),
        #               nn.ReLU(),
        #               nn.Linear(mlp_hidden, HORIZON * N_SERIES)
        #   (nn.Sequential recebe as camadas como argumentos, em ordem)
        raise NotImplementedError("TODO: GRUForecaster.__init__")

    def forward(self, hist: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """(B, 672, 2), (B, 54) -> (B, 96, 2)"""
        # TODO(lição 4):
        #   1. saidas, h_n = self.gru(hist)
        #      - saidas: (B, T, hidden) — o estado em CADA passo do histórico
        #      - h_n:    (1, B, hidden) — só o estado final (eixo 0 = nº de
        #        camadas da GRU, aqui 1). Pegue o resumo com h_n[0]  -> (B, hidden)
        #        (equivale a saidas[:, -1, :], o último passo)
        #   2. junte com as features estáticas:
        #        z = torch.cat([resumo, static], dim=1)  -> (B, hidden+54)
        #        (dim=1 = eixo das features; dim=0 é o batch, não mexa nele)
        #   3. self.head(z) -> (B, 96*2); então .reshape(-1, HORIZON, N_SERIES)
        #      para virar (B, 96, 2). -1 deixa o PyTorch inferir o batch.
        raise NotImplementedError("TODO: GRUForecaster.forward")
