# -*- coding: utf-8 -*-
"""Feature caches: transformam previsões em observações para o SAC.

A interface é a mesma para M3 e M4 — essa igualdade É o experimento:
    GroundTruthFeatureCache  -> blocos do futuro real       (M3, oracle)
    ForecastFeatureCache     -> blocos da saída do modelo   (M4)

Ambas produzem, por série-ano, um array (35040, 24): para cada passo t,
as próximas 24 h de [demanda, pv] comprimidas em 12 blocos de 2 h.
Como as séries são exógenas (não dependem das ações do agente), o cache é
pré-computado uma vez e salvo em disco; o treino do SAC só lê.
"""

import os

import numpy as np
import torch

from model.spec import HIST_STEPS, HORIZON

N_BLOCKS = 12                      # 24 h em blocos de 2 h
BLOCK = HORIZON // N_BLOCKS        # 8 passos de 15 min por bloco
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "forecast_cache")


class BlockCompressor:
    """(N, 96, 2) -> (N, 24): média de cada bloco de 2 h, por série,
    concatenadas como [d0..d11, pv0..pv11]."""

    def __call__(self, horizon: torch.Tensor) -> torch.Tensor:
        # TODO(lição 4): sem loops! Só reshape + redução + concat.
        #   1. horizon.reshape(-1, N_BLOCKS, BLOCK, 2)   (N, 12, 8, 2)
        #        agrupa os 96 passos em 12 blocos de 8 (= 2 h)
        #   2. .mean(dim=2)                              (N, 12, 2)
        #        média DENTRO de cada bloco (dim=2 é o eixo dos 8 passos)
        #   3. separe as séries e junte lado a lado:
        #        blocos[..., 0] -> demanda (N, 12);  blocos[..., 1] -> pv (N, 12)
        #        torch.cat([demanda, pv], dim=1)         (N, 24)
        #        (`...` = "todos os eixos anteriores"; indexa só o último)
        raise NotImplementedError("TODO: BlockCompressor.__call__")


class GroundTruthFeatureCache:
    """Blocos calculados do futuro REAL da série — o teto de informação (M3).

    power: tensor (35040, 2) normalizado (mesma Normalizer do dataset).
    """

    def __init__(self, power: torch.Tensor):
        self.power = power
        self.compress = BlockCompressor()

    def build(self) -> np.ndarray:
        """(35040, 24); nos últimos HORIZON-1 passos o futuro é truncado —
        repetimos a última linha completa (borda do ano, sem episódios lá)."""
        n = len(self.power)
        # TODO(lição 4): monte todas as janelas futuras de uma vez (vetorizado).
        #   1. self.power.unfold(0, HORIZON, 1)   -> (n-95, 2, 96)
        #        unfold(dim, tamanho, passo) cria janelas deslizantes SEM copiar:
        #        para cada t, os HORIZON passos a partir dele. Note que ele põe
        #        o eixo da janela por ÚLTIMO, daí o passo 2.
        #   2. .permute(0, 2, 1)                  -> (n-95, 96, 2)
        #        reordena eixos para o formato que o compressor espera
        #   3. self.compress(janelas)             -> (n-95, 24)  (é um Tensor;
        #        use .numpy() para converter)
        #   4. o resultado tem n-95 linhas, faltam as últimas 95 (fim do ano,
        #        sem futuro completo): crie um array (n, 24) e repita a última
        #        linha válida nelas. Funções: np.empty((n, 24), np.float32) e
        #        atribuição por fatia; ou np.pad(arr, ((0, 95),(0,0)), "edge").
        raise NotImplementedError("TODO: GroundTruthFeatureCache.build")


class ForecastFeatureCache:
    """Blocos calculados da PREVISÃO do modelo para cada origem do ano (M4).

    series: dict como os de ForecastWindows.series (power, weather, index).
    O modelo roda em batches — 35 mil forwards um a um desperdiçariam a GPU.
    """

    def __init__(self, model, series: dict, batch_size: int = 512,
                 device: str | None = None):
        self.model = model
        self.series = series
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compress = BlockCompressor()

    def build(self) -> np.ndarray:
        """(35040, 24). Origens válidas: t em [HIST_STEPS, n-HORIZON]; as
        bordas (primeiros 7 dias / últimas 24 h) repetem a previsão válida
        mais próxima — episódios não começam nem terminam lá."""
        # TODO(lição 5): rode o modelo em batches (nunca 1 origem por vez).
        #   Origens válidas: range(HIST_STEPS, n - HORIZON + 1).
        #   Para cada batch dessas origens t:
        #     1. monte hist e static (mesma receita do __getitem__) e EMPILHE:
        #          torch.stack([...], dim=0)  transforma lista de (672,2) em
        #          um tensor (B, 672, 2); idem static -> (B, 54)
        #     2. self.model.eval(); with torch.no_grad():
        #          pred = self.model(hist.to(self.device), static.to(self.device))
        #     3. self.compress(pred).cpu()  -> guarde numa lista
        #   Depois: torch.cat(lista, dim=0).numpy()  junta todos os batches.
        #   4. bordas: primeiras HIST_STEPS e últimas HORIZON-1 linhas ficam
        #      sem previsão -> repita a linha válida mais próxima (np.pad "edge"
        #      nos dois lados, ou monte (n, 24) e preencha as pontas).
        raise NotImplementedError("TODO: ForecastFeatureCache.build")


def cache_path(case: str, scenario: str, kind: str) -> str:
    """kind: 'truth' (M3) ou 'forecast' (M4)."""
    return os.path.join(CACHE_DIR, f"{kind}_{scenario}_{case}.npy")


def save_cache(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr.astype(np.float32))
