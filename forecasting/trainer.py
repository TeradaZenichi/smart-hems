# -*- coding: utf-8 -*-
"""Training loop with early stopping and checkpointing.

A orquestração (paciência, melhor checkpoint, log) está pronta; os dois
métodos que tocam gradiente — o coração do PyTorch — são seus TODOs.
"""

import copy
import os

import torch
from torch.utils.data import DataLoader


class Trainer:
    """Treina um forecaster com MSE, Adam e early stopping no MSE de validação.

    Uso:
        t = Trainer(model)
        historia = t.fit(train_ds, val_ds)
        t.save("results/forecaster_gru.pt")
    """

    def __init__(self, model, lr: float = 3e-4, batch_size: int = 256,
                 patience: int = 8, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.patience = patience
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.best_state = None
        self.best_val = float("inf")

    # ----- os dois TODOs centrais ------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> float:
        """Uma época de treino; retorna o MSE médio (ponderado por batch)."""
        self.model.train()  # ativa dropout/batchnorm se existirem
        # TODO(lição 4): itere `for hist, static, target in loader:` e, em cada
        # batch, execute o ciclo canônico do PyTorch:
        #   1. hist = hist.to(self.device)  (idem static, target)
        #        -> tensores e modelo PRECISAM estar no mesmo device
        #   2. self.opt.zero_grad()      zera os .grad da iteração anterior
        #   3. pred = self.model(hist, static)
        #   4. loss = self.loss_fn(pred, target)
        #   5. loss.backward()           autograd preenche cada p.grad
        #   6. self.opt.step()           Adam anda um passo nos pesos
        # Métrica: some `loss.item() * hist.size(0)` (soma ponderada pelo
        #   tamanho do batch) num acumulador e divida por len(loader.dataset).
        #   .item() extrai o float e SOLTA o grafo (senão você segura memória).
        # Ordem importa: zero_grad ANTES de backward, pois .backward() SOMA
        #   nos .grad (esquecer = gradientes de batches anteriores vazando).
        raise NotImplementedError("TODO: Trainer._train_epoch")

    def _evaluate(self, loader: DataLoader) -> float:
        """MSE médio sem treinar (validação)."""
        self.model.eval()
        # TODO(lição 4): igual ao _train_epoch, mas SEM os passos 2/5/6
        # (nada de zero_grad/backward/step — validação não treina) e com todo
        # o loop dentro de:
        #     with torch.no_grad():
        #         for hist, static, target in loader:
        #             ...
        # torch.no_grad() diz ao autograd para NÃO montar o grafo: o forward
        # fica ~2x mais rápido e não acumula memória. Retorne o MSE médio
        # ponderado, do mesmo jeito que no treino.
        raise NotImplementedError("TODO: Trainer._evaluate")

    # ----- orquestração (pronta) -------------------------------------------
    def fit(self, train_ds, val_ds, max_epochs: int = 100) -> list[dict]:
        # num_workers=0: no Windows, workers de DataLoader sobem processos
        # novos (spawn) — para datasets já em RAM, o overhead não compensa.
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=False, num_workers=0)
        history, bad_epochs = [], 0
        for epoch in range(1, max_epochs + 1):
            train_mse = self._train_epoch(train_loader)
            val_mse = self._evaluate(val_loader)
            history.append({"epoch": epoch, "train": train_mse, "val": val_mse})
            marker = ""
            if val_mse < self.best_val:
                self.best_val = val_mse
                self.best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs, marker = 0, "  <- melhor"
            else:
                bad_epochs += 1
            print(f"epoca {epoch:3d}  train {train_mse:.5f}  val {val_mse:.5f}{marker}")
            if bad_epochs >= self.patience:
                print(f"early stopping ({self.patience} epocas sem melhora)")
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return history

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(),
                    "best_val": self.best_val}, path)

    @staticmethod
    def load_into(model, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
