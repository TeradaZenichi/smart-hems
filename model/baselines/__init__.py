# -*- coding: utf-8 -*-
"""Persistence baselines — the naive guesses the network must beat.

Not trained architectures: no weights, no model.json entry, not nn.Module —
hence kept separate. They expose the same forward(hist, static) interface as
GRUForecaster (duck typing) so the evaluator can treat all forecasters alike.
"""

from typing import Optional

import torch

from model.spec import HORIZON


class PersistenceForecaster:
    """'Tomorrow = today': prediction = last 24 h of the history."""

    def forward(self, hist: torch.Tensor,
                static: Optional[torch.Tensor] = None) -> torch.Tensor:
        # `static` is ignored; kept in the signature to match GRUForecaster.
        return hist[:, -HORIZON:, :]

    __call__ = forward  # allow baseline(hist, static) like a Module


class SeasonalPersistenceForecaster:
    """'Tomorrow = same weekday last week': steps 0..95 of the 7-day history
    are exactly D-7."""

    def forward(self, hist: torch.Tensor,
                static: Optional[torch.Tensor] = None) -> torch.Tensor:
        return hist[:, :HORIZON, :]

    __call__ = forward
