"""Optimizer-update EMA with exception-safe temporary parameter swapping."""

from __future__ import annotations

from contextlib import contextmanager

import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.9):
        self.decay = decay
        self.ema_parameters = [p.detach().clone() for p in parameters]

    @torch.no_grad()
    def update(self, parameters, completed_updates):
        decay = min((1 + completed_updates) / (10 + completed_updates), self.decay)
        for average, parameter in zip(self.ema_parameters, parameters, strict=True):
            average.add_((1 - decay) * (parameter - average))

    @contextmanager
    def apply(self, parameters):
        parameters = list(parameters)
        saved = [p.detach().cpu().clone() for p in parameters]
        try:
            with torch.no_grad():
                for parameter, average in zip(parameters, self.ema_parameters, strict=True):
                    parameter.copy_(average)
            yield
        finally:
            with torch.no_grad():
                for parameter, value in zip(parameters, saved, strict=True):
                    parameter.copy_(value)

    def state_dict(self):
        return {"decay": self.decay, "ema_parameters": self.ema_parameters}

    def load_state_dict(self, state):
        values = state["ema_parameters"]
        if len(values) != len(self.ema_parameters):
            raise ValueError("EMA parameter count differs from the current adapter")
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay differs from the checkpoint")
        for destination, value in zip(self.ema_parameters, values, strict=True):
            if destination.shape != value.shape:
                raise ValueError("EMA parameter shape differs from the current adapter")
            destination.copy_(value.to(destination))
