"""Rectified-flow noise schedule with Wan's timestep shift: x_sigma = (1 - sigma) x_0 + sigma eps, velocity eps - x_0."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowMatchScheduler:
    shift: float = 3.0

    def sigmas_for(self, t: torch.Tensor) -> torch.Tensor:
        return self.shift * t / (1.0 + (self.shift - 1.0) * t)

    def sample_training_sigma(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return self.sigmas_for(torch.rand(batch, device=device))

    def inference_sigmas(self, num_steps: int, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """num_steps + 1 noise levels from 1 down to 0."""
        return self.sigmas_for(torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype))
