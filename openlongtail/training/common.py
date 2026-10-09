"""Distributed setup, fp32 master weights, the two-group AdamW and its LR schedule."""
from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


# ---- distributed -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistributedContext:
    """torchrun (RANK / WORLD_SIZE / LOCAL_RANK) or srun (SLURM_*) launch, one process per GPU."""
    for key, value in (('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1'), ('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')):
        os.environ.setdefault(key, value)
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        if 'MASTER_ADDR' not in os.environ:
            nodes = os.environ.get('SLURM_JOB_NODELIST')
            first_node = subprocess.check_output(['scontrol', 'show', 'hostnames', nodes], text=True).splitlines()[0] if nodes else None
            os.environ['MASTER_ADDR'] = first_node or '127.0.0.1'
        os.environ.setdefault('MASTER_PORT', '29500')
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def wrap_ddp(module: nn.Module, local_rank: int) -> DistributedDataParallel:
    return DistributedDataParallel(module, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
                                   find_unused_parameters=True, gradient_as_bucket_view=True)


# ---- optimisation ----------------------------------------------------------------------------------------
class MasterWeights:
    """fp32 copies of bf16 parameters. An AdamW update of ~lr is far below one bf16 ULP of most weights, so stepping
    the bf16 tensors directly would round most updates away; the optimizer steps these copies instead and the result
    is written back into the model after every step."""

    def __init__(self, model_params: Iterable[nn.Parameter]) -> None:
        self.model_params = list(model_params)
        self.master_params = [p if p.dtype == torch.float32 else nn.Parameter(p.detach().float().clone()) for p in self.model_params]

    def pairs(self):
        return zip(self.model_params, self.master_params)

    @torch.no_grad()
    def grads_to_master(self) -> None:
        for param, master in self.pairs():
            if master is param:
                continue
            if param.grad is None:
                master.grad = None
            elif master.grad is None:
                master.grad = param.grad.detach().to(master.dtype)
            else:
                master.grad.copy_(param.grad)

    @torch.no_grad()
    def master_to_model(self) -> None:
        for param, master in self.pairs():
            if master is not param:
                param.copy_(master)

    def zero_model_grads(self) -> None:
        for param, master in self.pairs():
            if master is not param:
                param.grad = None

    @torch.no_grad()
    def load(self, values: dict[int, torch.Tensor]) -> int:
        """Restore masters from full-precision tensors keyed by id(model parameter); returns how many were restored."""
        restored = 0
        for param, master in self.pairs():
            value = values.get(id(param))
            if value is not None:
                master.copy_(value.to(device=master.device, dtype=master.dtype))
                restored += 1
        self.master_to_model()
        return restored

    def by_param_id(self) -> dict[int, torch.Tensor]:
        return {id(param): master for param, master in self.pairs()}


class HybridOptimizer:
    """AdamW over fp32 masters, with separate hyper-parameters for the new modules and for the LoRA / attention weights."""

    def __init__(self, new_params: Iterable[nn.Parameter], slow_params: Iterable[nn.Parameter], new_lr: float, slow_lr: float,
                 new_weight_decay: float, slow_weight_decay: float = 0.0, betas: tuple[float, float] = (0.9, 0.95)) -> None:
        new_params, slow_params = list(new_params), list(slow_params)
        self.masters = MasterWeights(new_params + slow_params)
        new_masters, slow_masters = self.masters.master_params[:len(new_params)], self.masters.master_params[len(new_params):]
        self.optimized_params = new_masters + slow_masters
        self.new_optimizer = torch.optim.AdamW(new_masters, lr=new_lr, betas=betas, weight_decay=new_weight_decay) if new_masters else None
        self.lora_optimizer = torch.optim.AdamW(slow_masters, lr=slow_lr, betas=betas, weight_decay=slow_weight_decay)
        self._grads_on_master = False

    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [opt for opt in (self.new_optimizer, self.lora_optimizer) if opt is not None]

    def zero_grad(self) -> None:
        for opt in self.optimizers():
            opt.zero_grad(set_to_none=True)
        self.masters.zero_model_grads()
        self._grads_on_master = False

    def prepare_grads(self) -> list[nn.Parameter]:
        """Move this step's gradients onto the fp32 masters and return them (clip on these)."""
        if not self._grads_on_master:
            self.masters.grads_to_master()
            self._grads_on_master = True
        return self.optimized_params

    def step(self) -> None:
        self.prepare_grads()
        for opt in self.optimizers():
            opt.step()
        self.masters.master_to_model()
        self._grads_on_master = False

    def state_dict(self) -> dict[str, object]:
        return {'new_optimizer': self.new_optimizer.state_dict() if self.new_optimizer is not None else None,
                'lora_optimizer': self.lora_optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if self.new_optimizer is not None and state.get('new_optimizer') is not None:
            self.new_optimizer.load_state_dict(state['new_optimizer'])
        self.lora_optimizer.load_state_dict(state['lora_optimizer'])


class WarmupCosineLR:
    """Linear warm-up, then cosine decay to min_ratio of the base learning rates."""

    def __init__(self, optimizer: HybridOptimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.1) -> None:
        self.optimizer, self.total_steps, self.warmup_steps, self.min_ratio = optimizer, total_steps, warmup_steps, min_ratio
        self.base_lrs = [[group['lr'] for group in opt.param_groups] for opt in optimizer.optimizers()]
        self.step_idx = 0

    def _scale(self) -> float:
        if self.step_idx < self.warmup_steps:
            return float(self.step_idx + 1) / float(max(1, self.warmup_steps))
        progress = (self.step_idx - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    def step(self) -> None:
        scale = self._scale()
        for opt, base_lrs in zip(self.optimizer.optimizers(), self.base_lrs):
            for group, base_lr in zip(opt.param_groups, base_lrs):
                group['lr'] = base_lr * scale
        self.step_idx += 1
