"""Training of the OpenLongTail model on both tasks. Each step is a cross step (stage 1: the two cross views from the depth
warp) with probability --cross-prob, otherwise a rear step (stage 2: the three rear views from the ground prior, with
lane-marking pixels of the ground-truth rear views up-weighted); the task is drawn from the step number, so all ranks run
the same task. Conditions are jittered and occasionally dropped, the target views get independent noise levels 30% of
the time and one of them is sometimes given clean, since at inference the conditions come from estimated depth and
generated views.

Trained: the LoRA of every self-attention, the base weights of the main blocks' self-attention and the camera / pose
embeddings, with fp32 master weights. Starts from the released checkpoint by default (--init-from none: from Wan2.1-VACE
with fresh LoRA and embeddings).

    torchrun --nproc_per_node 8 -m openlongtail.training.train --latent-cache-root data/openlongtail_cache --output-dir outputs/train

Resumes automatically from the newest checkpoint in --output-dir.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from openlongtail.config import MODEL, MODEL_CKPT, WAN21_VACE_DIR
from openlongtail.data.latent_cache import SurroundClipDataset, collate
from openlongtail.flow import FlowMatchScheduler
from openlongtail.models.surround import TASK_VIEWS, SurroundDiT, build_model, load_weights, prepare_inputs, run_model
from openlongtail.training.common import HybridOptimizer, WarmupCosineLR, setup_distributed, wrap_ddp

CHECKPOINT_FILES = ('trainable.pt', 'optimizer.pt', 'metadata.json')
TASK_SEED = 20260928


def task_for_step(step: int, cross_prob: float) -> str:
    return 'cross' if random.Random(TASK_SEED * 1_000_003 + step).random() < cross_prob else 'rear'


def _target_noise(z_all: torch.Tensor, views: tuple[int, ...], flow: FlowMatchScheduler, device, independent_sigma_prob: float, given):
    """Noisy targets, per-view sigmas (0 = given clean) and the velocity target."""
    B, V = z_all.shape[0], len(views)
    x0 = z_all[:, list(views)]
    sigma = flow.sample_training_sigma(B, device)[:, None].expand(B, V).clone()
    independent = torch.rand(B, device=device) < independent_sigma_prob
    if independent.any():
        sigma[independent] = flow.sample_training_sigma(int(independent.sum()) * V, device).view(-1, V)
    given(sigma)
    eps = torch.randn_like(x0)
    s = sigma.to(x0.dtype).view(B, V, 1, 1, 1, 1)
    return (1 - s) * x0 + s * eps, sigma, eps - x0


def _jitter(latents: torch.Tensor, visibility: torch.Tensor, device, noise_prob: float, noise_min: float, noise_max: float, drop_prob: float):
    """Noise the condition latents of some samples and drop the condition of others."""
    B = latents.shape[0]
    noisy = (torch.rand(B, device=device) < noise_prob).view(B, 1, 1, 1, 1, 1)
    std = torch.empty(B, 1, 1, 1, 1, 1, device=device).uniform_(noise_min, noise_max)
    latents = torch.where(noisy, latents + std.to(latents.dtype) * torch.randn_like(latents), latents)
    drop = (torch.rand(B, device=device) < drop_prob).view(B, 1, 1, 1, 1, 1)
    return torch.where(drop, torch.zeros_like(latents), latents), torch.where(drop, torch.zeros_like(visibility), visibility), noisy, drop


def training_step_cross(batch: dict, model, flow: FlowMatchScheduler, device: torch.device, given_prob: float = 0.1) -> tuple[torch.Tensor, dict]:
    z_all = batch['z_all'].to(device=device, dtype=torch.bfloat16)
    B, V = z_all.shape[0], len(TASK_VIEWS['cross'])

    def given(sigma):                                          # one cross view given clean, the other generated
        mask = torch.rand(B, device=device) < given_prob
        if mask.any():
            which = torch.randint(0, V, (B,), device=device)
            sigma[mask, which[mask]] = 0.0
    xt, sigma, v_target = _target_noise(z_all, TASK_VIEWS['cross'], flow, device, 0.3, given)
    warp_latents = batch['warp_latents'].to(device=device, dtype=torch.bfloat16)
    warp_visibility = batch['warp_visibility'].to(device=device, dtype=torch.bfloat16)
    warp_latents, warp_visibility, noisy, drop = _jitter(warp_latents, warp_visibility, device, 0.3, 0.02, 0.15, 0.1)
    inputs = prepare_inputs('cross', z_all[:, 0], batch['K'].to(device).float(), batch['E'].to(device).float(),
                            batch['T_anchor_front'].to(device).float(), warp_latents, warp_visibility)
    v_pred = run_model(model, 'cross', xt, sigma, batch['text_emb'].to(device=device, dtype=torch.bfloat16), inputs)
    per_view = ((v_pred.float() - v_target.float()) ** 2).mean(dim=(2, 3, 4, 5))
    valid = (sigma > 0).float()                                # a given (clean) view carries no loss
    loss = (per_view * valid).sum() / valid.sum().clamp(min=1.0)
    metrics = {'loss': float(loss.detach()), 'sigma_mean': float(sigma.mean()), 'cond_noised': float(noisy.float().mean()),
               'cond_dropped': float(drop.float().mean()), 'warp_visibility': float(warp_visibility.float().mean())}
    for i, name in enumerate(('cross_left', 'cross_right')):
        if valid[:, i].sum() > 0:
            metrics[f'flow_{name}'] = float((per_view[:, i] * valid[:, i]).sum().detach() / valid[:, i].sum())
    return loss, metrics


def training_step_rear(batch: dict, model, flow: FlowMatchScheduler, device: torch.device, lane_weight: float = 4.0,
                       given_prob: float = 0.15) -> tuple[torch.Tensor, dict]:
    z_all = batch['z_all'].to(device=device, dtype=torch.bfloat16)
    B = z_all.shape[0]

    def given(sigma):                                          # rear-left or rear-right given clean, the others generated
        mask = torch.rand(B, device=device) < given_prob
        if mask.any():
            side = torch.where(torch.rand(B, device=device) < 0.5, 0, 2)
            sigma[mask, side[mask]] = 0.0
    xt, sigma, v_target = _target_noise(z_all, TASK_VIEWS['rear'], flow, device, 0.3, given)
    ground_latents = batch['ground_latents'].to(device=device, dtype=torch.bfloat16)
    ground_visibility = batch['ground_visibility'].to(device=device, dtype=torch.bfloat16)
    ground_latents, ground_visibility, noisy, drop = _jitter(ground_latents, ground_visibility, device, 0.3, 0.02, 0.15, 0.1)
    inputs = prepare_inputs('rear', z_all[:, 0], batch['K'].to(device).float(), batch['E'].to(device).float(),
                            batch['T_anchor_front'].to(device).float(), ground_latents, ground_visibility)
    v_pred = run_model(model, 'rear', xt, sigma, batch['text_emb'].to(device=device, dtype=torch.bfloat16), inputs)
    err = (v_pred.float() - v_target.float()) ** 2                                 # (B,3,16,T,H,W)
    w = 1.0 + lane_weight * batch['rear_lane'].to(device=device, dtype=torch.float32)   # (B,3,1,T,H,W)
    per_view = (err * w).sum(dim=(2, 3, 4, 5)) / (w.sum(dim=(2, 3, 4, 5)) * err.shape[2])
    valid = (sigma > 0).float()
    loss = (per_view * valid).sum() / valid.sum().clamp(min=1.0)
    metrics = {'loss': float(loss.detach()), 'sigma_mean': float(sigma.mean()), 'cond_noised': float(noisy.float().mean()),
               'cond_dropped': float(drop.float().mean()), 'lane_frac': float(batch['rear_lane'].float().mean())}
    for i, name in enumerate(('rear_left', 'rear_tele', 'rear_right')):
        if valid[:, i].sum() > 0:
            metrics[f'flow_{name}'] = float((per_view[:, i] * valid[:, i]).sum().detach() / valid[:, i].sum())
    return loss, metrics


def trainable_state(model: SurroundDiT, optimizer: HybridOptimizer) -> dict[str, torch.Tensor]:
    masters = optimizer.masters.by_param_id()
    return {n: masters.get(id(p), p).detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def complete_steps(output_dir: Path) -> list[int]:
    return sorted(int(d.name[5:]) for d in output_dir.glob('step_*') if all((d / f).exists() for f in CHECKPOINT_FILES))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--latent-cache-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--init-from', type=str, default=str(MODEL_CKPT), help='checkpoint dir to start from, or "none"')
    ap.add_argument('--wan21-vace-dir', type=Path, default=WAN21_VACE_DIR)
    ap.add_argument('--num-steps', type=int, default=40000)
    ap.add_argument('--warmup-steps', type=int, default=200)
    ap.add_argument('--save-every', type=int, default=500)
    ap.add_argument('--new-module-lr', type=float, default=5e-5, help='camera / pose embeddings')
    ap.add_argument('--slow-lr', type=float, default=1e-5, help='LoRA and self-attention weights')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--cross-prob', type=float, default=0.5, help='share of the steps on the cross-view task')
    ap.add_argument('--lane-weight', type=float, default=4.0, help='rear task: extra loss weight of lane-marking pixels')
    ap.add_argument('--num-workers', type=int, default=3)
    ap.add_argument('--index-name', type=str, default='index.jsonl')
    ap.add_argument('--max-items', type=int, default=None)
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    ctx = setup_distributed()
    model = build_model(ctx.device, a.wan21_vace_dir)
    new, slow = model.trainable_parameters()
    model.requires_grad_(False)
    for p in new + slow:
        p.requires_grad = True
    model.train()
    model.enable_gradient_checkpointing()
    optimizer = HybridOptimizer(new, slow, new_lr=a.new_module_lr, slow_lr=a.slow_lr, new_weight_decay=0.0)
    lr_schedule = WarmupCosineLR(optimizer, a.num_steps, a.warmup_steps)   # before a resume overwrites the lrs with scheduled ones
    a.output_dir.mkdir(parents=True, exist_ok=True)
    steps = complete_steps(a.output_dir)
    resumed = steps[-1] if steps else 0
    if ctx.is_distributed:                                   # every rank resumes from the step rank 0 saw
        t = torch.tensor([resumed], device=ctx.device)
        dist.broadcast(t, src=0)
        resumed = int(t.item())
    start = 1
    init = a.output_dir / f'step_{resumed:08d}' if resumed else (None if a.init_from.lower() == 'none' else Path(a.init_from))
    if init is not None:
        state = load_weights(model, init)
        by_name = dict(model.named_parameters())
        optimizer.masters.load({id(by_name[n]): t for n, t in state.items() if n in by_name})
    if resumed:
        d = a.output_dir / f'step_{resumed:08d}'
        optimizer.load_state_dict(torch.load(d / 'optimizer.pt', map_location='cpu', weights_only=True))
        start = json.loads((d / 'metadata.json').read_text())['step'] + 1
    if start > a.num_steps:
        return
    dataset = SurroundClipDataset(a.latent_cache_root, index_name=a.index_name, max_items=a.max_items)
    sampler = DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank) if ctx.is_distributed else None
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=sampler is None, num_workers=a.num_workers, collate_fn=collate,
                        persistent_workers=a.num_workers > 0)
    net = wrap_ddp(model, ctx.local_rank) if ctx.is_distributed else model
    for _ in range(start - 1):
        lr_schedule.step()
    if ctx.rank == 0:
        print(f'[train] {len(dataset)} clips, new {sum(p.numel() for p in new) / 1e6:.1f}M + slow {sum(p.numel() for p in slow) / 1e6:.1f}M '
              f'parameters, steps {start}..{a.num_steps}, ' + (f'resumed from step {resumed}' if resumed else f'initialised from {init}'), flush=True)
    flow = FlowMatchScheduler(MODEL.flow_shift)
    epoch = start - 1                                        # a resumed run draws a fresh permutation instead of replaying epoch 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    it = iter(loader)
    for step in range(start, a.num_steps + 1):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(loader)
            batch = next(it)
        task = task_for_step(step, a.cross_prob)
        optimizer.zero_grad()
        if task == 'cross':
            loss, metrics = training_step_cross(batch, net, flow, ctx.device)
        else:
            loss, metrics = training_step_rear(batch, net, flow, ctx.device, lane_weight=a.lane_weight)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.prepare_grads(), a.grad_clip)
        optimizer.step()
        lr_schedule.step()
        if ctx.rank == 0:
            metrics.update(step=step, task=task, grad_norm=float(grad_norm), step_sec=time.perf_counter() - t0,
                           peak_mem_gb=torch.cuda.max_memory_allocated(ctx.device) / 1e9)
            with (a.output_dir / 'train_log.jsonl').open('a') as h:
                h.write(json.dumps(metrics) + '\n')
            if step % a.save_every == 0 or step == a.num_steps:
                d, tmp = a.output_dir / f'step_{step:08d}', a.output_dir / f'.saving_{step:08d}'
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                torch.save(trainable_state(model, optimizer), tmp / 'trainable.pt')
                torch.save(optimizer.state_dict(), tmp / 'optimizer.pt')
                metadata = {'step': step, 'cross_prob': a.cross_prob, 'lane_weight': a.lane_weight}
                (tmp / 'metadata.json').write_text(json.dumps(metadata))
                shutil.rmtree(d, ignore_errors=True)
                tmp.rename(d)                                # a step directory only ever exists complete
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
