"""Explicit parameter broadcast and one gradient average per optimizer update."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

    @classmethod
    def initialize(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("Model training requires CUDA; use `dvoc check` for CPU preflight")
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        if world > 1:
            dist.init_process_group(
                "nccl", rank=rank, world_size=world, device_id=device, timeout=timedelta(minutes=60)
            )
        return cls(rank, world, device)

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def broadcast_parameters(self, parameters):
        if self.world_size > 1:
            for parameter in parameters:
                dist.broadcast(parameter, src=0)

    def average_gradients(self, parameters):
        # Every rank must call every reduction in the same order. A missing local
        # gradient becomes zero, so a rank-dependent unused adapter cannot hang.
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if self.world_size > 1:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)

    def reduce_metrics(self, values, *, maximum=False):
        keys = sorted(values)
        tensor = torch.tensor(
            [float(values[k]) for k in keys], device=self.device, dtype=torch.float64
        )
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
            if not maximum:
                tensor /= self.world_size
        return dict(zip(keys, tensor.cpu().tolist()))

    def gather_objects(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
