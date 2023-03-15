"""
torchrun --standalone --nproc_per_node=1 repro.py
torchrun --standalone --nproc_per_node=2 repro.py
"""

import os

from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


os.environ["TORCHDYNAMO_PRINT_GUARDS"] = "1"
NUM_ITERS = 3
LIN_DIM = int(1e3) + 3


def dist_setup():
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)


class OverArch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for i in range(10):
            setattr(self, f"_{i}", nn.Parameter(torch.randn((LIN_DIM, LIN_DIM), device="cuda")))
        module_list: List[nn.Module] = []
        for i in range(5):
            module_list.append(nn.Linear(LIN_DIM, LIN_DIM, device="cuda"))
        self._mod_list = nn.ModuleList(module_list)
        module_dict = {}
        for i in range(5):
            module_dict[f"_dict_{i}"] = nn.Linear(LIN_DIM, LIN_DIM, device="cuda")
        self._mod_dict = nn.ModuleDict(module_dict)
        self._dense_proj = DenseProj()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.to(self._0.dtype)
        for i in range(10):
            z = torch.nn.functional.linear(z, getattr(self, f"_{i}"))
            z = torch.nn.functional.relu(z)
        for i in range(5):
            z = self._mod_list[i](z)
            z = torch.nn.functional.relu(z)
        for i in range(5):
            z = self._mod_dict[f"_dict_{i}"](z)
            z = torch.nn.functional.relu(z)
        return z


class DenseProj(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for i in range(5):
            setattr(self, f"_{i}", nn.Parameter(torch.randn((LIN_DIM, LIN_DIM), device="cuda")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.to(self._0.dtype)
        for i in range(5):
            z = torch.nn.functional.linear(z, getattr(self, f"_{i}"))
            z = torch.nn.functional.relu(z)
        return z


class SparseArch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Parameter(torch.randn((LIN_DIM, LIN_DIM), device="cuda"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.to(self.emb.dtype)
        return z @ self.emb


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.over_arch = OverArch()
        self.sparse_arch = SparseArch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.sparse_arch(x)
        z = self.over_arch(z)
        return z


def init():
    torch.manual_seed(0)
    fsdp_kwargs = {"use_orig_params": True}
    model = Model()
    model.over_arch = FSDP(model.over_arch, **fsdp_kwargs)
    model.over_arch = torch.compile(model.over_arch)
    model = FSDP(
        model,
        **fsdp_kwargs,
        ignored_modules=[model.sparse_arch],
    )
    if dist.get_rank() == 0:
        print(model)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    return model, optim


def run(model, optim):
    for i in range(NUM_ITERS):
        if dist.get_rank() == 0:
            print(f"[Rank 0] iter {i}")
        optim.zero_grad(set_to_none=True)
        inp = torch.randn((16, LIN_DIM), device="cuda")
        out = model(inp)
        out.sum().backward()
        optim.step()


def main():
    assert torch.cuda.is_available(), f"Expects CUDA for FSDP communication"
    dist_setup()
    model, optim = init()
    run(model, optim)


if __name__ == "__main__":
    main()
