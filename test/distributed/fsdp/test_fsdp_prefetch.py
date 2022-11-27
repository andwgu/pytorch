# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.utils.checkpoint import checkpoint

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 128
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class TestFSDPPrefetch(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _get_fsdp_model(self):
        model = CNN().cuda()
        module_wrap_policy = ModuleWrapPolicy(
            {nn.BatchNorm2d, nn.Sequential, nn.Linear},
        )
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=module_wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        )
        return fsdp_model

    def test_backward_prefetch(self):
        fsdp_model = self._get_fsdp_model()
        if self.rank == 0:
            print(fsdp_model)
        criterion = nn.MSELoss()
        batch_size = 2
        inp = torch.randn(size=(batch_size, 3, 224, 224), device="cuda")
        out = fsdp_model(inp)
        loss = criterion(out, torch.randn((batch_size, 10), device="cuda"))
        with torch.autograd.set_multithreading_enabled(False):
            loss.backward()


if __name__ == "__main__":
    run_tests()
