# Owner(s): ["oncall: distributed"]

import functools
import sys

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(5, 5))
        self.p2 = nn.Parameter(torch.randn(5, 4))
        self.p3 = nn.Parameter(torch.randn(4, 3))
        self.relu = nn.ReLU()

    def forward(self, x):
        for p in (self.p1, self.p2, self.p3):
            x = x @ p
            x = self.relu(x)
        return x


class TestFSDPUseOrigParams(FSDPTest):
    def _get_optim(self, model):
        """Constructs three parameter groups, one for weights, one for biases,
        and one for everything else, each with different weight decay and
        learning rates."""
        param_groups = [
            {"params": [], "weight_decay": 0.1, "lr": 1e-2},
            {"params": [], "weight_decay": 0.01, "lr": 1e-3},
            {"params": []}
        ]
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                param_groups[0]["params"].append(param)
            elif "bias" in param_name:
                param_groups[1]["params"].append(param)
            else:
                param_groups[2]["params"].append(param)
        return torch.optim.Adam(param_groups, lr=5e-3)

    def _get_ddp_transformer(self, find_unused_params: bool = False):
        """Returns a transformer with shared parameters wrapped with DDP and
        a corresponding Adam optimizer."""
        group = dist.distributed_c10d._get_default_group()
        model = self._get_nonwrapped_model(group).cuda().eval()  # disable dropout
        ddp_model = DDP(
            model,
            device_ids=[self.rank],
            find_unused_parameters=find_unused_params,
        )
        ddp_optim = self._get_optim(ddp_model)
        return ddp_model, ddp_optim

    def _get_fsdp_transformer(self, init_optim_before_wrap: bool = False):
        """Returns a transformer with shared parameters wrapped with FSDP and
        a corresponding Adam optimizer."""
        group = dist.distributed_c10d._get_default_group()
        # Wrap several `Linear`s in the same FSDP instance to guarantee
        # different hyperparameter settings within that instance's single
        # `FlatParameter`
        config = {
            "auto_wrap_policy": functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={TransformerEncoderLayer, TransformerDecoderLayer},
            ),
            "use_orig_params": True,
        }
        if init_optim_before_wrap:
            model = self._get_nonwrapped_model(group).eval()  # disable dropout
            fsdp_optim = self._get_optim(model)
            fsdp_model = FSDP(model, group, **config)
        else:
            fsdp_model = self._get_wrapped_model(
                group,
                cuda_first=True,
                config=config,
            ).eval()  # disable dropout
            fsdp_optim = self._get_optim(fsdp_model)
        return fsdp_model, fsdp_optim

    def _check_train_parity(
        self,
        ddp_model: DDP,
        ddp_optim: torch.optim.Optimizer,
        fsdp_model: FSDP,
        fsdp_optim: torch.optim.Optimizer,
        num_iters: int = 5,
    ):
        """Checks training parity between DDP and FSDP."""
        device = torch.device("cuda")
        for _ in range(num_iters):
            iter_losses = []
            for model, optim in ((ddp_model, ddp_optim), (fsdp_model, fsdp_optim)):
                module = model.module
                optim.zero_grad()
                inp = module.get_input(device)
                output = model(*inp)
                loss = module.get_loss(inp, output).to(device)
                iter_losses.append(loss)
                module.run_backward(loss)
                optim.step()
            self.assertEqual(iter_losses[0].item(), iter_losses[1].item())
            iter_losses.clear()
        with FSDP.summon_full_params(fsdp_model):
            for p1, p2 in zip(ddp_model.parameters(), fsdp_model.parameters()):
                torch.testing.assert_close(p1, p2)

    @skip_if_lt_x_gpu(2)
    @parametrize("init_optim_before_wrap", [False, True])
    def test_diff_hyperparams(self, init_optim_before_wrap: bool):
        """
        Tests FSDP parity with DDP when using multiple parameter groups with
        different hyperparameter settings.

        Args:
            init_optim_before_wrap (bool): If ``True``, initializes the
                FSDP optimizer before wrapping the model with FSDP; otherwise,
                initializes the FSDP optimizer after wrapping the model with
                FSDP. We permit both forms of initialization to give users
                flexibility.
        """
        ddp_model, ddp_optim = self._get_ddp_transformer()
        fsdp_model, fsdp_optim = self._get_fsdp_transformer(init_optim_before_wrap)
        self._check_train_parity(ddp_model, ddp_optim, fsdp_model, fsdp_optim)

    @skip_if_lt_x_gpu(2)
    def test_diff_trainability(self):
        """
        Tests FSDP parity with DDP when using multiple parameter groups and
        freezing the parameters in one parameter group.
        """
        ddp_model, ddp_optim = self._get_ddp_transformer(True)
        fsdp_model, fsdp_optim = self._get_fsdp_transformer()
        # Freeze all biases (which are all in the same parameter group)
        for param_name, param in ddp_model.named_parameters():
            if "bias" in param_name:
                param.requires_grad_(False)
        for param_name, param in fsdp_model.named_parameters():
            if "bias" in param_name:
                param.requires_grad_(False)
        self._check_train_parity(ddp_model, ddp_optim, fsdp_model, fsdp_optim)


instantiate_parametrized_tests(TestFSDPUseOrigParams)

if __name__ == "__main__":
    run_tests()
