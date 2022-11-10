# Owner(s): ["oncall: distributed"]

import copy
import functools
import itertools

import sys
from typing import Any, Dict, List, Optional
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ExecOrderPolicy
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    AlwaysWrapNestedWrappedModule,
    CUDAInitMode,
    DummyDDP,
    FSDPInitMode,
    FSDPTest,
    MixtureOfExperts,
    NestedWrappedModule,
    NestedWrappedModuleWithDelay,
    subtest_name,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
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


class TestExecOrderPolicy(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_ddp_parity_for_nested_model(self):
        """
        Tests FSDP parity with DDP when using ``ExecOrderPolicy`` for a nested
        model.
        """
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            {},
            deterministic=True,
        )
        ddp_model = DDP(copy.deepcopy(model), device_ids=[self.rank])
        # Hand choose a size to group some but not all parameters together
        comm_size = int(5e2)
        fsdp_model = FSDP(
            model,
            process_group=self.process_group,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=ExecOrderPolicy(comm_size),
            use_orig_params=True,
        )
        self._test_ddp_parity(ddp_model, fsdp_model)

    def test_ddp_parity_for_transformer_with_shared_params(self):
        # TODO: Broken looks like due to shared parameter problems
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            {},
            deterministic=True,
        )
        ddp_model = DDP(copy.deepcopy(model), device_ids=[self.rank])
        # Hand choose a size to group some but not all parameters together
        comm_size = int(5e2)
        fsdp_model = FSDP(
            model,
            process_group=self.process_group,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=ExecOrderPolicy(comm_size),
            use_orig_params=True,
        )
        self._test_ddp_parity(ddp_model, fsdp_model)

    def _test_ddp_parity(self, ddp_model: DDP, fsdp_model: FSDP):
        LR = 1e-2
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)
        device = torch.device("cuda")
        for i in range(6):
            inp = fsdp_model.module.get_input(device)
            fsdp_out = fsdp_model(*inp)
            ddp_out = ddp_model(*inp)
            fsdp_loss = fsdp_model.module.get_loss(inp, fsdp_out)
            ddp_loss = ddp_model.module.get_loss(inp, ddp_out)
            fsdp_model.module.run_backward(fsdp_loss)
            ddp_model.module.run_backward(ddp_loss)
            ddp_optim.step()
            fsdp_optim.step()
            if i % 2 == 0:
                # Only check every other iteration to allow for some async
                # kernel execution across iterations
                with FSDP.summon_full_params(fsdp_model):
                    for (fsdp_param_name, fsdp_param), (
                        ddp_param_name,
                        ddp_param,
                    ) in zip(
                        fsdp_model.named_parameters(),
                        ddp_model.module.named_parameters(),
                    ):
                        self.assertEqual(fsdp_param_name, ddp_param_name)
                        self.assertEqual(fsdp_param, ddp_param)


if __name__ == "__main__":
    run_tests()
