# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp.wrap import (
    _ExecOrderBasePolicy,
    _ExecOrderPolicy,
    ModuleWrapPolicy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestFSDPExecOrderPolicy(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_sibling_shared_params(self):
        """
        Tests the case of sibling shared parameters for ``fully_shard``. If the
        policy is not an execution order policy, then an error should be
        raised. Otherwise, the sibling shared parameters should be assigned to
        their lowest common ancestor module, and training should work.
        """
        d_vocab = 23
        d_model = 16

        class ModelWithSharedParams(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(d_vocab, d_model)
                self.lin = nn.Linear(d_model, d_model)
                self.seq = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
                )
                self.output_proj = nn.Linear(d_model, d_vocab)
                self.relu = nn.ReLU()
                # Share a parameter across siblings, where the LCA module is
                # this `ModelWithSharedParams` instance
                self.output_proj.weight = self.embed_tokens.weight

            def forward(self, x: torch.Tensor):
                z = self.embed_tokens(x)
                z = self.relu(self.lin(z))
                z = self.relu(self.seq(z))
                return self.output_proj(z)

        # Check that a non-execution order policy raises an error when there
        # are sibling shared parameters
        with self.assertRaisesRegex(
            RuntimeError,
            "Only parent-child shared parameters are supported. FSDP found a "
            "shared parameter between the two modules:",
        ):
            module_classes = {nn.Linear, nn.Embedding}
            fully_shard(
                nn.Sequential(
                    ModelWithSharedParams(),
                    nn.Linear(d_vocab, d_vocab),
                ),
                process_group=self.process_group,
                policy=ModuleWrapPolicy(module_classes),
                device_id=torch.cuda.current_device(),
            )

        # Check correctness when using an execution order policy
        composable_module = nn.Sequential(
            ModelWithSharedParams(), nn.Linear(d_vocab, d_vocab)
        )
        ddp_module = DDP(
            copy.deepcopy(composable_module).cuda(), device_ids=[self.rank]
        )
        fully_shard(
            composable_module,
            process_group=self.process_group,
            policy=_ExecOrderBasePolicy(),
            device_id=torch.cuda.current_device(),
        )

        # Check that the shared embedding/output projection weight is flattened
        # once, meaning that only one name should appear in the FQNs
        has_output_proj_weight = False
        has_embed_tokens_weight = False
        for handle in fully_shard.state(composable_module)._handles:
            has_output_proj_weight |= any(
                "output_proj.weight" in fqn for fqn in handle.flat_param._fqns
            )
            has_embed_tokens_weight |= any(
                "embed_tokens.weight" in fqn for fqn in handle.flat_param._fqns
            )
        self.assertEqual(has_output_proj_weight + has_embed_tokens_weight, 1)

        # Check that we can running a few training iterations without erroring
        ddp_optim = torch.optim.Adam(ddp_module.parameters(), lr=1e-2)
        composable_optim = torch.optim.Adam(composable_module.parameters(), lr=1e-2)
        for i in range(4):
            losses = []
            for (module, optim) in (
                (ddp_module, ddp_optim),
                (composable_module, composable_optim),
            ):
                optim.zero_grad(set_to_none=(i % 2 == 0))
                inp = torch.arange(12, device=torch.device("cuda")).view(6, 2)
                loss = module(inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_ddp_parity_for_nested_model(self):
        """
        Tests ``fully_shard`` parity with DDP when using ``_ExecOrderPolicy``
        for a nested model.
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
        fully_shard(
            model,
            process_group=self.process_group,
            policy=_ExecOrderPolicy(comm_size),
            device_id=torch.cuda.current_device(),
        )
        self._test_ddp_parity(ddp_model, model)

    @skip_if_lt_x_gpu(2)
    def test_ddp_parity_for_transformer_with_shared_params(self):
        """
        Tests ``fully_shard`` parity with DDP when using ``_ExecOrderPolicy``
        for a transformer model with shared parameters.
        """
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
        fully_shard(
            model,
            process_group=self.process_group,
            policy=_ExecOrderPolicy(comm_size),
            device_id=torch.cuda.current_device(),
        )
        self._test_ddp_parity(ddp_model, model)

    def _test_ddp_parity(self, ddp_model: DDP, fsdp_model: nn.Module):
        LR = 1e-2
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)
        device = torch.device("cuda")
        for i in range(6):
            losses = []
            inp = fsdp_model.get_input(device)
            for model, optim in ((ddp_model, ddp_optim), (fsdp_model, fsdp_optim)):
                optim.zero_grad(set_to_none=(i % 2 == 0))
                out = model(*inp)
                module = model.module if isinstance(model, DDP) else model
                loss = module.get_loss(inp, out)
                losses.append(loss)
                module.run_backward(loss)
                optim.step()
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
