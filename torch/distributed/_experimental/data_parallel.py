import string
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed._tensor.placement_types import _Partial
from torch.testing._internal.composite_compliance import is_view_fn
from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen, checkpoint

funcol = torch.ops.c10d_functional


def _fsdp_recomp_policy():
    def _custom_policy(mode, func, *args, **kwargs):
        return not is_view_fn(func) and func not in {
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            torch.ops._c10d_functional.wait_tensor.default,
        }

    return _custom_policy


class ReplicateComputation(nn.Module):
    def __init__(
        self, device_mesh: DeviceMesh, param_placements: Tuple[Placement, ...]
    ):
        super().__init__()
        self.device_mesh = device_mesh
        self.param_placements = param_placements
        self.compute_placements = [Replicate()] * self.device_mesh.ndim
        self.grad_placements = [_Partial(reduce_op="avg")] * self.device_mesh.ndim

    def forward(self, x: DTensor):
        return x.redistribute(placements=self.compute_placements).to_local(
            grad_placements=self.grad_placements
        )


class ToDtype(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        return x.to(self.dtype)


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    *,
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: Optional[torch.dtype] = None,
):
    if mode == "replicate":
        param_placements = (Replicate(),)
    elif mode == "fully_shard":
        param_placements = (Shard(0),)
    elif mode == "hybrid_shard":
        param_placements = (Replicate(), Shard(0))
        assert (
            device_mesh.ndim == 2
        ), f"HSDP requires 2D DeviceMesh but got {device_mesh}"
    else:
        raise ValueError(f"Unsupported mode {mode}")

    if reduce_dtype is None:
        reduce_dtype = param_dtype
    # Save modules and named parameters before iterating since parametrization
    # modifies them during registration
    modules_list = list(model.modules())
    for module in modules_list:
        named_params = dict(module.named_parameters(recurse=False))
        for param_name, param in named_params.items():
            # TODO: This should be a right inverse to support setting.
            module.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, device_mesh, param_placements)),
            )
            if param_dtype is not None and reduce_dtype == param_dtype:
                # Gradient reduction before cast back to fp32
                nn.utils.parametrize.register_parametrization(
                    module, param_name, ToDtype(param_dtype), unsafe=True
                )
            nn.utils.parametrize.register_parametrization(
                module,
                param_name,
                ReplicateComputation(device_mesh, param_placements),
                unsafe=True,
            )
            if param_dtype is not None and reduce_dtype == param.dtype:
                # Cast back to fp32 before gradient reduction
                nn.utils.parametrize.register_parametrization(
                    module, param_name, ToDtype(param_dtype), unsafe=True
                )

    if mode == "fully_shard" or mode == "hybrid_shard":

        def fsdp_policy():
            return _pt2_selective_checkpoint_context_fn_gen(_fsdp_recomp_policy())

        orig_forward = model.forward
        model.forward = lambda *args, **kwargs: checkpoint(
            orig_forward, *args, use_reentrant=False, context_fn=fsdp_policy, **kwargs
        )

    return model


# TODO: In practice, we want to run all preceding parametrizations for each
# parameter until we reach the `ReplicateComputation` parametrization and only
# replace that. If `ReplicateComputation` is not the last parametrization,
# existing infra does not seem to be able to continue with the remaining ones.
class BatchedAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *params: DTensor):
        ctx.device_mesh = params[0].device_mesh
        ctx.placements = params[0].placements
        for param in params[1:]:
            assert param.device_mesh == ctx.device_mesh
            assert param.placements == ctx.placements
        local_params = tuple(param.to_local() for param in params)
        return local_params

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        if any(grad is None for grad in grads):
            raise NotImplementedError()
        grad_sizes = [grad.size() for grad in grads]
        flat_grad = torch.cat(grad.view(-1) for grad in grads)
        del grads
        flat_grad = funcol.all_reduce(flat_grad, op="avg")
        # TODO: This split will force a wait on the ACT, which will prevent
        # overlap in eager mode.
        reduced_flat_grads = flat_grad.split(
            [grad_size.numel() for grad_size in grad_sizes]
        )
        reduced_grads = tuple(
            DTensor.from_local(grad.view(grad_size), ctx.device_mesh, ctx.placements)
            for grad, grad_size in zip(reduced_flat_grads, grad_sizes)
        )
        return tuple(reduced_grads)


def batch_all_reduces(modules: List[nn.Module], param_names: List[str]):
    distributed_params = [
        getattr(module, param_name) for module, param_name in zip(modules, param_names)
    ]
    local_params = BatchedAllReduce.apply(*distributed_params)
    for module, param_name, local_param in zip(modules, param_names, local_params):
        # parametrizations.weight.original -> weight
        parametrized_name = (
            param_name.replace("parametrizations.", "")
            .rstrip(string.digits)
            .replace(".original", "")
        )
        key = (id(module), parametrized_name)
        nn.utils.parametrize._cache[key] = local_param
