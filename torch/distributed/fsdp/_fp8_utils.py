from typing import Callable, Tuple

import torch

from torch.distributed.utils import _p_assert


class SplitAndViewAsFloat8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unused_leaf: torch.Tensor,
        # TODO: Optionally unroll this `handle` into the lists needed.
        handle: "FlatParamHandle",
        post_backward_hook: Callable,  # hook(handle, flat_grad) -> None
    ) -> Tuple["Float8Tensor", ...]:
        assert unused_leaf.requires_grad
        ctx.unused_leaf = unused_leaf
        ctx.handle = handle
        ctx.post_backward_hook = post_backward_hook
        flat_param = handle.flat_param
        splits = torch.split(flat_param, flat_param._numels_with_padding, dim=0)
        idx = 0
        views_fp8 = []
        for split, is_padding in zip(splits, flat_param._is_padding_mask):
            if is_padding:
                continue
            views_fp8.append(split.view(flat_param._shapes[idx]))
            idx += 1
        assert len(views_fp8) == len(handle._amaxes)
        assert len(views_fp8) == len(handle._scales)
        views_float8 = tuple(
            handle._to_float8_from_fp8_fn(view_fp8, scale, handle._orig_param_dtype)
            for view_fp8, scale in zip(views_fp8, handle._scales)
        )
        return views_float8

    @staticmethod
    def backward(
        ctx,
        *grads_fp32: Tuple[torch.Tensor, ...],
    ):
        if len(grads_fp32) == 0:
            return torch.empty_like(ctx.unused_leaf), None, None, None, None
        handle = ctx.handle
        # if handle.rank == 0:
        #     print(f"SplitAndViewAsFloat8.backward!")
        flat_param = handle.flat_param
        # TODO: Handle custom reduce_dtype.
        # TODO: Can validate uniform dtype if desired.
        # Reconstruct an fp32 unsharded gradient including padding
        if handle.uses_sharded_strategy:
            flat_grad_numel = flat_param._padded_unsharded_size.numel()
        else:
            flat_grad_numel = flat_param._unpadded_unsharded_size.numel()
        flat_grad = torch.empty(
            (flat_grad_numel,),
            dtype=grads_fp32[0].dtype,
            device=handle.device,
        )
        idx = 0
        flat_grad_offset = 0
        for numel, is_padding in zip(
            flat_param._numels_with_padding, flat_param._is_padding_mask
        ):
            if is_padding:
                flat_grad_offset += numel
                continue
            _p_assert(
                idx < len(grads_fp32),
                f"Index of {idx} is out of bounds for {len(grads_fp32)} gradients",
            )
            flat_grad[flat_grad_offset : flat_grad_offset + numel].view_as(
                grads_fp32[idx]
            ).copy_(grads_fp32[idx])
            flat_grad_offset += numel
            idx += 1
        handle._autograd_computed_grad = flat_grad
        ctx.post_backward_hook()
        return torch.empty_like(ctx.unused_leaf), None, None
