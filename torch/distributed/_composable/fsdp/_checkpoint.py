"""
https://github.com/fairinternal/xformers/blob/main/xformers/checkpoint.py
"""

from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, ContextManager, Dict, List

import torch

from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


def _get_default_policy(allow_list=None):
    _default_allow_list = [
        "xformers.efficient_attention_forward_cutlass.default",
        "xformers_flash.flash_fwd.default",
        "aten.addmm.default",
        "aten.mm.default",
    ]
    if allow_list is None:
        allow_list = _default_allow_list

    def _default_policy(func, *args, **kwargs):
        return str(func) in allow_list

    return _default_policy


class CachedTorchDispatchMode(_CachedTorchDispatchMode):
    def pop_from_storage(self, func, args, kwargs):
        if self.storage[func]:
            return self.storage[func].pop(0)
        return func(*args, **kwargs)


def selective_checkpoint_context_fn(policy_fn=None):
    """An activation checkpoint context_fn for selectively deciding what to
    store and what to recompute. Accepts a custom policy.

    Args:
        policy_fn(Union[List[Op], callable]): policy for deciding what to
            store (instead of recompute). If it's a function, it should
            be of form (func, *args, **kwargs) -> bool which indicates
            if func outputs with *args and **kwargs should be stored or not.
            Additionally, a list[Op] is also supported for easier cases.
            The op should be in the format `torch.ops.***`, where the `***`
            names of operators can be obtained with `list_operators`.
    """
    if policy_fn is None:
        policy_fn = _get_default_policy()
    elif isinstance(policy_fn, list):
        policy_fn = _get_default_policy(policy_fn)
    else:
        assert callable(policy_fn), "policy_fn should be None, list or a callable"

    temp_storage: Dict[Any, List[Any]] = defaultdict(list)
    # assumption: grad_mode doesn't change inside function
    caching_mode: ContextManager[None]
    if torch.is_grad_enabled():
        caching_mode = _CachingTorchDispatchMode(deepcopy(policy_fn), temp_storage)
    else:
        caching_mode = nullcontext()
    cached_mode = CachedTorchDispatchMode(deepcopy(policy_fn), temp_storage)

    return caching_mode, cached_mode
