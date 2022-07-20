"""
Learning about saved tensor hooks and non-reentrant activation checkpointing.
"""
import functools
import gc
import sys
import pdb

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
)
from torch.utils.checkpoint import checkpoint

BATCH_SIZE = 4
IN_DIM = 7
OUT_DIM = 5

def get_tensors(only_cuda=False, omit_objs=[]):
    tensors = {}

    def add_tensor(obj):
        if torch.is_tensor(obj):
            tensor = obj
        elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
            tensor = obj.data
        else:
            return

        if (only_cuda and tensor.is_cuda) or (not only_cuda):
            tensors[id(tensor)] = tensor

    for obj in gc.get_objects():
        try:
            # Add the obj if it is a tensor.
            add_tensor(obj)
            # Some tensors are "saved & hidden" for the backward pass.
            if hasattr(obj, 'saved_tensors') and (id(obj) not in omit_objs):
                for tensor_obj in obj.saved_tensors:
                    add_tensor(tensor_obj)
        except Exception as ex:
            pass
    return tensors.values()  # return a list of detected tensors


def grad_hook(name, *args, **kwargs):
    print(f"grad hook for {name}!")
    pdb.set_trace()


class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_dim, out_dim))
        self.gelu = torch.nn.GELU()

    def forward(self, inp):
        """
        Args:
            inp (Tensor): (BATCH_SIZE, IN_DIM).

        Returns:
            Tensor: (BATCH_SIZE, OUT_DIM)
        """
        print(f"Model.forward()!")
        lin_out = inp @ self.weight    # activation: `inp`
        lin_out.register_hook(functools.partial(grad_hook, "lin_out"))
        print(f"[lin_out] shape={lin_out.shape} id={id(lin_out)} refcount={sys.getrefcount(lin_out)-1}")
        gelu_out = self.gelu(lin_out)  # activation: `lin_out`
        return gelu_out


# torch.cuda.set_device(0)
# model = Model(IN_DIM, OUT_DIM).cuda()
# model = CheckpointWrapper(model, CheckpointImpl.NO_REENTRANT)
model = torch.nn.Sequential(
    CheckpointWrapper(Model(IN_DIM, OUT_DIM), CheckpointImpl.NO_REENTRANT),
    CheckpointWrapper(Model(OUT_DIM, OUT_DIM), CheckpointImpl.NO_REENTRANT),
)
inp = torch.randn((BATCH_SIZE, IN_DIM), device=torch.device("cpu"))
print(" Init ".center(80, '*'))
print(f"[inp]    shape={inp.shape} id={id(inp)}")
# print(f"[weight] shape={model.weight.shape} id={id(model.weight)}")
print(f"[weight] shape={model[0].weight.shape} id={id(model[0].weight)}")
print(" Forward ".center(80, '*'))
out = model(inp).sum()
out.register_hook(functools.partial(grad_hook, "out"))
# model.weight.register_hook(functools.partial(grad_hook, "weight"))
model[0].weight.register_hook(functools.partial(grad_hook, "0.weight"))
model[1].weight.register_hook(functools.partial(grad_hook, "1.weight"))
for tensor in get_tensors():
    print(f"shape={tensor.shape} id={id(tensor)}")
print(" Backward ".center(80, '*'))

out.backward()
# for tensor in get_tensors():
#     print(f"shape={tensor.shape} id={id(tensor)}")

