from typing import Iterable

import torch
from torch import nn


def module_converter(module: nn.Module, keys: Iterable[str]) -> nn.ModuleDict:
    base = nn.ModuleDict()
    for n, m in module.named_modules():
        if n in keys:
            base.update({n: m})

    return base


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input: torch.Tensor):
        return input.reshape(input.size(0), -1)
