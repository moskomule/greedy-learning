from typing import Optional

import torch
from homura.utils.containers import Map
from torch import nn


class NaiveGreedyModule(nn.Module):
    def __init__(self, module: nn.Module, aux: nn.Module):
        super(NaiveGreedyModule, self).__init__()
        if not (isinstance(module, nn.ModuleDict) or isinstance(aux, nn.ModuleDict)):
            raise RuntimeError
        if set(module.keys()) >= set(aux.keys()):
            raise RuntimeError
        self.module = module
        self.aux = aux

    def forward(self, input: torch.Tensor or Map, key: Optional[str] = None) -> torch.Tensor or Map:

        if self.training:
            x = input.features if isinstance(input, Map) else input
            for name, module in self.module.items():
                with torch.set_grad_enabled(name == key):
                    x = module(x)
                if name == key:
                    break
            pred = self.aux[key](x)
            return Map(pred=pred, features=x)
        else:
            x = input
            if key is not None:
                raise RuntimeWarning(f"key is given ({key}) but no effect!")
            for module in self.module.values():
                x = module(x)
            return x
