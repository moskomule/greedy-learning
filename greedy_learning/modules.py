from collections import deque
from typing import Optional

import torch
from homura.utils.containers import Map
from torch import nn


class GreedyModuleBase(nn.Module):
    def __init__(self, module: nn.Module, aux: nn.Module, tail: nn.Module):
        """
        naive implementation of model-learning
        :param module: nn.ModuleDict. module can be expressed as nn.Sequential
        :param aux: nn.ModuleDict.
        :param tail: nn.Module. The rest part
        """
        super(GreedyModuleBase, self).__init__()
        if not (isinstance(module, nn.ModuleDict) or isinstance(aux, nn.ModuleDict)):
            raise RuntimeError
        if set(module.keys()) <= set(aux.keys()):
            raise RuntimeError
        self.module = module
        self.aux = aux
        self.tail = tail
        self.keys = list(self.module.keys())
        self.keys.append(None)
        self.initialize()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class NaiveGreedyModule(GreedyModuleBase):
    def __init__(self, module: nn.Module, aux: nn.Module, tail: nn.Module):
        """
        naive implementation of model-learning
        :param module: nn.ModuleDict. module can be expressed as nn.Sequential
        :param aux: nn.ModuleDict.
        :param tail: nn.Module. The rest part
        """
        super(NaiveGreedyModule, self).__init__(module, aux, tail)

    def forward(self, input: torch.Tensor or Map, key: Optional[str] = None) -> torch.Tensor or Map:
        if self.training:
            return self.forward_train(input, key)
        else:
            return self.forward_val(input)

    def forward_val(self, input):
        x = input
        for module in self.module.values():
            x = module(x)
        return self.tail(x)

    def forward_train(self, input: torch.Tensor or Map, key: Optional[str]):
        x, start = (input.features, input.start) if isinstance(input, Map) else (input, 0)
        for _key in self.keys[start:-1]:
            with torch.set_grad_enabled(_key == key):
                x = self.module[_key](x)
            if _key == key:
                break

        if key is None:
            pred = self.tail(x)
        elif key in self.module.keys():
            pred = self.aux[key](x)
        else:
            raise RuntimeError(f"got {key}")
        return Map(pred=pred, features=x, start=self.keys.index(key) + 1)

    def simplify(self):
        base = [module for module in self.module.values()] + [self.tail]
        return nn.Sequential(*base)


class ReplayBufferGreedyModule(GreedyModuleBase):
    def __init__(self, module: nn.Module, aux: nn.Module, tail: nn.Module,
                 buffer_size: int = 50):
        super(ReplayBufferGreedyModule, self).__init__(module, aux, tail)
        self.replay_buffer = {k: deque(maxlen=buffer_size) for k in self.keys}

    def forward(self, input: torch.Tensor or Map, straight: bool = True):
        if not self.training or straight:
            x = input
            for module in self.module.values():
                x = module(x)
            return self.tail(x)
        else:
            pass
