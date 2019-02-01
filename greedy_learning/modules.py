from typing import Optional

import torch
from homura.utils.containers import Map
from torch import nn


class NaiveGreedyModule(nn.Module):
    def __init__(self, module: nn.Module, aux: nn.Module, tail: nn.Module):
        """
        naive implementation of greedy-learning
        :param module: nn.ModuleDict. module can be expressed as nn.Sequential
        :param aux: nn.ModuleDict.
        :param tail: nn.Module. The rest part
        """
        super(NaiveGreedyModule, self).__init__()
        if not (isinstance(module, nn.ModuleDict) or isinstance(aux, nn.ModuleDict)):
            raise RuntimeError
        if set(module.keys()) <= set(aux.keys()):
            raise RuntimeError
        self.module = module
        self.aux = aux
        self.tail = tail
        self.keys = list(self.module.keys())
        self.keys.append(None)

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


if __name__ == '__main__':
    from homura.vision.models.cifar import resnet20
    from utils import module_converter, Flatten
    from torch.nn import functional as F

    resnet = module_converter(resnet20(num_classes=10),
                              keys=["conv1", "bn1", "relu", "layer1", "layer2", "layer3"])
    aux = nn.ModuleDict({
        "conv1": nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Conv2d(16, 16, 1, bias=False), nn.AdaptiveAvgPool2d(1),
                               Flatten(), nn.Linear(16, 10)),
        "layer1": nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(16, 16, 1, bias=False), nn.AdaptiveAvgPool2d(1),
                                Flatten(), nn.Linear(16, 10)),
        "layer2": nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(32, 16, 1, bias=False), nn.AdaptiveAvgPool2d(1),
                                Flatten(), nn.Linear(16, 10)),
        "layer3": nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(64, 16, 1, bias=False), nn.AdaptiveAvgPool2d(1),
                                Flatten(), nn.Linear(16, 10)),
    })
    greedy = NaiveGreedyModule(resnet, aux=aux,
                               tail=nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(64, 10)))
    conv1_output = greedy(torch.randn(1, 3, 32, 32), "conv1")
    F.cross_entropy(conv1_output.pred, torch.tensor([1])).backward()
    layer1_output = greedy(conv1_output, "layer1")
    F.cross_entropy(layer1_output.pred, torch.tensor([1])).backward()
    final_output = greedy(layer1_output)
    F.cross_entropy(final_output.pred, torch.tensor([1])).backward()
