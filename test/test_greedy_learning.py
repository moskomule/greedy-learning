import torch
from homura.vision.models.cifar import resnet20
from torch import nn
from torch.nn import functional as F

from greedy_learning.modules import NaiveGreedyModule
from greedy_learning.utils import module_converter, Flatten


def test_naive():
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
