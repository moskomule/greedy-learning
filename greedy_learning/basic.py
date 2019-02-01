from homura.vision.models.cifar import resnet20
from torch import nn

from modules import NaiveGreedyModule
from utils import module_converter, Flatten


def main():
    resnet = module_converter(resnet20(num_classes=10),
                              keys=["conv1", "bn1", "relu" "layer1", "layer2", "layer3"])
    aux = nn.ModuleDict({
        "conv1": nn.Sequential(nn.AdaptiveAvgPool2d(2), Flatten(), nn.Linear(2 * 2 * 16, 10)),
        "layer1": nn.Sequential(nn.AdaptiveAvgPool2d(2), Flatten(), nn.Linear(2 * 2 * 16, 10)),
        "layer2": nn.Sequential(nn.AdaptiveAvgPool2d(2), Flatten(), nn.Linear(2 * 2 * 32, 10)),
        "layer3": nn.Sequential(nn.AdaptiveAvgPool2d(2), Flatten(), nn.Linear(2 * 2 * 64, 10)),
    })
    greedy = NaiveGreedyModule(resnet, aux=aux)


if __name__ == '__main__':
    main()
