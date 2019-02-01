# greedy-learning [WIP]

To catch up recent progress on greedy (layer-wise) learning

```python
from greedy_learning.modules import NaiveGreedyModule
from greedy_learning.utils import module_converter, Flatten
from homura.vision.models.cifar import resnet20
from torch import nn

# convert module to ModuleDict
resnet = module_converter(resnet20(num_classes=10),
                          keys=["conv1", "bn1", "relu" "layer1", "layer2", "layer3"],
                          )
aux = nn.ModuleDict({
    "conv1": nn.Sequential(nn.AdaptiveAvgPool2d(2), Flatten(), nn.Linear(4*16, 10)),
    "layer1": nn.Sequential(nn.AdaptiveAvgPool2d(2), Flatten(), nn.Linear(4*16, 10)),
    "layer2": nn.Sequential(nn.AdaptiveAvgPool2d(2), Flatten(), nn.Linear(4*32, 10)),
    "layer3": nn.Sequential(nn.AdaptiveAvgPool2d(2), Flatten(), nn.Linear(4*64, 10)),
})
# aux should have the same keys with modules
greedy = NaiveGreedyModule(resnet, aux=aux, tail=nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(64, 10)))
greedy.train()

# training phase
# update 1st module
module1_output = greedy(input, "conv1")
F.cross_entropy(module1_output.pred, target).backward()

# update 2nd module
module2_output = greedy(module1_output, "layer1")
F.cross_entropy(module2_output.pred, target).backward()
...
# update the tail
output = greedy(module3_output)
F.cross_entropy(output, target).backward()
```

## requirements

* Python 3.7
* homura `pip install -U git+https://github.com/moskomule/homura`