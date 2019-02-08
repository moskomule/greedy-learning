from collections import OrderedDict

from homura import optim, lr_scheduler
from homura.utils import trainer as _trainer, callbacks as _callbacks, reporter
from homura.utils.containers import Map
from homura.vision.data import cifar10_loaders
from homura.vision.models.cifar.resnet import resnet56
from torch import nn
from tqdm import trange

from modules import NaiveGreedyModule
from utils import module_converter, Flatten


class Trainer(_trainer.TrainerBase):
    def __init__(self, model, optimizer, loss_f, callbacks,
                 scheduler=None):
        super(Trainer, self).__init__(model, optimizer, loss_f, callbacks=callbacks, scheduler=scheduler)
        self.keys = list(self.model.aux.keys())
        self.keys.append(None)

    def iteration(self, data):
        input, target = data
        if self.is_train:
            output_map = self.train_iteration(input, target)
        else:
            output = self.model(input)
            loss = self.loss_f(output, target)
            output_map = Map(output=output, loss=loss,
                             # todo: fix homura
                             loss_conv1=0, loss_layer1=0, loss_layer2=0, loss_layer3=0)
        return output_map

    def train_iteration(self, input, target):
        output_map = Map()
        for key in self.keys:
            input = self.model(input, key)
            loss = self.loss_f(input.pred, target)
            output_map[f"loss_{key}"] = loss

            loss.backward()
            input.features.detach_()
            input.features.clone()

        self.optimizer.step()
        self.optimizer.zero_grad()

        output_map.output = input.pred  # last one
        output_map.loss = output_map.pop(f"loss_{None}")
        return output_map


def greedy_loss_by_name(name):
    name = f"loss_{name}"

    @_callbacks.metric_callback_decorator(name=name)
    def f(data):
        return data[name]

    return f


def generate_aux(input_size: int, input_features: int, num_classes: int,
                 num_fully_conv: int = 3, num_fully_connected: int = 3):
    base = [nn.AdaptiveAvgPool2d(input_size // 4)]
    base += [nn.Sequential(nn.Conv2d(input_features, input_features, 1, bias=False), nn.ReLU())
             for _ in range(num_fully_conv)]
    base += [nn.AdaptiveAvgPool2d(2), Flatten()]
    base += [nn.Sequential(nn.Linear(4 * input_features if i == 0 else 16, 16),
                           nn.ReLU())
             for i in range(num_fully_connected - 1)]
    base += [nn.Linear(4 * input_features if num_fully_connected == 1 else 16, num_classes)]
    return nn.Sequential(*base)


if __name__ == '__main__':
    import miniargs
    from torch.nn import functional as F

    p = miniargs.ArgumentParser()
    p.add_int("--batch_size", default=128)
    p.add_int("--epochs", default=300)
    p.add_str("--optimizer", choices=["sgd", "adam"])
    p.add_float("--lr", default=1e-2)
    p.add_multi_str("--group", default=["conv1", "layer1", "layer2", "layer3"])
    p.add_int("--step", default=50)
    p.add_int("--num_convs", default=3)
    p.add_int("--num_fcs", default=3)
    args = p.parse()

    optimizer = {"adam": optim.Adam(lr=3e-4, weight_decay=1e-4),
                 "sgd": optim.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-4)}[args.optimizer]

    train_loader, test_loader = cifar10_loaders(args.batch_size)
    resnet = module_converter(resnet56(num_classes=10), keys=["conv1", "bn1", "relu", "layer1", "layer2", "layer3"])
    aux = nn.ModuleDict(OrderedDict({k: v for k, v in {
        # 32x32
        "conv1": generate_aux(32, 16, 10, args.num_convs, args.num_fcs),
        # 32x32
        "layer1": generate_aux(32, 16, 10, args.num_convs, args.num_fcs),
        # 16x16
        "layer2": generate_aux(16, 32, 10, args.num_convs, args.num_fcs),
        # 8x8
        "layer3": generate_aux(8, 64, 10, args.num_convs, args.num_fcs),
    }.items() if k in args.group}))
    model = NaiveGreedyModule(resnet, aux=aux,
                              tail=nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(64, 10)))

    # import torch
    # from homura.debug import module_debugger as simple_debugger
    # simple_debugger(model, (torch.randn(1, 3, 32, 32), "layer1"), target=torch.tensor([1]),
    #                 loss=lambda x, y: F.cross_entropy(x.pred, y))

    print(args)
    # print(model)
    greedy_loss = [greedy_loss_by_name(name) for name in args.group]
    tb = reporter.TensorboardReporter([_callbacks.LossCallback(), _callbacks.AccuracyCallback()] + greedy_loss,
                                      "../results")
    # tb.enable_report_params()
    trainer = Trainer(model, optimizer, F.cross_entropy, callbacks=tb,
                      scheduler=lr_scheduler.StepLR(args.step, 0.2) if args.optimizer == "sgd" else None)
    for _ in trange(args.epochs, ncols=80):
        trainer.train(train_loader)
        trainer.test(test_loader)
