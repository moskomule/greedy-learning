from homura import optim, lr_scheduler
from homura.utils import trainer as _trainer, callbacks as _callbacks, reporter
from homura.utils.containers import Map
from homura.vision.data import cifar10_loaders
from homura.vision.models.cifar import resnet20
from torch import nn

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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            input.features.detach_()

        output_map.output = input.pred  # last one
        output_map.loss = output_map.pop(f"loss_{None}")
        return output_map


def greedy_loss_by_name(name):
    name = f"loss_{name}"

    @_callbacks.metric_callback_decorator(name=name)
    def f(data):
        return data[name]

    return f


if __name__ == '__main__':
    import miniargs
    from torch.nn import functional as F

    p = miniargs.ArgumentParser()
    p.add_int("--batch_size", default=128)
    args = p.parse()

    keys = ["conv1", "bn1", "relu", "layer1", "layer2", "layer3"]

    train_loader, test_loader = cifar10_loaders(args.batch_size)
    resnet = module_converter(resnet20(num_classes=10), keys=keys)
    aux = nn.ModuleDict({
        # 32x32
        "conv1": nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Conv2d(16, 16, 1, bias=False),
                               nn.Conv2d(16, 16, 1, bias=False), nn.AdaptiveAvgPool2d(1),
                               Flatten(), nn.Linear(16, 10)),
        # 32x32
        "layer1": nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Conv2d(16, 16, 1, bias=False),
                                nn.Conv2d(16, 16, 1, bias=False), nn.AdaptiveAvgPool2d(1),
                                Flatten(), nn.Linear(16, 10)),
        # 16x16
        "layer2": nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(32, 16, 1, bias=False),
                                nn.Conv2d(16, 16, 1, bias=False), nn.AdaptiveAvgPool2d(1),
                                Flatten(), nn.Linear(16, 10)),
        # 4x4
        "layer3": nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(64, 16, 1, bias=False),
                                nn.Conv2d(16, 16, 1, bias=False), nn.AdaptiveAvgPool2d(1),
                                Flatten(), nn.Linear(16, 10)),
    })
    greedy = NaiveGreedyModule(resnet, aux=aux,
                               tail=nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(64, 10)))
    tb = reporter.TensorboardReporter([_callbacks.LossCallback(), _callbacks.AccuracyCallback(),
                                       greedy_loss_by_name("conv1"),
                                       greedy_loss_by_name("layer1"),
                                       greedy_loss_by_name("layer2"),
                                       greedy_loss_by_name("layer3")],
                                      "../results")
    tb.enable_report_params()
    trainer = Trainer(greedy, optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4), F.cross_entropy, callbacks=tb,
                      scheduler=lr_scheduler.MultiStepLR([100, 150]))
    for _ in range(200):
        trainer.train(train_loader)
        trainer.test(test_loader)
