import torch.nn as nn
import torch.nn.init as init


def apply_kaiming_initialization(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0)


class InitializationStrategy:
    @staticmethod
    def apply(model, strategy='kaiming'):
        if strategy == 'kaiming':
            model.apply(apply_kaiming_initialization)
        return model
