import torch
import random
import numpy as np


class ConfigRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.seed = 2021
            self.datapath = './data'
            self._initialized = True
            self._apply_seed()
    
    def _apply_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True


def get_config():
    return ConfigRegistry()
