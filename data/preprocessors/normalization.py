import torch


class MinMaxNormalizer:
    def __init__(self):
        self._max = None
        self._min = None
    
    def fit(self, data):
        self._max = data.max()
        self._min = data.min()
        return self
    
    def transform(self, data):
        if self._max is None or self._min is None:
            raise ValueError("Normalizer not fitted")
        return (data - self._min) / (self._max - self._min + 1e-8)
    
    def inverse_transform(self, data):
        if self._max is None or self._min is None:
            raise ValueError("Normalizer not fitted")
        return data * (self._max - self._min + 1e-8) + self._min
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)
