import numpy as np


def compute_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def compute_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))


def compute_mape(predictions, targets, epsilon=1e-5):
    return np.mean(np.abs((predictions - targets) / (targets + epsilon)))


def compute_rmse(predictions, targets):
    return np.sqrt(compute_mse(predictions, targets))


class MetricAggregator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._mse_acc = 0
        self._mae_acc = 0
        self._mape_acc = 0
        self._count = 0
    
    def update(self, preds, targets, batch_size):
        self._mse_acc += compute_mse(preds, targets) * batch_size
        self._mae_acc += compute_mae(preds, targets) * batch_size
        self._mape_acc += compute_mape(preds, targets) * batch_size
        self._count += batch_size
    
    def compute(self):
        if self._count == 0:
            return {'rmse': 0, 'mae': 0, 'mape': 0}
        return {
            'rmse': np.sqrt(self._mse_acc / self._count),
            'mae': self._mae_acc / self._count,
            'mape': self._mape_acc / self._count
        }
