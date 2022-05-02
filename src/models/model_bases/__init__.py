import torch.optim
from .lstm import LSTM

STR2BASE = {
    'lstm': LSTM
}

STR2OPTIM = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

__all__ = [
    'STR2BASE',
    'STR2OPTIM'
]