from .models import CroplandMapper, FloodMapper
from .engine import train_model

STR2MODEL = {
    'cropland': CroplandMapper,
    'flood': FloodMapper
}

__all__ = [
    'STR2MODEL',
    'train_model',
    'CroplandMapper',
    'FloodMapper'
]