from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass
class BoundingBox:

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

def combine_bounding_boxes(names: Union[str, list[str]]) -> BoundingBox:
    """
    Combines multiple countries' BoundingBoxes into a larger BoundingBox
    
    Args:
        names: one of:
            - a predefined region name in: 
                ['SSA', 'East Africa', 'West Africa & Chad', 'South Africa']
            - a list of (contiguous) country names in:
                ['Burkina Faso', 'Chad', 'Ethiopia', 'Kenya', 'Malawi',
                'Mali', 'Mozambique', 'Niger', 'Nigeria', 'Somalia', 
                'South Sudan', 'Sudan', 'Uganda', 'Zambia', 'Zimbabwe']
    """

    if names in ['SSA', 'East Africa', 'South Africa', 'West Africa & Chad']:
        bboxes = [STR2BB[name] for name in REGIONS[names]]
    elif all(name in STR2BB.keys() for name in names):
        bboxes = [STR2BB[name] for name in names]
    else:
        raise ValueError(f"names should be one of {REGIONS.keys()} or a list composed of {STR2BB.keys()}.")

    min_lon = np.min([bbox.min_lon for bbox in bboxes])
    min_lat = np.min([bbox.min_lat for bbox in bboxes])
    max_lon = np.max([bbox.max_lon for bbox in bboxes])
    max_lat = np.max([bbox.max_lat for bbox in bboxes])

    return BoundingBox(
        min_lon = min_lon,
        min_lat = min_lat,
        max_lon = max_lon,
        max_lat = max_lat
    )


"""The following lon/lat values were taken from https://github.com/azurro/country-bounding-boxes"""
STR2BB ={
    'Burkina Faso': BoundingBox(
        min_lon=-5.5175099, min_lat=9.393889, max_lon=2.40261, max_lat=15.084
    ),
    'Chad': BoundingBox(
        min_lon=13.47348, min_lat=7.44107, max_lon=24.0, max_lat=23.4975
    ),
    'Ethiopia': BoundingBox(
        min_lon=32.997734, min_lat=3.397448, max_lon=47.9823797, max_lat=14.8944684
    ),
    'Kenya': BoundingBox(
        min_lon=33.9098987, min_lat=-4.8995203, max_lon=41.899578, max_lat=4.62
    ),
    'Malawi': BoundingBox(
        min_lon=32.6703616, min_lat=-17.1295216, max_lon=35.9187531, max_lat=-9.368326
    ),
    'Mali': BoundingBox(
        min_lon=-12.240999, min_lat=10.147811, max_lon=4.2673828, max_lat=25.001084
    ),
    'Mozambique': BoundingBox(
        min_lon=30.2131759, min_lat=-26.9209426, max_lon=41.0545908, max_lat=-10.3252148
    ),
    'Niger': BoundingBox(
        min_lon=0.1689653, min_lat=11.693756, max_lon=15.996667, max_lat=23.517178
    ),
    'Nigeria': BoundingBox(
        min_lon=2.676932, min_lat=4.1042196, max_lon=14.677982, max_lat=13.885645
    ),
    'Somalia': BoundingBox(
        min_lon=40.9864985, min_lat=-1.8031968, max_lon=51.6177696, max_lat=12.1889121
    ),
    'South Sudan': BoundingBox(
        min_lon=23.447778, min_lat=3.48898, max_lon=35.948997, max_lat=12.236389
    ),
    'Sudan': BoundingBox(
        min_lon=21.8145046, min_lat=9.347221, max_lon=38.7017098, max_lat=22.224918
    ),
    'Uganda': BoundingBox(
        min_lon=29.5727424, min_lat=-1.4787899, max_lon=35.000308, max_lat=4.2340766
    ),
    'Zambia': BoundingBox(
        min_lon=21.996389, min_lat=-18.0766964, max_lon=33.7025, max_lat=-8.2712821
    ),
    'Zimbabwe': BoundingBox(
        min_lon=25.237368, min_lat=-22.4241095, max_lon=33.0563, max_lat=-15.6097038
    )
}

REGIONS = {
    'SSA': STR2BB.keys(),
    'East Africa': ['Ethiopia', 'Kenya', 'Somalia', 'South Sudan', 'Sudan', 'Uganda'],
    'West Africa & Chad': ['Burkina Faso', 'Chad', 'Mali', 'Niger', 'Nigeria'],
    'South Africa': ['Malawi', 'Mozambique', 'Zambia', 'Zimbabwe']
}