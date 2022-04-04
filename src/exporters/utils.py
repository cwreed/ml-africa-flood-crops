from typing import Union, Optional
from math import cos, radians

from src.utils.regions import BoundingBox, STR2BB, REGIONS, combine_bounding_boxes

import ee

class EEBoundingBox(BoundingBox):
    """A BoundingBox with additional Earth Engine functions"""

    def __init__(self, 
        bbox: Optional[BoundingBox]=None, 
        min_lon: Optional[float]=None, 
        min_lat: Optional[float]=None,
        max_lon: Optional[float]=None,
        max_lat: Optional[float]=None
    ) -> None:

        assert (
            (bbox is not None) | all([min_lon, min_lat, max_lon, max_lat])
        ), "Must supply either a BoundingBox or min_lon, min_lat, max_lon, max_lat to initialize EEBoundingBox"

        if bbox:
            min_lon=bbox.min_lon
            min_lat=bbox.min_lat
            max_lon=bbox.max_lon
            max_lat=bbox.max_lat

        super().__init__(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat
        )

    def to_ee_polygon(self) -> ee.Geometry.Polygon:
        return ee.Geometry.Polygon(
            [
                [self.min_lon, self.min_lat],
                [self.min_lon, self.max_lat],
                [self.max_lon, self.max_lat],
                [self.max_lon, self.min_lat]
            ]
        )

def meters_per_degree(mid_lat: float) -> tuple[float, float]:
    """
    Helper function to calculate a conversion rate of meters per degree
    in both latitude and longitude for a given latitude.
    """
    m_per_deg_lat = (
        111132.954 - 559.822 * cos(2.0 * mid_lat) + 1.175 * cos(radians(4.0 * mid_lat))
    )
    m_per_deg_lon = (3.14159265359 / 180) * 6367449 * cos(radians(mid_lat))

    return m_per_deg_lat, m_per_deg_lon

def bounding_box_from_center(
        mid_lat: float, 
        mid_lon: float, 
        surrounding_meters: Union[int, tuple[int, int]]
    ) -> EEBoundingBox:
    """Helper function used to create an EEBoundingBox around a point label"""

    m_per_deg_lat, m_per_deg_lon = meters_per_degree(mid_lat)

    if isinstance(surrounding_meters, int):
        surrounding_meters = (surrounding_meters, surrounding_meters)

    surrounding_lat, surrounding_lon = surrounding_meters

    deg_lat = surrounding_lat / m_per_deg_lat
    deg_lon = surrounding_lon / m_per_deg_lon

    min_lat, max_lat = mid_lat - deg_lat, mid_lat + deg_lat
    min_lon, max_lon = mid_lon - deg_lon, mid_lon + deg_lon

    return EEBoundingBox(
        min_lon = min_lon, min_lat = min_lat, max_lon = max_lon, max_lat = max_lat
    )

def bounding_boxes_to_polygon(ee_bboxes: list[EEBoundingBox]) -> ee.Geometry.MultiPolygon:
    """Helper function to convert a list of EEBoundingBoxes into a MultiPolygon"""

    ee_polygons = [bbox.to_ee_polygon() for bbox in ee_bboxes]

    return ee.Geometry.MultiPolygon(ee_polygons).dissolve().simplify(maxError=1)


def _initialize_ee_regions(class_object, region: Union[str, list[str]], combine_regions: bool=False):
    """Helper function to help initialize region attributes across many classes"""
    assert (
            (
                (isinstance(region, str)) &
                (
                    (region in REGIONS.keys()) | 
                    (region in STR2BB.keys())
                )
            ) |
            (
                (isinstance(region, list)) & 
                all(r in STR2BB.keys() for r in region)
            )
        ), f"Region must be one of {REGIONS.keys()} or one or more of {STR2BB.keys()}."
    

    class_object.region = region

    if combine_regions:
        class_object.region_type = 'single'
        class_object.region_bbox = combine_bounding_boxes(region)
        class_object.ee_region_geo = EEBoundingBox(class_object.region_bbox).to_ee_polygon()
    else:
        if (isinstance(region, str)) & (region in REGIONS.keys()):
            class_object.region_type = 'multiple'
            class_object.region_bbox = [STR2BB[r] for r in REGIONS[region]]
            class_object.ee_region_geo = bounding_boxes_to_polygon([EEBoundingBox(region) for region in class_object.region_bbox])
        elif (isinstance(region, str)) & (region in STR2BB.keys()):
            class_object.region_type = 'single'
            class_object.region_bbox = STR2BB[region]
            class_object.ee_region_geo = EEBoundingBox(class_object.region_bbox).to_ee_polygon()
        else:
            class_object.region_type = 'multiple'
            class_object.region_bbox = [STR2BB[r] for r in region]
            class_object.ee_region_geo = bounding_boxes_to_polygon([EEBoundingBox(region) for region in class_object.region_bbox])
    
    if class_object.region_type == 'multiple':
        class_object.region_name = "_".join(class_object.region).lower()
    else:
        class_object.region_name = class_object.region.lower()
