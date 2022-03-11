import os
from pathlib import Path
import logging
from .base import BaseExporter
from cropharvest.datasets import CropHarvestLabels

class CropHarvestExporter(BaseExporter):
    """
    Exporter class to download binary crop labels from NASA CropHarvest dataset.
    See https://github.com/nasaharvest/cropharvest (Tseng et al., 2021)
    """

    dataset = 'cropharvest'

    def export(self):
        """
        Download the CropHarvest labels and store in the output folder.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        logger.info("Beginning export of NASA CropHarvest cropland labels")
        outpath = os.path.join(self.output_folder, 'labels.geojson')
        
        if outpath.exists():
            logger.info(f"Data already exported and available at {outpath}")
            CropHarvestLabels(self.output_folder, download=False)
        else:
            logger.info(f"Downloading data and writing to {outpath}")
            CropHarvestLabels(self.output_folder, download=True)
        