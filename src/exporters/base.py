import os
from pathlib import Path
from typing import Optional

class BaseExporter():
    """
    Base for all dataset-specific exporter classes. It creates the appropriate directory in the data directory
    
    """

    dataset: Optional[str] = None
    ee_im_coll: Optional[str] = None

    def __init__(self, data_folder: Path = Path('data')) -> None:

        self.data_folder = data_folder
        self.raw_folder = data_folder / 'raw'

        self.raw_folder.mkdir(exist_ok=True)

        if self.dataset is not None:
            self.output_folder = self.raw_folder / self.dataset
            self.output_folder.mkdir(exist_ok=True)