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
        self.raw_folder = os.path.join(self.data_folder, 'raw')

        if not os.path.exists(self.raw_folder):
            os.mkdir(self.raw_folder)

        if self.dataset is not None:
            self.output_folder = os.path.join(self.raw_folder, self.dataset)
            if not os.path.exists(self.output_folder):
                os.mkdir(self.output_folder)