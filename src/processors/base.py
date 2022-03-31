import os
from pathlib import Path

class BaseProcessor:
    """
    Base for all dataset-specific processor classes. It creates the appropriate directory in the data directory.
    """

    dataset: str

    def __init__(self, data_folder: Path = Path('data')) -> None:
        self.data_folder = data_folder
        self.raw_folder = os.path.join(self.data_folder, 'raw', self.dataset)
        self.processed_folder = os.path.join(self.data_folder, 'processed')

        assert os.path.exists(self.raw_folder), f"{self.raw_folder} does not exist. Must run export script for {self.dataset}."

        if not os.path.exists(self.processed_folder):
            os.mkdir(self.processed_folder)

        self.output_folder = os.path.join(self.processed_folder, self.dataset)

        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)