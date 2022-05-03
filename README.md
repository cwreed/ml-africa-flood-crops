# Deep learning for cropland and flood detection in sub-Saharan Africa

This repo is the body of my work for an independent study at NYU CDS using Google Earth Engine and deep learning to classify floods on crop lands. It is structured to emulate the analysis pipelines endorsed by researchers at NASA Harvest to ensure modularity and reproducibility, and further to develop some of my practices in good software engineering.

So far I've built out the code to export, process, and engineer datasets for training two models: one to classify cropland in sub-Saharan Africa and one to classify floods. I intend to combine their predictions later to create a new dataset to experiment with spatial time series models for predicting crop stress in response to floods.

## Setup

A conda environment can be created from the provided `environment.yml` file containing all the packages needed to run this code:
```
    conda env create -f environment.yml
```

Furthermore, you will need a Google Earth Engine account to run this code, which you can sign up [here](https://earthengine.google.com). I additionally recommend cloning this repo into your Google Drive (of the same account you use for Google Earth Engine) to make data export seamless. More on that below.


## Contents

* `src/`: Classes to construct the training dataset and model it as a supervised learning problem
    * `exporters/`: Classes to download data labels and earth observation imagery from NASA CropHarvest dataset and Google Earth Engine
    * `processors/`: Classes to process labels into a unified format that can be aligned with other geospatial data
    * `engineers/`: Classes to align labels with earth observation data and format into labeled arrays for training
    * `models/`: LSTM models and training boilerplate implemented in PyTorch Lightning
* `scripts/`: Scripts to execute the data construction and modeling pipeline:
    * Intended order: `export_labels.py` > `process_labels.py` > `export_data.py` > `engineer.py` > `train_model.py`
        * Note: `export_data.py` exports data from Google Earth Engine to the Google Drive linked with your account. If this repo is stored in and run from your Google Drive, then this script will export the data its correct place in the repo's `data/` folder. Otherwise, you will need to manually download the exported data from Google Drive and add it to the correct subfolder in `data/` 
    * Should be executed for a specified nation or region in sub-Saharan Africa (e.g., "Nigeria" or "East Africa")
        * Implemented regions are listed in `src/utils/regions.py`

