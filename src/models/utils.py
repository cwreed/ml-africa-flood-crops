from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

def preds_to_xr(
    predictions: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    feature_labels: Optional[list[str]]
) -> xr.Dataset:

    data_dict: dict[str, np.ndarray] = {'lat': lats, 'lon': lons}

    for prediction_idx in range(predictions.shape[1]):
        if feature_labels is not None:
            prediction_label = feature_labels[prediction_idx]
        else:
            prediction_label = f'prediction_{prediction_idx}'

        data_dict[prediction_label] = predictions[:, prediction_idx]

    return pd.DataFrame(data=data_dict).set_index(['lat', 'lon']).to_xarray().to_array()