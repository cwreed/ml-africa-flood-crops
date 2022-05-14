from pathlib import Path
from datetime import date, datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def sentinel_2_as_tci(sentinel_ds: xr.DataArray, scale: bool=False) -> xr.DataArray:
    band2idx = {
        band: idx for idx, band in enumerate(sentinel_ds.attrs['band_descriptions'])
    }

    tci_bands = ['B4', 'B3', 'B2']
    tci_indices = [band2idx[band] for band in tci_bands]
    if scale:
        return sentinel_ds.isel(band=tci_indices) / 10000 * 2.5
    else:
        return sentinel_ds.isel(band=tci_indices) * 2.5

def sentinel_1_as_rgb(sentinel_ds: xr.DataArray) -> xr.DataArray:
    ds = sentinel_ds.assign_attrs(VV_VH = sentinel_ds['VV'] / sentinel_ds['VH'])

    return ds


def plot_preds_with_rgb(preds: xr.DataArray, rgb: xr.DataArray, pred_date: date, save_file: Path) -> None:
    plt.clf()

    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(30, 10), subplot_kw={'projection': ccrs.PlateCarree()}
    )

    fig.suptitle(
        f"Model predictions (Bottom left corner (lon, lat) = ({float(preds.lon.min())}, {float(preds.lat.min())}))"
        f"\n Inference date: {datetime.strftime(pred_date, '%Y-%m-%d')}",
        fontsize=15,
    )

    img_extent_1 = (rgb.x.min(), rgb.x.max(), rgb.y.min(), rgb.y.max())
    img = np.clip(np.moveaxis(rgb.values, 0, -1), 0, 1)

    ax[0].set_title("True color image")
    ax[0].imshow(
        img, origin='upper', extent=img_extent_1, transform=ccrs.PlateCarree()
    )

    ax[1].set_title("Mask")
    im = ax[1].imshow(
        preds.values,
        origin='upper',
        extent=img_extent_1,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=1,
        cmap='magma'
    )

    ax[2].set_title("Mask on top of the RGB image")
    ax[2].imshow(
        img, origin='upper', extent=img_extent_1, transform=ccrs.PlateCarree()
    )    
    ax[2].imshow(
        preds.values > 0.5,
        origin='upper',
        extent=img_extent_1,
        transform=ccrs.PlateCarree(),
        alpha=0.3,
        vmin=0,
        vmax=1,
        cmap='magma',
    )

    fig.colorbar(
        im, ax=ax.ravel().tolist()
    )

    plt.savefig(
        save_file, bbox_inches='tight', dpi=300
    )