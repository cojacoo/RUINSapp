import xarray as xr
import os

from ruins.core import DataManager
from ruins.core.data_manager import HDF5Source

# some datasources are backed by git-lfs which have to be disabled on 
# github actions
NO_LFS = 'NO_LFS' in os.environ


def test_default_manager():
    """Instantiate the default data manager"""
    dm = DataManager()

    assert dm.cache == True

    # find some datasets weather dataset
    assert 'cordex_coast' in dm.datasources
    assert 'CMIP5grid' in dm.datasources


def test_weather_dataset():
    """Test the weather dataset"""
    dm = DataManager()

    # check weather dataset was loaded
    assert 'weather' in dm.datasources

    # check Source type
    weather = dm['weather']
    assert isinstance(weather, HDF5Source)

    # load the data
    if NO_LFS:
        print('No LFS, skipping partial test')
        return

    data = weather.read()
    assert isinstance(data, xr.Dataset)
