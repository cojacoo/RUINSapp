"""
Build a :class:`Config <ruins.core.Config>` and a 
:class:`DataManager <ruins.core.DataManager>` from a kwargs dict.
"""
from typing import Union, Tuple, Dict, List
import os
import shutil
import requests
import io
import zipfile
import streamlit as st

from .config import Config
from .data_manager import DataManager


st.experimental_singleton
def contextualized_data_manager(**kwargs) -> DataManager:
    return DataManager(**kwargs)


def build_config(omit_dataManager: bool = False, url_params: Dict[str, List[str]] = {}, **kwargs) -> Tuple[Config, Union[None, DataManager]]:
    """
    """
    # prepare the url params, if any
    # url params are always a list: https://docs.streamlit.io/library/api-reference/utilities/st.experimental_get_query_params
    # TODO: This should be sanitzed to avoid injection attacks!
    ukwargs = {k: v[0] if len(v) == 1 else v for k, v in url_params.items()}
    kwargs.update(ukwargs)

    # extract the DataManager, if it was already instantiated
    if 'dataManager' in kwargs:
        dataManager = kwargs.pop('dataManager')
    else:
        dataManager = None

    # build the Config
    config = Config(**kwargs)

    if omit_dataManager:
        return config,  None
    else:
        if dataManager is None:
            dataManager = contextualized_data_manager(**config)
        return config, dataManager


def download_data_archive(path: str = None, url: str = 'http://116.203.189.3/data.zip', DOI: str = None, if_exists: str = 'error'):
    """Download the data archive and extract into the data folder.
    If the path is None, the default path inside the repo itself is used.
    Then, you also need to change the datapath property of the application config.
    If the data folder already exists and is not empty, the function will error on default.
    You can pass ``if_exists='prune'`` to remove the existing data folder and replace it with the new one.
    """
    # use default path if none was provided
    if path is None:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data'))
    
    # check if the data folder already exists
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        if if_exists == 'error':
            raise OSError(f"The data path {path} already exists and is not empty. Pass if_exists='prune' to remove it.")
        elif if_exists == 'prune':
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            raise AttributeError(f'if_exists must be one of "error", "prune"')
    
    # check which download route is used:
    if DOI is None:
        # now the data folder exists - download the archive
        print(f'Found Server URL: {url}\nStart downloading...', end='', flush=True)
        
        req = requests.get(url, stream=True)
        zip = zipfile.ZipFile(io.BytesIO(req.content))

        print(f'done.\nExtracting to {path}...', end='', flush=True)
        zip.extractall(os.path.abspath(os.path.join(path, '..')))
        print('done.', flush=True)
    else:
        # now the data folder exists - download the archive
        print(f'Found DOI: {DOI}\nStart downloading...', end='', flush=True)

        # Build the URL from Zenodo DOI
        chunk = DOI.split('/')[-1]
        record = chunk.split('.')[1]

        # request the existing data from Zenodo API
        dat = requests.get(f'https://zenodo.org/api/records/{record}').json()
        for f in dat['files']:
            if f['type'] == 'zip':
                req = requests.get(f['links']['self'], stream=True)
                zip = zipfile.ZipFile(io.BytesIO(req.content))

                # extract the data to the data folder
                print(f'done.\nExtracting to {path}...', end='', flush=True)
                zip.extractall(os.path.abspath(os.path.join(path, '..')))
                print('done.', flush=True)
                break
