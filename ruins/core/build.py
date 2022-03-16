"""
Build a :class:`Config <ruins.core.Config>` and a 
:class:`DataManager <ruins.core.DataManager>` from a kwargs dict.
"""
from typing import Union, Tuple, Dict, List
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
