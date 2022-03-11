"""
Build a :class:`Config <ruins.core.Config>` and a 
:class:`DataManager <ruins.core.DataManager>` from a kwargs dict.
"""
from types import Union, Tuple

from .config import Config
from .data_manager import DataManager



def build_config(omit_dataManager: bool = False, **kwargs) -> Tuple[Config, Union[None, DataManager]]:
    """
    """
    # extract the DataManager, if it was already instantiated
    if 'dataManager' in kwargs:
        dataManager = kwargs.pop('dataManager')
    else:
        dataManager = None

    # build the Config
    config = Config(**kwargs)

    if omit_dataManager:
        return config
    else:
        if dataManager is None:
            dataManager = DataManager(**config)
        return config, dataManager
