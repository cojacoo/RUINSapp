import os
from os.path import join as pjoin
import json
from collections.abc import Mapping

class Config(Mapping):
    """
    Streamlit app Config object.

    This class holds all configs needed to run the apps. 
    It can be instantiated just like this to load default
    values. 
    If a path is provided, it will load the configs from
    the referenced json file. Any config can be updated
    by passed kwargs.

    This design makes the config updateable and easy to
    to manage. At the same time it can be persisted to
    the disk and even mirrored to a database if needed in
    the future.

    """
    def __init__(self, path: str = None, **kwargs) -> None:
        # set the default values

        # debug mode
        self.debug = False

        # path 
        self.basepath = os.path.abspath(pjoin(os.path.dirname(__file__), '..', '..'))
        self.datapath = pjoin(self.basepath, 'data')

        # mime readers
        self.default_sources = {
            'nc': 'HDF5Source',
            'csv': 'CSVSource',
        }
        self.default_sources.update(kwargs.get('include_mimes', {}))

        # reader args
        self.sources_args = {
            'stats.csv': dict(index_col=0),
            'hsim_collect.csv': dict(index_col=0),
            'windpowerx.csv': dict(index_col=0),
        }
        self.sources_args.update(kwargs.get('include_args', {}))

        # app management
        self.layout = 'centered'

        # app content
        self.topic_list = ['Warming', 'Weather Indices', 'Drought/Flood', 'Agriculture', 'Extreme Events', 'Wind Energy']

        # store the keys
        self._keys = ['debug', 'basepath', 'datapath', 'default_sources', 'sources_args', 'layout', 'topic_list']

        # check if a path was provided
        conf_args = self.from_json(path) if path else {}

        # update with kwargs
        conf_args.update(kwargs)
        self._update(conf_args)


    def from_json(self, path: str) -> dict:
        """loads the content of the JSON config file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise AttributeError(f"Config file {path} does not exist")
    
    def _update(self, new_settings: dict) -> None:
        """Update this instance with new settings"""
        for k, v in new_settings.items():
            setattr(self, k, v)
            if k not in self._keys:
                self._keys.append(k)
    
    def get(self, key: str, default = None):
        return getattr(self, key, default)
    
    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        for k in self._keys:
            yield k
    
    def __getitem__(self, key: str):
        return getattr(self, key)
