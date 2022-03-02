"""
Data Manager
============

The DataManager is a wrapper around all data sources used by RUINSapp.
It can be configures by any :class:`Config <ruins.core.config.Config>` class
and organizes or caches all data sources using a 
:class:`DataSource <ruins.core.data_manager.DataSource>` inherited class.
This makes the read and filter interface available on all sources, no matter
where they are stored.
Using the :class:`Config <ruins.core.config.Config>` to instantiate a data
manager can in principle enabled different profiles, or even an interaction
with the frontend, although not implemented nor desired at the current stage.

Example
-------

.. code-block:: python

    from ruins import core

    # create default config
    conf = core.Config()

    # create a data manager from this
    dm = core.DataManager(**conf)

Of course, the data manager can also be used without the config, ie. to open it
in debug mode:

.. code-block:: python
    
    # using conf with conf.debug=False and overwrite it
    dm = core.DataManager(**conf, debug=True)

"""
import abc
import os
import glob
import xarray as xr
from collections.abc import Mapping
from typing import Type, List


DEFAULT_MIMES = {
    'nc': 'HDF5Source'
}


class DataSource(abc.ABC):
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @abc.abstractmethod
    def read(self):
        pass
    
    @abc.abstractmethod
    def filter(self, **kwargs):
        pass


class HDF5Source(DataSource):
    def __init__(self, path: str, cache: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.cache = cache

    def _load_source(self):
        """Method to load the actual source on the disk"""
        return xr.load_dataset(self.path)

    def read(self):
        if self.cache:
            if not hasattr(self, 'data'):
                self.data = self._load_source()
            return self.data

        else:
            return self._load_source()
    
    def filter(self):
        pass


class DataManager(Mapping):
    """Main class for accessing different data sources.

    The DataManager holds and manages all data sources. The default behavior is
    to scan the specified path for files of known file extension and cache them
    in memory.

    Parameters
    ----------
    datapath : str
        A location where the data is stored. The class will load all sources 
        there and make them accessible through DataSource classes.
    cache : bool
        Will be passed to the DataSource classes. It true, the source will only
        be read once and then stored in memory until the DataManager gets
        deconstructed.
    include_mimes : dict
        A dictionary of file extensions and their corresponding DataSource.
        If something is not listed, the DataManager will ignore the file type.
        The include_mimes can be overwritten by passing filenames directly.

    """
    def __init__(self, datapath: str = None, cache: bool = True, debug: bool = False, **kwargs) -> None:
        """
        You can pass in a Config as kwargs.
        """
        # check if the no config - or config without datapath - was passed
        if datapath is None:
            from ruins.core import Config
            self.from_config(**Config(**kwargs))
        else:
            self.from_config(datapath=datapath, cache=cache, debug=debug, **kwargs)
    
    def from_config(self, datapath: str = None, cache: bool = True, debug: bool = False, **kwargs) -> None:
        """
        Initialize the DataManager from a :class:`Config <ruins.core.Config>` object.
        """
        # store the main settings
        self._config = kwargs
        self._datapath = datapath
        self.cache = cache
        self.debug = debug

        # file settings
        self._data_sources = {}

        # infer data source
        if self._datapath is not None:
            self._infer_from_folder()
    
    @property
    def datapath(self) -> str:
        return self._datapath

    @datapath.setter
    def datapath(self, path: str) -> None:
        if os.path.exists(path):
            self._datapath = path
            self._infer_from_folder()
        else:
            raise OSError(f"{path} does not exist.")
    
    @property
    def datasources(self) -> List[DataSource]:
        return list(self._data_sources.keys())

    def _infer_from_folder(self) -> None:
        """
        Read all files from the datapath as specified on instantiation.
        Calls :func:`add_source` on each file.
        """
        # get a list of all files
        file_list = glob.glob(os.path.join(self.datapath, '*'))
        file_list.extend(glob.glob(os.path.join(self.datapath, '**', '*')))


        for fname in file_list:
            self.add_source(path=fname, not_exists='warn' if self.debug else 'ignore')

    def add_source(self, path: str, not_exists: str = 'raise') -> None:
        """
        Add a file as data source to the DataManager.
        Only if the file has an allowed file extension, it will be managed.
        Files of same name will be overwritten, this is also true if they had
        different extensions.

        """
        # load the tracked
        mimes = self._config.get('include_mimes', DEFAULT_MIMES)

        # get the basename
        try:
            basename, mime = os.path.basename(path).split('.')
        except ValueError:
            if self.debug:
                print(f"[Warning]: {path} has no extension.")
            return 
        
        if mime in mimes.keys():
            # get the class - overwirte by direct kwargs settings if needed
            clsName = mimes[mime] if basename not in self._config else self._config[basename]
            BaseClass = self.resolve_class_name(clsName)
            
            # add the source
            args = self._config.get(basename, {})
            args.update({'path': path, 'cache': self.cache})
            self._data_sources[basename] = BaseClass(**args)
        else:
            if not_exists == 'raise':
                raise OSError(f"{path} is not a configured data source")
            elif not_exists == 'ignore':
                pass
            elif not_exists == 'warn':
                print(f"{path} is found, but not a configured data source")

    def resolve_class_name(self, cls_name: str) -> Type[DataSource]:
        # checkout globals
        cls = globals().get(cls_name, False)
        
        # do we have a class?
        if not cls:
            # TODO, there is maybe an extension module to search one day
            raise RuntimeError(f"Can't find class {cls_name}.")
        
        return cls

    def __len__(self):
        """Return the number of managed data sources"""
        return len(self._data_sources)

    def __iter__(self):
        """Iterate over all dataset names"""
        for name in self._data_sources.keys():
            yield name
    
    def __getitem__(self, key: str):
        """Return the requested datasource"""
        return self._data_sources[key]

    def __repr__(self):
        return f"{self.__class__.__name__}(datapath={self.datapath}, cache={self.cache})"
    
    def __str__(self):
        return f"<DataManager of {len(self)} sources>"
