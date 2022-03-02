import os
from ruins.core import Config

def get_test_config() -> Config:
    """Get a test config"""
    # overwrite some settings for unit tests
    args = dict(
        datapath=os.path.abspath(os.path.join(os.path.dirname(__file__), 'testdata')),
        debug=True
    )
    return Config(**args)
