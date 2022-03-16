"""
Test the config object
"""
import os
import json
import pytest

from ruins import core


def test_default_config():
    """Check default paths"""
    # get the actual path
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..'))
    datapath = os.path.join(basepath, 'data')

    # initialize config without arguments
    conf = core.Config()

    assert conf.datapath.lower() == datapath.lower()
    assert conf.basepath.lower() == basepath.lower()
    assert conf.debug is False


def test_config_overwrites():
    """Check that default config can be overwritten"""
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..'))
    temppath = '/tmp/'

    conf = core.Config(datapath=temppath)

    assert conf.basepath.lower() == basepath.lower()
    assert conf.datapath.lower() == temppath.lower()


def test_from_json():
    """Build a config from file"""
    with open('test.json', 'w') as f:
        json.dump({'foo': 'bar'}, f)
    
    # build from config file
    conf = core.Config(path='test.json')

    assert conf['foo'] == 'bar'

    # remove test file
    os.remove('test.json')

def test_config_as_dict():
    """Test that the config behaves like a dict"""
    c = core.Config(foo='bar')

    # check custom configuration
    assert c['foo'] == 'bar'
    
    # check len and iter behavior
    i = 0
    for k in c:
        i += 1
    assert len(c) == i
    assert 'datapath' in c._keys

    # check default get behavior
    assert c.get('doesNotExist') is None
    assert c.get('doesNotExists', 'foobar') == 'foobar'


def test_config_key_error():
    """An unknown key should throw a key error"""
    c = core.Config()

    with pytest.raises(KeyError):
        c['doesNotExist']
