from ruins.apps import weather
from ruins.tests.util import get_test_config

from ruins.core import DataManager


# TODO use the config and inject the dedub config here

# TODO only run this test when the Value Error is solved
#def test_run_app():
#    """Make sure the appp runs without failing"""
#    weather.main_app()


def test_climate_indices():
    """Test only climate indices """
    conf = get_test_config()
    dm = DataManager(**conf)

    w = dm['weather'].read()
    c = dm['cordex_coast'].read()

    # run
    weather.climate_indices(w, c)
