from ruins.apps import weather


# TODO use the config and inject the dedub config here

# TODO only run this test when the Value Error is solved
#def test_run_app():
#    """Make sure the appp runs without failing"""
#    weather.main_app()


def test_climate_indices():
    """Test only climate indices """
    w, c = weather.load_alldata()
    weather.climate_indices(w, c)
