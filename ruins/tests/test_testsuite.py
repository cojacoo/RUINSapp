"""
Test the test-suite itself. A dummy streamlit app, which raises an attribute
error should be catched by the test, although not printed to StdErr if run
by streamlit.

For the actual unittests, we import the real functions from the real app.
In the fututre, this test-suite can be extended to test the edge-cases
where streamlit runs into an error, that the plain function does not see.

"""
import pytest


def dummy_app():
    import streamlit as st

    st.title("Dummy App")
    raise AttributeError

def test_dummy_raises():
    with pytest.raises(AttributeError):
        dummy_app()
