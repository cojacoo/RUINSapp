"""
Data Select
===========

Alex will das dokumentieren....

"""
import streamlit as st

from ruins.core import DataManager, Config
from ruins.plotting import plt_map

__dimensions__ = ['selected_station', 'temporal_agg', 'include_climate']


def _map_wrapper(dataManager: DataManager, config: Config, expander_container = st.sidebar):
    # create_session_variables('inclue_climate', 'selected_station')
    
    # create correct map
    if config['include_climate']:
        fig = plt_map(dataManager, sel=config['selected_station'], cm='CORDEX')
        text = '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>). The climate model grid is given in <span style="color:orange">orange</span> with the selected references as filled dots.'''
    else:
        fig = plt_map(dataManager, sel=config['selected_station'])
        text = '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>).'''

    # add
    expander_container.plotly_chart(fig, use_container_width=True)
    expander_container.markdown(text, unsafe_allow_html=True)


def selected_station_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    pass


def temporal_agg_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    pass


def include_climate_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    pass


def compact_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    # placehoder for map
    map_placeholder = expander_container.container()

    # get a station list
    weather = dataManager['weather'].read()
    station_list = list(weather.keys()) # TODO station names krummhoern, coast, inland, niedersachsen?
    selected_station = expander_container.selectbox('Select station/group:', station_list)

    # select a temporal aggregation
    aggregations = config.get('temporal_aggregations', ['Annual', 'Monthly'])
    temp_agg = expander_container.selectbox('Select temporal aggregation:', aggregations)

    # include climate projections
    include_climate = expander_container.checkbox('Include climate projections (for coastal region)?', value=False)

    # add settings
    st.session_state.selected_station = selected_station
    st.session_state.temporal_agg = temp_agg
    st.session_state.include_climate = include_climate

    # session is set, add the map
    _map_wrapper(dataManager=dataManager, config=config, expander_container=map_placeholder)



def data_select(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    """
    # set default story mode 
    story_mode = config.get('story_mode', True)
    if 'story_mode' in kwargs:
        story_mode = kwargs['story_mode']

    # switch the story_mode
    if not story_mode:
        return compact_selector(dataManager=dataManager, config=config, expander_container=expander_container)
    
    # we are in story mode! hurray!
    if config.get('selected_station') is None:
        selected_station_selector(dataManager=dataManager, config=config, expander_container=expander_container)
    elif config.get('temporal_agg') is None:
        temporal_agg_selector(dataManager=dataManager, config=config, expander_container=expander_container)
    elif config.get('include_climate') is None:
        include_climate_selector(dataManager=dataManager, config=config, expander_container=expander_container)
    else:
        compact_selector(dataManager=dataManager, config=config, expander_container=expander_container)
