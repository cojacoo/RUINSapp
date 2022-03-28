"""
Data Select
===========

Alex will das dokumentieren....

"""
import streamlit as st
import time

from ruins.core import DataManager, Config
from ruins.plotting import plt_map

__dimensions__ = ['selected_station', 'temporal_agg', 'include_climate']


def _map_wrapper(dataManager: DataManager, config: Config, expander_container = st.sidebar):
    # create correct map
    if config.get('include_climate', False):
        fig = plt_map(dataManager, sel=config.get('selected_station', 'all'), cm='CORDEX')
        text = '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>). The climate model grid is given in <span style="color:orange">orange</span> with the selected references as filled dots.'''
    else:
        fig = plt_map(dataManager, sel=config.get('selected_station', 'all'))
        text = '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>).'''

    # add
    expander_container.plotly_chart(fig, use_container_width=True)
    expander_container.markdown(text, unsafe_allow_html=True)


def selected_station_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    """
    container = st if 'container' not in kwargs else kwargs['container']
    container.title('Welcome to Weather explorer')
    
    # placehoder for map
    _map_wrapper(dataManager=dataManager, config=config, expander_container=container)
    
    # introduction
    container.markdown('HERE WE EXPLAIN EVERYTHING THE USER NEEDS TO KNOW ABOUT THE DATA')

    container.info("Right now, you can't select a station by clicking on it. Please use the select below")

    # get a station list
    weather = dataManager['weather'].read()
    station_list = [' - Please select a station - '] + list(weather.keys()) # TODO station names krummhoern, coast, inland, niedersachsen?
    selected_station = container.selectbox('Select station/group:', station_list)

    # check input
    if selected_station == ' - Please select a station - ':
        st.stop()
    else:
        st.session_state.selected_station = selected_station
        st.experimental_rerun()


def temporal_agg_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    """
    container = st if 'container' not in kwargs else kwargs['container']
    container.title('Select a temporal aggregation')

    left, right = container.columns(2)

    left.info('I am a yearly aggregated preview')
    right.info('I am a monthly aggregated preview')

    use_annual = left.button('Aggregate Annualy')
    use_monthly = right.button('Aggregate Monthly')

    if use_annual:
        st.session_state.temporal_agg = 'Annual'
    elif use_monthly:
        st.session_state.temporal_agg = 'Monthly'
    else:
        st.stop()
    
    #time.sleep(0.1)
    st.experimental_rerun()

    


def include_climate_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    """
    container = st if 'container' not in kwargs else kwargs['container']
    container.title('Include Climate projections?')

    container.markdown(
        '''RCPs are scenarios about possible greenhouse gas concentrations by the year 2100. RCP2.6 is a world in which little further greenhouse gasses are emitted -- similar to the Paris climate agreement from 2015. RCP8.5 was intendent to explore a rather risky, more worst-case future with further increased emissions. RCP4.5 is one candidate of a more moderate greenhouse gas projection, which might be more likely to resemble a realistic situation. It is important to note that the very limited differentiation between RCP scenarios have been under debate for several years. One outcome is the definition of Shared Socioeconomic Pathways (SSPs) for which today, however, there are not very many model runs awailable. For more information, please check with the [Climatescenario Primer](https://climatescenarios.org/primer/), [CarbonBrief](https://www.carbonbrief.org/explainer-how-shared-socioeconomic-pathways-explore-future-climate-change) and this [NatureComment](https://www.nature.com/articles/d41586-020-00177-3)''',
        unsafe_allow_html=True)

    container.info('RPCs are cool, but make stuff a bit slower')
    container.markdown('### Activate climate projections?')
    left, right = container.columns(2)
    yes = left.button('Yes, use RPCs')
    no = right.button('No, I\'m fine')

    if yes:
        st.session_state.include_climate = True
    elif no:
        st.session_state.include_climate = False
    else:
        st.stop()
    
    #time.sleep(0.1)
    st.experimental_rerun()




def compact_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    """
    # placehoder for map
    map_placeholder = expander_container.container()

    # get a station list
    weather = dataManager['weather'].read()
    station_list = list(weather.keys()) # TODO station names krummhoern, coast, inland, niedersachsen?

    # select a station
    if config.has_key('selected_station'):
        station_idx = station_list.index(config.get('selected_station'))
    else:
        station_idx = 0
    # selected_station = expander_container.selectbox('Select station/group:', station_list, index=station_idx)
    expander_container.selectbox('Select station/group:', station_list, key='selected_station', index=station_idx)

    # select a temporal aggregation
    aggregations = config.get('temporal_aggregations', ['Annual', 'Monthly'])
    if config.has_key('temporal_agg'):
        agg_idx = aggregations.index(config['temporal_agg'])
    else:
        agg_idx = 0
    # temp_agg = expander_container.selectbox('Select temporal aggregation:', aggregations, index=agg_idx)
    expander_container.selectbox('Select temporal aggregation:', aggregations, key='temporal_agg', index=agg_idx)

    # include climate projections
    #include_climate = expander_container.checkbox('Include climate projections (for coastal region)?', value=config['include_climate'] if config.has_key('include_climate') else False)
    expander_container.checkbox('Include climate projections (for coastal region)?', key='include_climate')

    # add settings
    #st.session_state.selected_station = selected_station
    #st.session_state.temporal_agg = temp_agg
    #st.session_state.include_climate = include_climate

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
    if not config.has_key('selected_station'):
        selected_station_selector(dataManager=dataManager, config=config, expander_container=expander_container)
    elif not config.has_key('temporal_agg'):
        temporal_agg_selector(dataManager=dataManager, config=config, expander_container=expander_container)
    elif not config.has_key('include_climate'):
        include_climate_selector(dataManager=dataManager, config=config, expander_container=expander_container)
    else:
        compact_selector(dataManager=dataManager, config=config, expander_container=expander_container)
