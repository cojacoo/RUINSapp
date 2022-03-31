"""
Data Select
===========

Alex will das dokumentieren....

"""
from distutils.command.build import build
import streamlit as st

from ruins.core import DataManager, Config
from ruins.plotting import plt_map




def _map_wrapper(dataManager: DataManager, config: Config, expander_container = st.sidebar, add_caption: bool = True):
    # create correct map
    if config.get('include_climate', False):
        fig = plt_map(dataManager, sel=config.get('selected_station', 'all'), cm='CORDEX')
        text = '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>). The climate model grid is given in <span style="color:orange">orange</span> with the selected references as filled dots.'''
    else:
        fig = plt_map(dataManager, sel=config.get('selected_station', 'all'))
        text = '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>).'''

    # add
    expander_container.plotly_chart(fig, use_container_width=True)

    if add_caption:
        expander_container.markdown(text, unsafe_allow_html=True)


def selected_station_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    Select a station.
    """
    container = st if 'container' not in kwargs else kwargs['container']
    
    # get a station list
    if 'station_list' in kwargs:
        station_list = kwargs['station_list']
    else:
        weather = dataManager['weather'].read()
        station_list = list(weather.keys()) # TODO station names krummhoern, coast, inland, niedersachsen?    
    
    if config.has_key('selected_station'):
        _map_wrapper(dataManager, config, expander_container, add_caption=False)
        expander_container.selectbox('Station(s) to use', station_list, key='selected_station')
        return

    container.title("Welcome to Weather explorer")
    
    # placehoder for map
    _map_wrapper(dataManager=dataManager, config=config, expander_container=container)
    
    # introduction
    container.markdown("HERE WE EXPLAIN EVERYTHING THE USER NEEDS TO KNOW ABOUT THE DATA")

    container.info("Right now, you can't select a station by clicking on it. Please use the select below")

    
    preselect = container.selectbox('Station(s) to use', [' - select - '] + station_list)

    # check input
    if preselect ==  ' - select - ':
        st.stop()
    else:
        st.session_state.selected_station = preselect
        st.experimental_rerun()


def temporal_agg_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    Select a temporal aggregation.
    """
    container = st if 'container' not in kwargs else kwargs['container']
    if config.has_key('temporal_agg'):
        expander_container.selectbox('Temporal aggregation', config.get('temporal_aggregations', ["Annual", "Monthly"]), key='temporal_agg')
        return

    container.title("Select a temporal aggregation")

    left, right = container.columns(2)

    left.info("I am a yearly aggregated preview")
    right.info("I am a monthly aggregated preview")

    use_annual = left.button('Aggregate Annualy')
    use_monthly = right.button('Aggregate Monthly')

    if use_annual:
        st.session_state.temporal_agg = "Annual"
    elif use_monthly:
        st.session_state.temporal_agg = "Monthly"
    else:
        st.stop()
    
    st.experimental_rerun()


def rcp_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    Decide whether to include or exclude climate projects. 
    If climate projections should be included, the user can select one of the projections.
    """
    # get the container
    container = st if 'container' not in kwargs else kwargs['container']

    # get all available climate projections
    projections = list(set([_.split('.')[-1] for _ in dataManager['cordex_coast'].read().keys()]))

    # check mode
    if config.has_key('include_climate'):
        expander_container.checkbox('Include climate projections (for coastal region)?', key='include_climate')

        # add the selectbox
        if config['include_climate']:
            expander_container.selectbox('Select RCP:', projections, key='current_rcp', format_func=lambda x: x.upper())
        return
    
    container.title('Include Climate projections?')

    container.markdown(
        """RCPs are scenarios about possible greenhouse gas concentrations by the year 2100. RCP2.6 is a world in which little further greenhouse gasses are emitted -- similar to the Paris climate agreement from 2015. RCP8.5 was intendent to explore a rather risky, more worst-case future with further increased emissions. RCP4.5 is one candidate of a more moderate greenhouse gas projection, which might be more likely to resemble a realistic situation. It is important to note that the very limited differentiation between RCP scenarios have been under debate for several years. One outcome is the definition of Shared Socioeconomic Pathways (SSPs) for which today, however, there are not very many model runs awailable. For more information, please check with the [Climatescenario Primer](https://climatescenarios.org/primer/), [CarbonBrief](https://www.carbonbrief.org/explainer-how-shared-socioeconomic-pathways-explore-future-climate-change) and this [NatureComment](https://www.nature.com/articles/d41586-020-00177-3)""",
        unsafe_allow_html=True)

    container.info("RPCs are cool, but make stuff a bit slower")
    container.markdown("### Activate climate projections?\nYou can select one of the scenarios below, or select 'No' to skip this step.")
    no = container.button('Continue without climate projections')

    # handle the no
    if no:
        st.session_state.include_climate = False
        st.experimental_rerun()
    
    # build the columns
    cols = container.columns(len(projections))
    url = config['rcp_video_url']
    use_rcp = []
    for col, rcp in zip(cols, projections):
        # video
        col.video(url.format(rcp=rcp[-2:]))
        col.markdown(f"### {rcp.upper()}")
        col.markdown(f'video linked form https://sos.noaa.gov/catalog/datasets/climate-model-temperature-change-rcp-{rcp[-2:]}-2006-2100/')
        use = col.button(f'Use {rcp.upper()}')
        use_rcp.append((use, rcp))
    
    # check if any button was used:
    for use, rcp in use_rcp:
        if use:
            st.session_state.include_climate = True
            st.session_state.current_rcp = rcp
            st.experimental_rerun()
    
    # if we didn't restart, stop the application
    st.stop()


def data_select(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    Select a station, a temporal aggregation and if climate projections 
    should be included.
    If we are either not in story_mode or if all of the above are already 
    selected, new data can be selected in a selectbox in the sidebar 
    (compact_selector()).
    Otherwise data is selected step by step alongside with explanations.
    """
    # set default story mode 
    story_mode = config.get('story_mode', True)
    if 'story_mode' in kwargs:
        story_mode = kwargs['story_mode']

    station_list = list(dataManager['weather'].read().keys())

    # switch the story_mode
    if not story_mode:
        st.session_state.selected_station = station_list[0]
        st.session_state.temporal_agg = "Annual"
        st.session_state.include_climate = False
    
    # we are in story mode! hurray!
    selected_station_selector(dataManager=dataManager, config=config, expander_container=expander_container, station_list=station_list)
    temporal_agg_selector(dataManager=dataManager, config=config, expander_container=expander_container)
    rcp_selector(dataManager=dataManager, config=config, expander_container=expander_container)


def debug_main(**kwargs):
    from ruins.core import build_config
    params = st.experimental_get_query_params()
    config, dm = build_config(url_params=params, **kwargs)

    data_select(dataManager=dm, config=config)

    st.json(st.session_state)

if __name__ == '__main__':
    import fire
    fire.Fire(debug_main)
