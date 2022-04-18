"""
Data Select
===========

Alex will das dokumentieren....

"""
from typing import List, Union
import streamlit as st

from ruins.core import DataManager, Config
from ruins.plotting import plt_map


# I have no better idea for translation right now
_TRANSLATE_EN = dict(
    map_caption_long='''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>). The climate model grid is given in <span style="color:orange">orange</span> with the selected references as filled dots.''',
    map_caption_short='''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>).''',
    welcome='Welcome to Weather Explorer',
    rcp_title='Include Climate Projections?',
    rcp_description="""RCPs are scenarios about possible greenhouse gas concentrations by the year 2100. RCP2.6 is a world in which little further greenhouse gasses are emitted -- similar to the Paris climate agreement from 2015. RCP8.5 was intendent to explore a rather risky, more worst-case future with further increased emissions. RCP4.5 is one candidate of a more moderate greenhouse gas projection, which might be more likely to resemble a realistic situation. It is important to note that the very limited differentiation between RCP scenarios have been under debate for several years. One outcome is the definition of Shared Socioeconomic Pathways (SSPs) for which today, however, there are not very many model runs awailable. For more information, please check with the [Climatescenario Primer](https://climatescenarios.org/primer/), [CarbonBrief](https://www.carbonbrief.org/explainer-how-shared-socioeconomic-pathways-explore-future-climate-change) and this [NatureComment](https://www.nature.com/articles/d41586-020-00177-3)""",
    rcp_infotext="### Activate climate projections?\nYou can select one of the scenarios below, or select 'No' to skip this step, then no climate projections will be included. Each projection has a linked video from https://sos.noaa.gov/ to illustrate the changes due to the scenario."
)
_TRANSLATE_DE = dict(
    map_caption_long='''Karte mit verfügbaren Stationen (<span style="color:blue">blauen Punkte</span>) und ausgewählte Referenzstation (<span style="color:magenta">magenta hervorgehoben</span>). Das Modellgrid ist in <span style="color:orange">orange</span> mit den ausgewählten Referenzen als gefüllte Punkte dargestellt.''',
    map_caption_short='''Karte mit verfügbaren Stationen (<span style="color:blue">blauen Punkte</span>) und ausgewählte Referenzstation (<span style="color:magenta">magenta hervorgehoben</span>).''',
    welcome='Willkommen bei Weather Explorer',
    rcp_title='Klimaprojektionen hinzufügen?',
    rcp_description="""RCPs sind mögliche Szenarios der Treibhausgaskonentrationen in der Atmosphäre bis zum Jahr 2100. RCP2.6 orientiert sich am Pariser Klimaabkommen von 2015, demnach kaum mehr Treibhasugase ausgestoßen werden. RCP8.5 beschreibt einen eher risikofreudigen Umgang, bei dem Konzentrationen ähnlich einem Worst-Case Szenario weiter ansteigen. RCP4.5 is ein moderates Modell, das eher an der Realität orientiert ist.. Es ist wichtig zu beachten, dass die schwierige Differenzierbarkeit zwischen RCP Szenarios seit langem diskutiert wird. Ein Ausgang ist die Definition von Shared Socioeconomic Pathways (SSPs), für die heute jedoch nur wenige Modell-Runs verfügbar sind. Für weitere Informationen, siehe [Climatescenario Primer](https://climatescenarios.org/primer/), [CarbonBrief](https://www.carbonbrief.org/explainer-how-shared-socioeconomic-pathways-explore-future-climate-change) oder dieser [Nature Comment](https://www.nature.com/articles/d41586-020-00177-3) kontaktieren.""",
    rcp_inftext = "### Klimaprojektionen hinzufügen?\nSie können eines der Szenarios unten auswählen, oder 'Keine' auswählen, um diesen Schritt zu überspringen. Jedes Szenario hat ein verlinktes Video von https://sos.noaa.gov/ zur Darstellung der Änderungen durch das Szenario."
)

_VIDEO_SOURCES = {
    'default': dict(
        title='### {rcp}',
        description='Video Source: [NOAA Science On a Sphere](https://sos.noaa.gov/catalog/datasets/climate-model-temperature-change-rcp-{rcp}-2006-2100/)',
        video_url='https://sos.noaa.gov/videos/rcp_ga_{rcp}_400.mp4'
    ),
    'rcp26': dict(
        title='### RCP 2.6',
        description='Video Source: [NOAA Science On a Sphere](https://sos.noaa.gov/catalog/datasets/climate-model-temperature-change-rcp-26-2006-2100/)',
        video_url='https://sos.noaa.gov/videos/rcp_ga_26_400.mp4'
    ),
    'rcp45': dict(
        title='### RCP 4.5',
        description='Video Source: [NOAA Science On a Sphere](https://sos.noaa.gov/catalog/datasets/climate-model-temperature-change-rcp-45-2006-2100/)',
        video_url='https://sos.noaa.gov/videos/rcp_ga_45_400.mp4'
    ),
    'rcp85': dict(
        title='### RCP 8.5',
        description='Video Source: [NOAA Science On a Sphere](https://sos.noaa.gov/catalog/datasets/climate-model-temperature-change-rcp-85-2006-2100/)',
        video_url='https://sos.noaa.gov/videos/rcp_ga_85_400.mp4'
    ),
    'allRCP': dict(
        title='### all RCPs',
        description='All RCPs try to project CO2 concentrations, like shown in the video. Source: [NOAA Science On a Sphere](https://sos.noaa.gov/catalog/datasets/carbon-dioxide-concentration-geos-5-model/)',
        video_url='https://sos.noaa.gov/videos/geos_5_carbon_400_audio.mp4'
    ),
    'weather': dict(
        title='### Weather Data',
        description='Compare with weather observations. The video shows the current global precipitation. [SOURCE](https://sos.noaa.gov/catalog/datasets/clouds-with-precipitation-real-time/)',
        video_url='https://sos.noaa.gov/videos/clouds_precip_400.mp4'
    ),
}


def _map_wrapper(dataManager: DataManager, config: Config, expander_container = st.sidebar, add_caption: bool = True, **kwargs):
    # get the translator
    t = kwargs.get('translator', config.translator(en=_TRANSLATE_EN, de=_TRANSLATE_DE))

    # create correct map
    if config.get('include_climate', False):
        fig = plt_map(dataManager, sel=config.get('selected_station', 'all'), cm='CORDEX')
        text = t('map_caption_long')
    else:
        fig = plt_map(dataManager, sel=config.get('selected_station', 'all'))
        text = t('map_caption_short')

    # add
    expander_container.plotly_chart(fig, use_container_width=True)

    if add_caption:
        expander_container.markdown(text, unsafe_allow_html=True)


def selected_station_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, **kwargs):
    """
    Select a station.
    """
    container = st if 'container' not in kwargs else kwargs['container']
    t = kwargs.get('translator', config.translator(en=_TRANSLATE_EN, de=_TRANSLATE_DE))
    
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

    container.title(t('welcome'))
    
    # placehoder for map
    _map_wrapper(dataManager=dataManager, config=config, expander_container=container)
    
    # introduction
    container.markdown("HERE WE EXPLAIN EVERYTHING THE USER NEEDS TO KNOW ABOUT THE DATA")

    container.info("Right now, you can't select a station by clicking on it. Please use the select below")

    cap = 'Station(s) to use' if config.lang == 'en' else 'Messstation(en):'
    preselect = container.selectbox(cap, [' - select - '] + station_list)

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

    use_annual = left.button('Jährliche Aggregierung' if config.lang == 'de' else 'Aggregate Annualy')
    use_monthly = right.button('Monatliche Aggregierung' if config.lang == 'de' else 'Aggregate Monthly')

    if use_annual:
        st.session_state.temporal_agg = "Annual"
    elif use_monthly:
        st.session_state.temporal_agg = "Monthly"
    else:
        st.stop()
    
    st.experimental_rerun()


def rcp_selector(dataManager: DataManager, config: Config, expander_container = st.sidebar, elements: Union[str, List[str]] = '__rpc__',  **kwargs):
    """
    Decide whether to include or exclude climate projects. 
    If climate projections should be included, the user can select one of the projections.
    """
    # RCP key to use - needed for multiple selects
    RCP_KEY = kwargs.get('RCP_KEY', 'current_rcp')

    # get the container
    container = st if 'container' not in kwargs else kwargs['container']
    t = kwargs.get('translator', config.translator(en=_TRANSLATE_EN, de=_TRANSLATE_DE))

    # check what to show as projections
    if elements == '__rpc__':
        # get all available climate projections
        projections = sorted(list(set([_.split('.')[-1] for _ in dataManager['cordex_coast'].read().keys()])))
    elif elements == '__all__':
        # get all available climate projections
        projections = sorted(list(set([_.split('.')[-1] for _ in dataManager['cordex_coast'].read().keys()])))
        projections = ['weather', 'allRCP'] + projections

    elif isinstance(elements, str):
        projections = [elements]
    else:
        projections = elements

    # check mode
    if config.has_key('include_climate') and kwargs.get('allow_skip', True):
        cap = 'Include climate projections' if config.lang == 'en' else 'Klimaprojektionen hinzufügen?'
        expander_container.checkbox(cap, key='include_climate')

    # add the selectbox
    if config.has_key(RCP_KEY):
        cap = 'Select RCP:' if config.lang == 'en' else 'RCP auswählen:'
        expander_container.selectbox(cap, projections, key=RCP_KEY, format_func=lambda x: x.upper())
        return
    
    # main content
    layout = kwargs.get('layout', 'columns')
    
    # title
    container.title(kwargs.get('title', t('rcp_title')))

    container.markdown(t('rcp_description'), unsafe_allow_html=True)

#    container.info("RPCs are cool, but make stuff a bit slower")
    container.markdown(t('rcp_infotext'))

    # check if the user can skip this step
    if kwargs.get('allow_skip', True):
        no = container.button('Ohne Klimaprojektionen fortfahren' if config.lang == 'de' else 'Continue without climate projections')

        # handle the no
        if no:
            st.session_state.include_climate = False
            st.experimental_rerun()
    
    # build the columns
    if layout == 'columns':
        cols = container.columns(len(projections))
    else:
        cols = [container.columns((1,2,2)) for _ in range(len(projections))]
    
    # container for click buttons
    use_rcp = []
    
    # go for it
    for col, rcp in zip(cols, projections):
        # get the layout and the data
        c = (col, col, col) if layout == 'columns' else col
        _dat = _VIDEO_SOURCES.get(rcp, _VIDEO_SOURCES['default'])
        
        print(rcp)
        # video
        c[0].video(_dat['video_url'].format(rcp=rcp[-2:]))
        
        # description
        c[1].markdown(_dat['title'].format(rcp=rcp.upper()))
        c[1].markdown(_dat['description'].format(rcp=rcp))
        
        # button
        use = c[2].button(f'Use {rcp.upper()}')
        use_rcp.append((use, rcp))
    
    # check if any button was used:
    for use, rcp in use_rcp:
        if use:
            st.session_state.include_climate = True
            st.session_state[RCP_KEY] = rcp
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
    
    # get a station list
    station_list = list(dataManager['weather'].read().keys())

    # get a translator
    t = config.translator(en=_TRANSLATE_EN, de=_TRANSLATE_DE)

    # switch the story_mode
    if not story_mode:
        st.session_state.selected_station = station_list[0]
        st.session_state.temporal_agg = "Annual"
        st.session_state.include_climate = False
    
    # we are in story mode! hurray!
    selected_station_selector(
        dataManager=dataManager,
        config=config,
        expander_container=expander_container,
        station_list=station_list,
        translator=t
    )
    temporal_agg_selector(
        dataManager=dataManager,
        config=config,
        expander_container=expander_container,
        translator=t
    )
    rcp_selector(
        dataManager=dataManager,
        config=config,
        expander_container=expander_container,
        translator=t
    )


def debug_main(**kwargs):
    from ruins.core import build_config
    params = st.experimental_get_query_params()
    config, dm = build_config(url_params=params, **kwargs)

    data_select(dataManager=dm, config=config)

    st.json(st.session_state)

if __name__ == '__main__':
    import fire
    fire.Fire(debug_main)
