"""
Model scale selector
"""
import streamlit as st

from ruins.core import DataManager, Config


_TRANSLATE_EN = dict(
    title='Climate models have scale',
    description="""
There are many scientists working in different groups to build and enhance computer models of the climate physics.
These models operate at different scales in space and time. In our case, we refer to i) **global climate model** 
projections with a spatial resolution of approx. 120 km of one pixel at the surface and to ii) 
**regional climate model** projections with a spatial resolution of approx. 11 km.
As the name suggests, global climate models (or GCMs) cover the whole globe.
Their speciality is to simulate the overall physics of the atmosphere and land surface interactions.
The regional climate models (or RCMs) build on top of GCMs and build physical and/or statistical relationships
between GCM-outputs and locally observed weather. They seek to compensate the very coarse approximations of GCMs at
the price of more data being extrapolated into the future.

For more detailed reading, please continue here:
[CarbonBrief - How do climate models work?](https://www.carbonbrief.org/qa-how-do-climate-models-work)
    """,
    options={
        'Global': 'Global model',
        'Regional': 'Regional model'
    }
)

_TRANSATE_DE = dict(
    title='Skalierung von Klimamodellen',
    description="""
Es gibt unzählige Wissenschaftler_innen und Institutionen, die an einer Verbesserung der Computermodelle und dem
zugrundeliegenden physikalischen Verständnis arbeiten. Diese Modelle operieren auf verschiedenen *räumlichen* und 
*zeitlichen* Skalen. In dieser App haben **Globale Klimamodelle** eine räumlcihe Auflösung von ca. 120km pro Pixel an der
Erdoberfläche. **Regionale Klimamodelle** haben hingegen eine räumliche Auflösung von ca. 11km.
Dafür bedecken globale Modelle, wie der Name schon sagt, die ganze Welt. Sie sind besonders gut geeignet um globale Strömungen
und Interaktionen zwischen der Atmophäre und der Landoberfläche zu simulieren.
Die Regionalen modelle (RCMs) bauen dann weiter auf den (GCMs) auf, indem Sie globale Modelle entweder über statistische
oder physikalische Zusammenhänge mit lokal erhobenen Messdaten verknüpfen. Damit soll die grobe Auflösung von GCMs
wettgemacht werden.

Für mehr Informationen, lese hier weiter:
[CarbonBrief - Wie funktionieren Klimamodelle?](https://www.carbonbrief.org/qa-how-do-climate-models-work)
    """,
    options={
        'Global': 'Globales Modell',
        'Regional': 'Regionales Modell'
    }
)


def model_scale_selector(dataManager: DataManager, config: Config, expander_container=st.sidebar, **kwargs):
    """
    """
    # get the container
    container = st if 'container' not in kwargs else kwargs['container']

    # get a translator
    t = config.translator(en=_TRANSLATE_EN, de=_TRANSATE_DE)

    # get the translated options
    OPTIONS = t('options')
    # check if main page was already shown
    if config.has_key('climate_scale'):
        expander_container.radio(
            'Climate model' if config.lang == 'en' else 'Klimamodell',
            options=list(OPTIONS.keys()),
            format_func=lambda k: OPTIONS.get(k),
            key='climate_scale'
        )
        return

    # add the title
    container.title(t('title'))

    # add the description
    container.markdown(t('description'), unsafe_allow_html=True)

    # add the columns
    right, left = container.columns(2)

    right.info('THIS IS A THUMBNAIL')
    use_global = right.button('USE GLOBAL MODELS' if config.lang == 'en' else 'GLOBALES MODELL')

    left.info('THIS IS A THUMBNAIL')
    use_regional = left.button('USE REGIONAL MODELS' if config.lang == 'en' else 'REGIONALES MODELL')

    if use_global:
        st.session_state.climate_scale = 'Global'
        st.experimental_rerun()
    elif use_regional:
        st.session_state.climate_scale = 'Regional'
        st.experimental_rerun()
    else:
        st.stop()


def model_scale_select(dataManager: DataManager, config: Config, expander_container=st.sidebar, **kwargs):
    """
    """
    # set default story mode 
    story_mode = config.get('story_mode', True)
    if 'story_mode' in kwargs:
        story_mode = kwargs['story_mode']

    # pre-select 
    if config.has_key('climate_scale') or not story_mode:
        st.session_state.climate_scale = config.get('climate_scale', 'Global')
    
    # call the seletor
    model_scale_selector(dataManager, config, expander_container=expander_container)


def debug_main(**kwargs):
    from ruins.core import build_config
    params = st.experimental_get_query_params()
    config, dataManager = build_config(url_params=params, **kwargs)

    model_scale_select(dataManager, config)

    st.json(st.session_state)


if __name__ == '__main__':
    import fire
    fire.Fire(debug_main)
