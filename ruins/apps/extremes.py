import streamlit as st

from ruins.core import build_config, debug_view, DataManager, Config


_TRANSLATE_EN = dict(
    title='Extreme events & flooding',
    introduction="""
    In this section, a model is presented to assess the influence of sea level change, inland precipitation, and 
    management options on flooding risks in a below sea level and thus drained area in northern Germany.
"""
)

_TRANSLATE_DE = dict(
    title='Extremereignisse & Überflutungen',
    introduction="""
    In diesem Abschnitt wird ein Modell vorgestellt, mit dem sich der Einfluss von Meeresspiegelveränderungen, 
    Inlandsniederschlägen und Managementoptionen auf Überflutungsrisiken in einem unterhalb des Meeresspiegels 
    liegenden und damit entwässerten Gebiet in Norddeutschland auswirken.
"""
)


def concept_explainer(config: Config, **kwargs):
    """Show an explanation, if it was not already shown.
    """
    # check if we saw the explainer already
    if config.has_key('extremes_explainer'):
        return
    
    # get the container and a translation function
    container = kwargs['container'] if 'container' in kwargs else st
    t = config.translator(en=_TRANSLATE_EN, de=_TRANSLATE_DE)

    # place title and intro
    container.title(t('title'))
    container.markdown(t('introduction'), unsafe_allow_html=True)

    # check if the user wants to continue
    accept = container.button('WEITER' if config.lang == 'de' else 'CONTINUE')
    if accept:
        st.session_state.extremes_explainer = True
        st.experimental_rerun()
    else:
        st.stop()


def demonstrate_flood_model(config:Config, **kwargs):
    """
    Demonstrate the flooding model using a real-life flood disaster.
    """
    container = kwargs['container'] if 'container' in kwargs else st
    t = config.translator(de=_TRANSLATE_DE, en=_TRANSLATE_EN)

    st.empty()


def flood_model(config:Config, **kwargs):
    """
    Version of the flooding model in which the user can play around with the parameters.
    """
    container = kwargs['container'] if 'container' in kwargs else st
    t = config.translator(de=_TRANSLATE_DE, en=_TRANSLATE_EN)

    st.empty()
    

def main_app(**kwargs):
    """
    """
    # build the config and dataManager from kwargs
    url_params = st.experimental_get_query_params()
    config, dataManager = build_config(url_params=url_params, **kwargs)

    # set page properties and debug view    
    st.set_page_config(page_title='Sea level rise Explorer', layout=config.layout)
    debug_view.debug_view(dataManager, config, debug_name='DEBUG - initial state')

    # explainer
    concept_explainer(config)

    # TODO: Demonstrate model using one or two real flood events.
    demonstrate_flood_model(config)

    # TODO: expert mode: user takes control over model parameters
    flood_model(config)

    # end state debug
    debug_view.debug_view(dataManager, config, debug_name='DEBUG - finished app')


if __name__ == '__main__':
    import fire
    fire.Fire(main_app)
