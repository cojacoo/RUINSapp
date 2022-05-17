import streamlit as st

from ruins.core import build_config, debug_view, DataManager, Config


def main_app(**kwargs):
    """
    """
    # build the config and dataManager from kwargs
    url_params = st.experimental_get_query_params()
    config, dataManager = build_config(url_params=url_params, **kwargs)

    # set page properties and debug view    
    st.set_page_config(page_title='Sea level rise Explorer', layout=config.layout)
    debug_view.debug_view(dataManager, config, debug_name='DEBUG - initial state')

    # build the app
    st.header('Sea Level Rise Explorer')
    # TODO: move this text also into story mode?
    st.markdown('''This section presents a model to calculate the influence of rising sea level and extreme precipitation events on inland channel flooding in northern Germany.''',unsafe_allow_html=True)

    # TODO: Demonstrate model using one or two real flood events.

    # TODO: expert mode: user takes control over model parameters

    # end state debug
    debug_view.debug_view(dataManager, config, debug_name='DEBUG - finished app')


if __name__ == '__main__':
    import fire
    fire.Fire(main_app)
