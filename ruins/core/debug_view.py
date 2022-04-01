import streamlit as st

from ruins.core import Config, DataManager


def debug_view(dataManager: DataManager, config: Config, debug_name: str = None) -> None:
    '''
    Set Config['debug'] = 'True' to display debug view which
    shows current Config and dataManager parameters.
    '''
    if config.debug:
        name = f'DEBUG [{debug_name}]' if debug_name else 'DEBUG'
        exp = st.expander(name, expanded=True)
        left, right = exp.columns(2)

        left.markdown('## Config paramters')
        left.json(dict(config))

        right.markdown('## Session state')
        right.json(st.session_state)
