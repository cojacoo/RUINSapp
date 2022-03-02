"""
Topic Selector
==============

"""
from typing import List
import streamlit as st


def topic_selector(topic_list: List[str], force_topic_select: bool = True, container=st, **kwargs) -> str:
    """
    Select a topic from a list of topics. The selected topic is returned and
    additionally published to the session cache.

    Parameters
    ----------
    topic_list : List[str]
        List of topics to select from.
    force_topic_select : bool
        If False, the dropdown will not be shown if a topic was already
        selected and is present in the streamlit session cache.
        Default: True
    container : streamlit.st.container
        Container to use for the dropdown. Defaults to main streamlit.
    **kwargs
        These keyword arguments are only accepted to directly inject
        :class:`Config <ruins.config.Config>` objects.
    """
    # check if a topic is already present
    if kwargs.get('no_cache', False):
        current_topic = kwargs.get('current_topic', None)
    else:  # pragma: no cover
        current_topic = st.session_state.get('topic', kwargs.get('current_topic', None))
    if current_topic is not None and not force_topic_select:
        return current_topic
    
    # otherwise print select
    topic = st.selectbox(
        'Select a topic',
        topic_list, 
        index=0 if current_topic is None else topic_list.index(current_topic)
    )

    # store topic in session cache
    if current_topic != topic and not kwargs.get('no_cache', False): # pragma: no cover
        st.session_state['topic'] = topic
    
    return topic
