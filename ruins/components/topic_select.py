"""
Topic Selector
==============

"""
from typing import List
import streamlit as st

from ruins.core import Config


def topic_selector(config: Config, container=st, config_expander=st) -> str:
    """
    TODO: Alex will das dokumentieren....
    """
    current_topic = config.get('current_topic')
    
    # get topic list
    topic_list = config['topic_list']

    # get the policy
    policy = config.get_control_policy('topic_selector')

    # create the control
    if current_topic is not None:
        topic = container.selectbox('Select a topic', topic_list)
    
    elif policy == 'show':
        topic = container.selectbox(
            'Select a topic',
            topic_list,
            #index=topic_list.index(config['current_topic'])
        )
    
    elif policy == 'hide':
        topic = config_expander.selectbox(
            'Select a topic',
            topic_list,
            #index=topic_list.index(config['current_topic'])
        )
    
    else:
        topic = current_topic

    # set the new topic
    if current_topic != topic:
        st.session_state.current_topic = topic

