"""
Topic Selector
==============

"""
from typing import List
import streamlit as st
from PIL import Image

from ruins.core import Config


def full_topic_selector(config: Config, expander_container = st.sidebar, **kwargs):
    ''''''
    container = st if 'container' not in kwargs else kwargs['container']
    container.title('Select a topic you would like to explore')

    # introduction
    container.markdown('Inside the RUINS App you can explore different topics. Please select one, you can change the topic later in the sidebar.')

    # use a new container for each row
    row_container = st
    # column topic select (image, description, button)
    image_column, description_column, button_column = row_container.columns([1,3,1])
    
    image = Image.open('RUINS_logo_small.png') # idea: display symbol images for each topic
    
    image_column.image(image)
    warming = button_column.button('''Click here to select topic Warming''')
    description_column.markdown('''**Warming**: In this topic we provide visualisations to explore changes in observed weather data. Based on different variables and climate indices it is possible to investigate how climate change manifests itself in different variables, at different stations and with different temporal aggregation.''')
    row_container.markdown('''___''')

    row_container = st
    # column topic select (image, description, button)
    image_column, description_column, button_column = row_container.columns([1,3,1])

    image_column.image(image)
    weather_indices = button_column.button('''Click here to select topic Weather indices''')
    description_column.markdown('''**Weather indices**: Short weather indices description.''') # TODO
    row_container.markdown('''___''')
    
    if warming:
        st.session_state.current_topic = 'Warming'    
    elif weather_indices:
        st.session_state.current_topic = 'Weather Indices'
    else:
        st.stop()

    st.experimental_rerun()


def compact_topic_selector(config: Config, expander_container = st.sidebar):
    ''''''    
    # get topic list
    topic_list = config['topic_list']

    # select a topic
    if config.has_key('current_topic'):
        topic_idx = topic_list.index(config.get('current_topic'))
    else:
        topic_idx = 0
    current_topic = expander_container.selectbox('Select topic:', topic_list, index=topic_idx)


    # TODO: do we need get_control_policy? not used in data_select
    ## get the policy
    #policy = config.get_control_policy('topic_selector')
#
    ## create the control
    #if current_topic is not None:
    #    topic = expander_container.selectbox('Select a topic', topic_list)
    #
    #elif policy == 'show': # pragma: no cover
    #    topic = container.selectbox(
    #        'Select a topic',
    #        topic_list,
    #        #index=topic_list.index(config['current_topic'])
    #    )
    #
    #elif policy == 'hide': # pragma: no cover
    #    topic = config_expander.selectbox(
    #        'Select a topic',
    #        topic_list,
    #        #index=topic_list.index(config['current_topic'])
    #    )
    #
    #else:
    #    topic = current_topic

    # set the new topic
    #if current_topic != topic:
    st.session_state.current_topic = current_topic


def topic_selector(config: Config, expander_container=st.sidebar, **kwargs) -> str:
    """
    TODO: Alex will das dokumentieren....
    """
    # set default story mode 
    story_mode = config.get('story_mode', True)
    if 'story_mode' in kwargs:
        story_mode = kwargs['story_mode']

    # switch the story_mode
    if not story_mode:
        return compact_topic_selector(config=config, expander_container=expander_container)
    
    # we are in story mode
    if not config.has_key('current_topic'):
        full_topic_selector(config=config, expander_container=expander_container)
    else:
        compact_topic_selector(config=config, expander_container=expander_container)

