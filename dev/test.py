import streamlit as st
from ruins.core import build_config

def func1(dm):
    weather = dm['weather'].read()
    options = list(weather.keys())

    st.title('Step One')


    st.write('option is not found, preselect an option')
    # clicked = st.button('OKI DOKI')

    # if clicked:
    #     st.session_state.option = 'option 2'
    #     st.experimental_rerun()
    # else:
    #     st.stop()
    preselect = st.selectbox('Select an option', [' - select - '] + options)

    if preselect == ' - select - ':
        st.stop()
    else:
        st.session_state.selected_station = preselect
        st.experimental_rerun()


def func2(dm):
    weather = dm['weather'].read()
    options = list(weather.keys())
    st.title('Step 2')
    st.write('Wow found the session state option')
    st.sidebar.selectbox('Select an option', options, key='selected_station')


def main(dm):
    if 'selected_station' in st.session_state:
        func2(dm)
    else:
        func1(dm)

def run(**kwargs):

    params = st.experimental_get_query_params()
    config, dm = build_config(params=params, **kwargs)

    main(dm)

    st.json(st.session_state)
    
if __name__ == '__main__':
    run()
