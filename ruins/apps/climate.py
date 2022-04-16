import streamlit as st
import numpy as np

import xarray as xr # TODO  this should be covered by the DataManager
import pandas as pd # TODO this should be covered by the DataManager

from ruins.core import build_config, debug_view, Config, DataManager
from ruins.plotting import monthlyx
from ruins.components import topic_select, model_scale_select, data_select


# TODO refactor the plots into the plotting module
def climate_explorer(dataManager: DataManager, config: Config):
    """
    """
    # get model scale
    option_container = st.sidebar.expander('OPTIONS', expanded=True)
    model_scale_select.model_scale_selector(dataManager, config, expander_container=option_container)

    cliproj = config['climate_scale']

    # topic selector
    topic_expander = st.sidebar.expander('Topic selection', expanded=True)
    topic = topic_select.topic_select(config=config, topic_list=['Warming'], expander_container=topic_expander)

    # ----------------------
    # SELECT FIRST REFERENCE
    data_select.rcp_selector(dataManager, config, title='Select first reference', expander_container=option_container, elements='__all__', layout='columns', RCP_KEY='rcp_reference_1', allow_skip=False)
    cref1 = config['rcp_reference_1']

    if cref1=='weather':
        data1 = dataManager['weather'].read()
        data1 = data1.coast
        drngx1 = (1980,2000)
    else:
        data1 = dataManager['cordex_coast'].read()
        drngx1 = (2050, 2070)
    
    # filter for rcps
    if cref1 == 'rcp26':
        data1 = data1.filter_by_attrs(RCP='rcp26')
    elif cref1 == 'rcp45':
        data1 = data1.filter_by_attrs(RCP='rcp45')
    elif cref1 == 'rcp85':
        data1 = data1.filter_by_attrs(RCP='rcp85')

    # create the date range silder
    drng1 = [pd.to_datetime(data1.isel(time=0, vars=1).time.values).year, pd.to_datetime(data1.isel(time=-1, vars=1).time.values).year]
    datarng1 = option_container.slider('Data range', drng1[0], drng1[1], drngx1, key='dr1')

    # -----------------------
    # SELECT SECOND REFERENCE
    data_select.rcp_selector(dataManager, config, title='Select second reference', expander_container=option_container, elements='__all__', layout='row', RCP_KEY='rcp_reference_2', allow_skip=False)
    cref2 = config['rcp_reference_2']

    if cref2 == 'weather':
        data2 = dataManager['weather'].read()
        drngx2 = (1980, 2000)
    else:
        data2 = dataManager['cordex_coast'].read()
        drngx2 = (2050, 2070)
    
    # filter for rcps
    if cref2 == 'rcp26':
        data2 = data2.filter_by_attrs(RCP='rcp26')
    elif cref2 == 'rcp45':
        data2 = data2.filter_by_attrs(RCP='rcp45')
    elif cref2 == 'rcp85':
        data2 = data2.filter_by_attrs(RCP='rcp85')

    # create the date range silder
    drng2 = [pd.to_datetime(data2.isel(time=0, vars=1).time.values).year, pd.to_datetime(data2.isel(time=-1, vars=1).time.values).year]
    datarng2 = option_container.slider('Data range', drng2[0], drng2[1], drngx2, key='dr2')

    # TODO: build this into a Component
    if cliproj=='Regional':
        st.warning('Sorry, we currently have issues with the Regional model data. Please come back later.')
        st.stop()
        regaggs = ['North Sea Coast', 'Krummhörn',  'Niedersachsen', 'Inland']
        regagg = st.sidebar.selectbox('Spatial aggregation:', regaggs)

        if regagg=='North Sea Coast':
            climate = xr.load_dataset('data/cordex_coast.nc')
            climate.filter_by_attrs(RCP='rcp45')
        elif regagg=='Krummhörn':
            climate = xr.load_dataset('data/cordex_krummh.nc')

    # ------------------
    # Main Area
    plot_opts, plot_area = st.columns((2, 8))
    
    # TODO: Use the actual selected topic here
    if config['current_topic'] == 'Warming':
        navi_vars = ['Maximum Air Temperature', 'Mean Air Temperature', 'Minimum Air Temperature']
        navi_var = plot_opts.radio("Select variable:", options=navi_vars)
        if navi_var[:4] == 'Mini':
            vari = 'Tmin'
            afu = np.min
            ag = 'min'
        elif navi_var[:4] == 'Maxi':
            vari = 'Tmax'
            afu = np.max
            ag = 'max'
        else:
            vari = 'T'
            afu = np.mean
            ag = 'mean'

        # apply aggreagtion
        dyp = data1.sel(vars=vari).to_dataframe().resample('1M').apply(afu)
        dyp = dyp.loc[(dyp.index.year>=datarng1[0]) & (dyp.index.year<datarng1[1]),dyp.columns[dyp.columns!='vars']]
        dyp2 = data2.sel(vars=vari).to_dataframe().resample('1M').apply(afu)
        dyp2 = dyp2.loc[(dyp2.index.year >= datarng2[0]) & (dyp2.index.year < datarng2[1]),dyp2.columns[dyp2.columns!='vars']]

        # create the plot
        fig = monthlyx(dyp, dyp2, 'Temperature (°C)', 'Monthly '+ag+' in Year ('+cref1+')', 'Monthly '+ag+' in Year ('+cref2+')')
        st.pyplot(fig=fig)


def main_app(**kwargs):
    """@TODO: describe kwargs here"""
    # build the config and dataManager from kwargs
    url_params = st.experimental_get_query_params()
    config, dataManager = build_config(url_params=url_params, **kwargs)

    # set page properties and debug view
    st.set_page_config(page_title='Climate Explorer', layout=config.layout)
    debug_view.debug_view(dataManager, config, debug_name='Startup')

    st.header('Climate Projections Explorer')
    st.markdown('''In this section we add climate model projections to the table. The same variables and climate indices are used to explore the projections of different climate models and downscaling models. It is also possible to compare projections under different scenarios about the CO<sub>2</sub>-concentration pathways to observed weather and between different model projections.''',unsafe_allow_html=True)
    
    
    # TODO refactor this
    climate_explorer(dataManager, config)


if __name__ == '__main__':
    main_app()
