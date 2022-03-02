import streamlit as st
import numpy as np

import xarray as xr # TODO  this should be covered by the DataManager
import pandas as pd # TODO this should be covered by the DataManager

from ruins.plotting import monthlyx
from ruins import components

# TODO: Build this into DataManager
def load_data(sel='Weather',regagg=None):
    #Read data from netcdf files and return xarray
    if sel == 'Weather':
        data = xr.load_dataset('data/weather.nc')
    elif sel == 'CORDEX':
        data = xr.load_dataset('data/cordex_coast.nc')
    else:
        if regagg == 'North Sea Coast':
            data = xr.load_dataset('data/cordex_coast.nc')
            data.filter_by_attrs(RCP=sel)

    return data


# TODO refactor the plots into the plotting module
def climate_explorer(w_topic: str):
    cliprojs = ["Global", "Regional"]
    cliproj = st.sidebar.radio("Climate Model Scaling:", options=cliprojs)

    # TODO: Re-implement this as a service
    # expl_md = read_markdown_file('explainer/climatescale.md')
    # st.sidebar.markdown(expl_md, unsafe_allow_html=True)

    if cliproj=='Regional':
        regaggs = ['North Sea Coast', 'Krummhörn',  'Niedersachsen', 'Inland']
        regagg = st.sidebar.selectbox('Spatial aggregation:', regaggs)

        if regagg=='North Sea Coast':
            climate = xr.load_dataset('data/cordex_coast.nc')
            climate.filter_by_attrs(RCP='rcp45')
        elif regagg=='Krummhörn':
            climate = xr.load_dataset('data/cordex_krummh.nc')

    st.subheader('Climate Model Comparison')
    crefs = ['Weather Data', 'all RCPs', 'RCP2.6', 'RCP4.5','RCP8.5']
    cref1 = st.sidebar.selectbox('Select first reference:', options=crefs)

    if cref1=='Weather Data':
        data1 = load_data('Weather')
        data1 = data1.coast
        drngx1 = (1980,2000)
    elif cref1 == 'all RCPs':
        data1 = load_data('CORDEX')
        drngx1 = (2050, 2070)
    elif cref1 == 'RCP2.6':
        data1 = load_data('rcp26','North Sea Coast')
        drngx1 = (2050, 2070)
    elif cref1 == 'RCP4.5':
        data1 = load_data('rcp45', 'North Sea Coast')
        drngx1 = (2050, 2070)
    elif cref1 == 'RCP8.5':
        data1 = load_data('rcp85','North Sea Coast')
        drngx1 = (2050, 2070)

    drng1 = [pd.to_datetime(data1.isel(time=0, vars=1).time.values).year, pd.to_datetime(data1.isel(time=-1, vars=1).time.values).year]
    datarng1 = st.sidebar.slider('Data range', drng1[0], drng1[1], drngx1, key='dr1')

    cref2 = st.sidebar.selectbox('Select second reference:', options=crefs)
    if cref2 == 'Weather Data':
        data2 = load_data('Weather')
        drngx2 = (1980, 2000)
    elif cref2 == 'all RCPs':
        data2 = load_data('CORDEX')
        drngx2 = (2050, 2070)
    elif cref2 == 'RCP2.6':
        data2 = load_data('rcp26','North Sea Coast')
        drngx2 = (2050, 2070)
    elif cref2 == 'RCP4.5':
        data2 = load_data('rcp45', 'North Sea Coast')
        drngx2 = (2050, 2070)
    elif cref2 == 'RCP8.5':
        data2 = load_data('rcp85','North Sea Coast')
        drngx2 = (2050, 2070)

    drng2 = [pd.to_datetime(data2.isel(time=0, vars=1).time.values).year, pd.to_datetime(data2.isel(time=-1, vars=1).time.values).year]
    datarng2 = st.sidebar.slider('Data range', drng2[0], drng2[1], drngx2, key='dr2')

    if w_topic == 'Warming':
        navi_vars = ['Maximum Air Temperature', 'Mean Air Temperature', 'Minimum Air Temperature']
        navi_var = st.sidebar.radio("Select variable:", options=navi_vars)
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

        dyp = data1.sel(vars=vari).to_dataframe().resample('1M').apply(afu)
        dyp = dyp.loc[(dyp.index.year>=datarng1[0]) & (dyp.index.year<datarng1[1]),dyp.columns[dyp.columns!='vars']]
        dyp2 = data2.sel(vars=vari).to_dataframe().resample('1M').apply(afu)
        dyp2 = dyp2.loc[(dyp2.index.year >= datarng2[0]) & (dyp2.index.year < datarng2[1]),dyp2.columns[dyp2.columns!='vars']]

        monthlyx(dyp, dyp2, 'Temperature (°C)', 'Monthly '+ag+' in Year ('+cref1+')', 'Monthly '+ag+' in Year ('+cref2+')')
        st.pyplot()


def main_app():
    st.header('Climate Projections Explorer')
    st.markdown('''In this section we add climate model projections to the table. The same variables and climate indices are used to explore the projections of different climate models and downscaling models. It is also possible to compare projections under different scenarios about the CO<sub>2</sub>-concentration pathways to observed weather and between different model projections.''',unsafe_allow_html=True)
    
    
    # TODO: refactor this
    topics = ['Warming', 'Weather Indices', 'Drought/Flood', 'Agriculture', 'Extreme Events', 'Wind Energy']
    
    # topic selector
    topic = components.topic_selector(topic_list=topics, container=st.sidebar)
    
    # TODO refactor this
    climate_explorer(topic)


if __name__ == '__main__':
    main_app()
