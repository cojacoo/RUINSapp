from typing import List, Callable, Tuple
import streamlit as st
import xarray as xr     # TODO: these references should be moved to DataManager
import pandas as pd     # TODO: these references should be moved to DataManager
import numpy as np
import matplotlib.pyplot as plt

from ruins.plotting import plt_map, kde, yrplot_hm
from ruins import components
from ruins.core import build_config, DataManager, Config
from ruins.core.cache import partial_memoize



####
# OLD STUFF
def applySDM(wdata, data, meth='rel', cdf_threshold=0.9999999, lower_limit=0.1):
    '''apply structured distribution mapping to climate data and return unbiased version of dataset'''
    from sdm import SDM
    data_ub = data.copy()

    for k in data_ub.columns:
        data_col = data_ub[k].dropna()
        overlapx = pd.concat(
            [wdata.loc[data_col.index[0]:wdata.index[-1]], data_col.loc[data_col.index[0]:wdata.index[-1]]], axis=1)
        overlapx.columns = ['obs', 'cm']
        overlapx = overlapx.dropna()
        try:
            data_ub[k] = SDM(overlapx.obs, overlapx.cm, data_col, meth, cdf_threshold, lower_limit)
        except:
            data_ub[k] = data_ub[k] * np.nan

    data_ub[data_ub == 0.0000000] = np.nan
    data_ub = data_ub.loc[data_ub.index[0]:pd.to_datetime('2099-12-31 23:59:59')]

    return data_ub


def climate_indi(ts, indi='Summer days (Tmax ≥ 25°C)'):
    '''
    Calculate climate indicator days.
    Input time series of meteorological data
    '''
    if pd.infer_freq(ts.index) != 'D':
        print('Please provide daily data.')
        return

    if indi == 'Summer days (Tmax ≥ 25°C)':  # summer days
        return (ts.Tmax >= 25.).groupby(ts.index.year).sum()
    elif indi == 'Ice days (Tmax < 0°C)':  # ice days
        return (ts.Tmax < 0.).groupby(ts.index.year).sum()
    elif indi == 'Frost days (Tmin < 0°C)':  # frost days
        return (ts.Tmin < 0.).groupby(ts.index.year).sum()
    elif indi == 'Hot days (Tmax ≥ 30°C)':  # hot days
        return (ts.Tmax >= 30.).groupby(ts.index.year).sum()
    elif indi == 'Tropic nights (Tmin ≥ 20°C)':  # tropic night
        return (ts.Tmin >= 20.).groupby(ts.index.year).sum()
    elif indi == 'Rainy days (Precip ≥ 1mm)':  # rainy days
        return (ts.Prec >= 1.).groupby(ts.index.year).sum()
    else:
        print('Nothing calculated.')
        return

# TODO: document + signature
# TODO: extract plotting
def climate_indices(dataManager: DataManager, config: Config):
    # get data
    weather = dataManager['weather'].read()
    climate = dataManager['cordex_coast'].read()

    # get the relevant settings
    stati = config.get('selected_station', 'coast')


    cindi = ['Ice days (Tmax < 0°C)', 'Frost days (Tmin < 0°C)', 'Summer days (Tmax ≥ 25°C)', 'Hot days (Tmax ≥ 30°C)','Tropic nights (Tmin ≥ 20°C)', 'Rainy days (Precip ≥ 1mm)']
    ci_topic = st.selectbox('Select Index:', cindi)

    if ci_topic == 'Rainy days (Precip ≥ 1mm)':
        vari = 'Prec'
        meth = 'rel'
    elif (ci_topic == 'Frost days (Tmin < 0°C)') | (ci_topic == 'Tropic nights (Tmin ≥ 20°C)'):
        vari = 'Tmin'
        meth = 'abs'
    else:
        vari = 'Tmax'
        meth = 'abs'

    w1 = weather[stati].sel(vars=vari).to_dataframe().dropna()
    w1.columns = ['bla', vari]

    plt.figure(figsize=(10,2.5))
    wi = climate_indi(w1, ci_topic).astype(int)
    wi.plot(style='.', color='steelblue', label='Coast weather')
    wi.rolling(10, center=True).mean().plot(color='steelblue', label='Rolling mean\n(10 years)')

    if config['include_climate']:
        c1 = climate.sel(vars=vari).to_dataframe()
        c1 = c1[c1.columns[c1.columns != 'vars']]
        c2 = applySDM(w1[vari], c1, meth=meth)

        firstitem = True
        for j in c2.columns:
            dummyt = pd.DataFrame(c2[j])
            dummyt.columns = [vari]
            ci = pd.DataFrame(climate_indi(dummyt, ci_topic).astype(int))
            ci.columns = [j]

            if firstitem:
                ci[j].plot(style='.', color='gray', alpha=0.1, label='all climate projections')
                cid = ci
                firstitem = False
            else:
                ci[j].plot(style='.', color='gray', alpha=0.1, label='_')
                cid = pd.concat([cid, ci], axis=1)

        rcpx = np.array([x[-5:] for x in cid.columns.values])
        for n in np.unique(rcpx):
            nx = n
            if n == np.unique(rcpx)[-1]:
                nx = n + '\n(Rolling of 5 years)'
            cid[cid.columns[rcpx == n]].mean(axis=1).rolling(5, center=True).mean().plot(label='Mean of ' + nx)

    plt.legend(ncol=2)
    plt.ylabel('Number of days')
    plt.title(ci_topic)
    st.pyplot()

    if ci_topic == 'Ice days (Tmax < 0°C)':
        st.markdown('''Number of days in one year which persistently remain below 0°C air temperature.''')
    elif ci_topic == 'Frost days (Tmin < 0°C)':
        st.markdown('''Number of days in one year which reached below 0°C air temperature.''')
    elif ci_topic == 'Summer days (Tmax ≥ 25°C)':
        st.markdown('''Number of days in one year which reached or exceeded 25°C air temperature.''')
    elif ci_topic == 'Hot days (Tmax ≥ 30°C)':
        st.markdown('''Number of days in one year which reached or exceeded 30°C air temperature.''')
    elif ci_topic == 'Tropic nights (Tmin ≥ 20°C)':
        st.markdown('''Number of days in one year which persistently remained above 20°C air temperature.''')
    elif ci_topic == 'Rainy days (Precip ≥ 1mm)':
        st.markdown('''Number of days in one year which received at least 1 mm precipitation.''')
    return


def data_select(dataManager: DataManager, config: Config, container=st) -> None:
    """Create the user interface to control the data view
    """
    # get a station list
    weather = dataManager['weather'].read()
    station_list = list(weather.keys())
    selected_station = container.selectbox('Select station/group (see map in sidebar for location):', station_list)

    # select a temporal aggregation
    aggregations = config.get('temporal_aggregations', ['Annual', 'Monthly'])
    temp_agg = container.selectbox('Select temporal aggregation:', aggregations)

    # include climate projections
    include_climate = container.checkbox('Include climate projections (for coastal region)', value=False)

    # add settings
    st.session_state.selected_station = selected_station
    st.session_state.temp_agg = temp_agg
    st.session_state.include_climate = include_climate


@partial_memoize(hash_names=['reducer', 'station', 'variable', 'time'])
def _reduce_weather_data(dataManager: DataManager, reducer: Callable, station: str, variable: str, time: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # get weather data
    weather = dataManager['weather'].read()

    # reduce to station and variable
    reduced = weather[station].sel(vars=variable).resample(time=time).apply(reducer)

    # change to dataframe
    wdata = reduced.to_dataframe()[station]
    wdata = wdata[~np.isnan(wdata)]
    
    # reduce for all stations
    reduced = weather.sel(vars=variable).resample(time=time).apply(reducer)
    reference = reduced.to_dataframe().iloc[:, 1:]

    return wdata, reference


def warming_data_plotter(dataManager: DataManager, config: Config):
    weather: xr.Dataset = dataManager['weather'].read()
    climate = dataManager['cordex_coast'].read()
    statios = dataManager['stats'].read()
    stat1 = config['selected_station']

    # TODO refactor in data-aggregator and data-plotter for different time frames

    # ----
    # data-aggregator controls
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

    # controls end
    # ----

    # TODO: this produces a slider but also needs some data caching
    if config['temp_agg'] == 'Annual':
        wdata, allw = _reduce_weather_data(dataManager, afu, config['selected_station'], vari, '1Y')
        
        dataLq = float(np.floor(allw.min().quantile(0.22)))
        datamin = float(np.min([dataLq, np.round(allw.min().min(), 1)]))
        
        if config['include_climate']:
            rcps = ['rcp26', 'rcp45', 'rcp85']
            rcp = st.selectbox(
                'RCP (Mean over all projections will be shown. For more details go to section "Climate Projections"):',
                rcps)

            data = climate.filter_by_attrs(RCP=rcp).sel(vars=vari).resample(time='1Y').apply(afu).to_dataframe()
            data = data[data.columns[data.columns != 'vars']]
            data_ub = applySDM(wdata, data, meth='abs')

            dataUq = float(np.ceil(data_ub.max().quantile(0.76)))
            datamax = float(np.max([dataUq, np.round(data_ub.max().max(), 1)]))
        else:
            dataUq = float(np.ceil(allw.max().quantile(0.76)))
            datamax = float(np.max([dataUq,np.round(allw.max().max(), 1)]))

        datarng = st.slider('Adjust data range on x-axis of plot:', min_value=datamin, max_value=datamax, value=(dataLq, dataUq), step=0.1, key='drangew')

        # -------------------
        # start plotting plot
        if config['include_climate']:
            ax = kde(wdata, data_ub.mean(axis=1), split_ts=3)
        else:
            ax = kde(wdata, split_ts=3)

        ax.set_title(stat1 + ' Annual ' + navi_var)
        ax.set_xlabel('T (°C)')
        ax.set_xlim(datarng[0],datarng[1])
        st.pyplot()

        sndstat = st.checkbox('Show second station for comparison')

        if sndstat:
            stat2 = st.selectbox('Select second station:', [x for x in statios if x != stat1])
            wdata2 = weather[stat2].sel(vars=vari).resample(time='1Y').apply(afu).to_dataframe()[stat2]
            ax2 = kde(wdata2, split_ts=3)
            ax2.set_title(stat2 + ' Annual ' + navi_var)
            ax2.set_xlabel('T (°C)')
            ax2.set_xlim(datarng[0],datarng[1])
            st.pyplot()

        # Re-implement this as a application wide service
        # expl_md = read_markdown_file('explainer/stripes.md')
        # st.markdown(expl_md, unsafe_allow_html=True)

    elif config['temp_agg'] == 'Monthly':
        wdata = weather[stat1].sel(vars=vari).resample(time='1M').apply(afu).to_dataframe()[stat1]
        wdata = wdata[~np.isnan(wdata)]
        ref_yr = st.slider('Reference period for anomaly calculation:', min_value=int(wdata.index.year.min()), max_value=2020,value=(max(1980, int(wdata.index.year.min())), 2000))

        if config['include_climate']:
            rcps = ['rcp26', 'rcp45', 'rcp85']
            rcp = st.selectbox('RCP (Mean over all projections will be shown. For more details go to section "Climate Projections"):', rcps)

            data = climate.filter_by_attrs(RCP=rcp).sel(vars=vari).resample(time='1M').apply(afu).to_dataframe()
            data = data[data.columns[data.columns != 'vars']]

            #ub = st.sidebar.checkbox('Apply SDM bias correction',True)
            ub = True # simplify here and automatically apply bias correction

            if ub:
                data_ub = applySDM(wdata, data, meth='abs')
                yrplot_hm(pd.concat([wdata.loc[wdata.index[0]:data.index[0] - pd.Timedelta('1M')], data_ub.mean(axis=1)]),ref_yr, ag, li=2006)
            else:
                yrplot_hm(pd.concat([wdata.loc[wdata.index[0]:data.index[0] - pd.Timedelta('1M')], data.mean(axis=1)]), ref_yr, ag, li=2006)

            plt.title(stat1 + ' ' + navi_var + ' anomaly to ' + str(ref_yr[0]) + '-' + str(ref_yr[1]))
            st.pyplot()


        # TODO: break up this as well
        else:
            yrplot_hm(wdata,ref_yr,ag)
            plt.title(stat1 + ' ' + navi_var + ' anomaly to ' + str(ref_yr[0]) + '-' + str(ref_yr[1]))
            st.pyplot()

            sndstat = st.checkbox('Compare to a second station?')

            if sndstat:
                stat2 = st.selectbox('Select second station:', [x for x in statios if x != stat1])
                data2 = weather[stat2].sel(vars=vari).resample(time='1M').apply(afu).to_dataframe()[stat2]
                data2 = data2[~np.isnan(data2)]

                ref_yr2 = list(ref_yr)
                if ref_yr2[1]<data2.index.year.min():
                    ref_yr2[0] = data2.index.year.min()
                    ref_yr2[1] = ref_yr2[0]+10
                if ref_yr2[0]<data2.index.year.min():
                    ref_yr2[0] = data2.index.year.min()
                    if ref_yr2[1] - ref_yr2[0] < 10:
                        ref_yr2[1] = ref_yr2[0] + 10

                yrplot_hm(data2, ref_yr2, ag)
                plt.title(stat2 + ' ' + navi_var + ' anomaly to ' + str(ref_yr2[0]) + '-' + str(ref_yr2[1]))
                st.pyplot()

        # Re-implement this as a application wide service
        # expl_md = read_markdown_file('explainer/stripes_m.md')
        # st.markdown(expl_md, unsafe_allow_html=True)


def weather_explorer(config: Config, dataManager: DataManager):
    """
    TODO: refactor this whole app into the main_app
    """
    # update session state with current data settings
    data_select(dataManager, config, container=st)

    # check config
    if config['include_climate']:
        fig = plt_map(dataManager, sel=config['selected_station'], cm='CORDEX')
        st.sidebar.plotly_chart(fig)
        st.sidebar.markdown(
            '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>). The climate model grid is given in <span style="color:orange">orange</span> with the selected references as filled dots.''',
            unsafe_allow_html=True)
    else:
        fig = plt_map(dataManager, sel=config['selected_station'])
        st.sidebar.plotly_chart(fig)
        st.sidebar.markdown(
            '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>).''',
            unsafe_allow_html=True)

    # switch the topic
    topic = config['current_topic']
    if topic == 'Warming':
        warming_data_plotter(dataManager, config)
    
    elif topic == 'Weather Indices':
        climate_indices(dataManager, config)

    if config['include_climate']:
        st.markdown(
            '''RCPs are scenarios about possible greenhouse gas concentrations by the year 2100. RCP2.6 is a world in which little further greenhouse gasses are emitted -- similar to the Paris climate agreement from 2015. RCP8.5 was intendent to explore a rather risky, more worst-case future with further increased emissions. RCP4.5 is one candidate of a more moderate greenhouse gas projection, which might be more likely to resemble a realistic situation. It is important to note that the very limited differentiation between RCP scenarios have been under debate for several years. One outcome is the definition of Shared Socioeconomic Pathways (SSPs) for which today, however, there are not very many model runs awailable. For more information, please check with the [Climatescenario Primer](https://climatescenarios.org/primer/), [CarbonBrief](https://www.carbonbrief.org/explainer-how-shared-socioeconomic-pathways-explore-future-climate-change) and this [NatureComment](https://www.nature.com/articles/d41586-020-00177-3)''',
            unsafe_allow_html=True)


def main_app(**kwargs):
    """Describe the params in kwargs here
    """
    # build the config and dataManager from kwargs
    url_params = st.experimental_get_query_params()
    config, dataManager = build_config(url_params=url_params, **kwargs)

    st.set_page_config(page_title='Weather Explorer', layout=config.layout)

    # config expander
    exp = st.sidebar.expander('Data select and config', expanded=False)

    # build the app
    st.header('Weather Data Explorer')
    st.markdown('''In this section we provide visualisations to explore changes in observed weather data. Based on different variables and climate indices it is possible to investigate how climate change manifests itself in different variables, at different stations and with different temporal aggregation.''',unsafe_allow_html=True)

    # topic selector
    components.topic_selector(config=config, container=st.sidebar, config_expander=exp)
    
    # TODO refactor this
    weather_explorer(config, dataManager)


if __name__ == '__main__':
    import fire
    fire.Fire(main_app)
