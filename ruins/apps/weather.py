from faulthandler import disable
from typing import List, Callable, Tuple
import streamlit as st
import xarray as xr     # TODO: these references should be moved to DataManager
import pandas as pd     # TODO: these references should be moved to DataManager
import numpy as np
import matplotlib.pyplot as plt
from plotly.express.colors import named_colorscales

from ruins.plotting import plt_map, kde, yrplot_hm
from ruins.components import data_select, model_scale_select
from ruins.core import build_config, debug_view, DataManager, Config
from ruins.core.cache import partial_memoize
from ruins.plotting.climate_parcoords import climate_projection_parcoords
from ruins.processing.sdm import SDM


_TRANSLATE_DE_CLIMATE = dict(
    title="Unser Wetter in der Zukunft",
    introduction="""Schon die Betrachtung von aktuellen Wetterdaten zeigt, dass diese nicht immer eindeutig sind.
Unsicherheiten bei der Beobachtung von Wetterphänomenen Unwissen über Prozesse wirken sich
auf aktuellen Wetterbeschreibungen aus und erschweren z.B. die Wettervorhersage.

In der Klimaforschung müssen wir nun die nächsten **80 Jahre** vorhersagen. Neben Vorhersageunsicherheiten,
sind aber auch die **Voraussetzungen und Bedingungen** von denen wir ausgehen müssen nicht bekannt und 
auch nicht zu beziffern. 

Wie entwickelt sich die Weltbevölkerung? 

Wie viel CO2 stoßen wir in 30 Jahren aus?

Außerdem: Der Mensch wird durch Managemententscheidungen jene Bedingungen ständig ändern. Dennoch müssen wir heute 
schon Informationen bereitstellen und Abschätzungen liefern. Deshalb ist es so wichtig Wissenslücken zu
schließen und noch wichtiger, Unsicherheit durch Unwissen in Managaementprozesse einzubinden.

Mit dem **Klimamodell Explorer** können einige dieser Szenarien mit einander verglichen werden.
"""
)

_TRANSLATE_EN_CLIMATE = dict(
    title="Projecting weather into the future",
    introduction="""Schon die Betrachtung von aktuellen Wetterdaten zeigt, dass diese nicht immer eindeutig sind.
Unsicherheiten bei der Beobachtung von Wetterphänomenen Unwissen über Prozesse wirken sich
auf aktuellen Wetterbeschreibungen aus und erschweren z.B. die Wettervorhersage.

In der Klimaforschung müssen wir nun die nächsten **80 Jahre** vorhersagen. Neben Vorhersageunsicherheiten,
sind aber auch die **Voraussetzungen und Bedingungen** von denen wir ausgehen müssen nicht bekannt und 
auch nicht zu beziffern. 

Wie entwickelt sich die Weltbevölkerung? 

Wie viel CO2 stoßen wir in 30 Jahren aus?

Außerdem: Der Mensch wird durch Managemententscheidungen jene Bedingungen ständig ändern. Dennoch müssen wir heute 
schon Informationen bereitstellen und Abschätzungen liefern. Deshalb ist es so wichtig Wissenslücken zu
schließen und noch wichtiger, Unsicherheit durch Unwissen in Managaementprozesse einzubinden.

Mit dem **Klimamodell Explorer** können einige dieser Szenarien mit einander verglichen werden.
"""
)

_TRANSLATE_DE_INDICES = dict(
    title="Projektionen in einer Zahl darstellen",
    introduction="""Klimaprojektionen sind der Versuch möglichst viele verschiedene mögliche zukünftige
Szenarien mit in die Betrachtung des Klimawandels einzubeziehen. Durch verschiedene Unsicherheiten können
sich die einzelnen Modelle eines RCPs voneinander jedoch stärker unterscheiden, als zwei andere Modelle, die
sogar von unterschiedlichen Voraussetzungen ausgehen.
Hierdurch wird eine scharfe Aussage über die Folgen von Managementetscheidungen scheinbar unmöglich.
Vor allem, weil Trends in den Modellen so nur sehr schwer zu indentifizieren sind und entscheidende Änderungen
im Rauschen untergehen können.

Mit dem letzten Kapitel wird der Versuch unternommen, all die Variabilität und Unsicherheit auf eine Zahl
runterzubrechen, einem **Klimaindex**.
Dieser Index muss vor allem eine konkrete Bedeutung für einen Bestimmten **Kontext** haben, z.b. einem 
betriebswirtschaftlichen Entscheidungsprozess.

> *Soll ich in den nächsten Jahren eher Winterweizen oder Mais anbauen?*

Die Erkenntnis, dass alle Klimamodelle eine Erhöhung der Durchschnittstemperaturen in den nächsten Jahrzenten
vorhersagen ist für konkrete Entscheidungsprozesse nicht wichtig. Sondern, z.B. wenn Winterweizen weniger sensibel
auf Hitzetage reagiert, ist wichtig ob die Zahl dieser Tage erheblich zunimmt.

Für alle betrachteten Stationen in RUINS können im letzten Kapitel die Projektion der **Klimaindices** bis 2100 erforscht werden.
"""
)

_TRANSLATE_EN_INDICES = dict(
    title="Breaking down projections into one metric",
    introduction="""
"""
)

####
# OLD STUFF
def applySDM(wdata, data, meth='rel', cdf_threshold=0.9999999, lower_limit=0.1):
    '''apply structured distribution mapping to climate data and return unbiased version of dataset'''
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

    # drop NA
    ts = ts.dropna()

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

    w1 = weather[stati].sel(vars=vari).to_dataframe()
    w1.columns = ['bla', vari]

    fig = plt.figure(figsize=(10,2.5))
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
    st.pyplot(fig)

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


@partial_memoize(hash_names=['name', 'station', 'variable', 'time', '_filter'])
def _reduce_weather_data(dataManager: DataManager, name: str, variable: str, time: str, station: str = None, _filter: dict = None) -> pd.DataFrame:
    # get weather data
    arr: xr.Dataset = dataManager[name].read()

    if _filter is not None:
        arr = arr.filter_by_attrs(**_filter)

    if station is None:
        base = arr
    else:
        base = arr[station]

    # reduce to station and variable
    reduced = base.sel(vars=variable).resample(time=time)

    if variable == 'Tmax':
        df = reduced.max(dim='time').to_dataframe()
    elif variable == 'Tmin':
        df = reduced.min(dim='time').to_dataframe()
    else:
        df = reduced.mean(dim='time').to_dataframe()
    
    if station is None:
        return df.loc[:, df.columns != 'vars']
    else:
        return df[station]       


def climate_data_selector(dataManager: DataManager, config: Config, it: int = 0, variable: str = 'T', expander_container = st.sidebar, layout: str = 'columns', **kwargs):
    """Handles the selection and display of one paralell coordinates plot"""
    # create a unique key
    key = f'rcp_reference_{it}'
    title = f'Select Station #{it + 1}' if config.lang == 'en' else f'Station #{it + 1} auswählen'

    # get the container
    container = kwargs['container'] if 'container' in kwargs else st

    # if this is not the first iteration, we pre-select an item for rcp
    if it > 0:
        st.session_state[key] = 'rcp45'

    # make the data selection
    data_select.rcp_selector(
        dataManager,
        config,
        title=title,
        expander_container=expander_container,
        elements='__all__',
        layout=layout,
        RCP_KEY=key,
        allow_skip=False
    )

    # get the reference from config
    ref = config[key]

    # make the data sub-selection
    if ref == 'weather':
        data = dataManager['weather'].read()
        drngx = (1980, 2000)
    else:
        data = dataManager['cordex_coast'].read()
        drngx = (2050, 2070)

    # filter for rcps
    if ref in ('rcp26', 'rcp45', 'rcp85'):
        data = data.filter_by_attrs(RCP=ref)

    # create the data range slider
    drng = [pd.to_datetime(data.isel(time=0, vars=1).time.values).year, pd.to_datetime(data.isel(time=-1, vars=1).time.values).year]
    datarng = expander_container.slider('Date Range', drng[0], drng[1], drngx, key=f'dr{it}')

    # switch the variable
    if variable == 'T':
        afu = np.mean
    elif variable == 'Tmin':
        afu = np.min
    elif variable == 'Tmax':
        afu = np.max

    # aggregate
    dyp = data.sel(vars=variable).to_dataframe().resample('1M').apply(afu)
    dyp = dyp.loc[(dyp.index.year>=datarng[0]) & (dyp.index.year<datarng[1]),dyp.columns[dyp.columns!='vars']]

    # plot
    fig = climate_projection_parcoords(data=dyp, colorscale=kwargs.get('colorscale', 'electric'))
    container.plotly_chart(fig, use_container_width=True)

# TODO refactor the plots into the plotting module
def climate_plots(dataManager: DataManager, config: Config, expander_container = st.sidebar):
    """
    """
    cliproj = config['climate_scale']


    # TODO: build this into a Component ?
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

    # TODO: Refactor this part - similar stuff is used in weather explorer
    navi_vars = ['Maximum Air Temperature', 'Mean Air Temperature', 'Minimum Air Temperature']
    navi_var = expander_container.radio("Select variable:", options=navi_vars)
    if navi_var[:4] == 'Mini':
        vari = 'Tmin'
        ag = 'min'
    elif navi_var[:4] == 'Maxi':
        vari = 'Tmax'
        ag = 'max'
    else:
        vari = 'T'
        ag = 'mean'

    # add plots as needed
    num_of_plots = st.sidebar.number_input('# of datasets to compare', min_value=1, max_value=5, value=1)

    for it in range(int(num_of_plots)):
        # create expander
        plot_expander = st.expander(f'Temperatur (°C) Monthly {ag}', expanded=(it==num_of_plots - 1))
        left, right = plot_expander.columns((2, 8))
        left.markdown('### Options')
        opt = left.container()
        # add the colorbar as option
        cmap = left.selectbox(f'Plot #{it + 1} Colorbar', options=named_colorscales(), format_func=lambda l: l.capitalize())

        # add the Parcoords plot
        climate_data_selector(dataManager, config, it=it, variable=vari, colorscale=cmap, expander_container=opt, container=right)


def warming_data_plotter(dataManager: DataManager, config: Config):
    weather: xr.Dataset = dataManager['weather'].read()
    statios = list(weather.keys())
    stat1 = config['selected_station']

    # build the placeholders
    plot_area = st.container()
    control_left, control_right = st.columns((1, 3))

    # TODO refactor in data-aggregator and data-plotter for different time frames


    # ----
    # data-aggregator controls
    navi_vars = ['Maximum Air Temperature', 'Mean Air Temperature', 'Minimum Air Temperature']
    navi_var = control_left.radio("Select variable:", options=navi_vars)
    if navi_var[:4] == 'Mini':
        vari = 'Tmin'
        ag = 'min'
    elif navi_var[:4] == 'Maxi':
        vari = 'Tmax'
        ag = 'max'
    else:
        vari = 'T'
        ag = 'mean'

    # controls end
    # ----

    # TODO: this produces a slider but also needs some data caching
    if config['temporal_agg'] == 'Annual':
        wdata = _reduce_weather_data(dataManager, name='weather', station=config['selected_station'], variable=vari, time='1Y')
        allw = _reduce_weather_data(dataManager, name='weather', variable=vari, time='1Y')

        dataLq = float(np.floor(allw.min().quantile(0.22)))
        datamin = float(np.min([dataLq, np.round(allw.min().min(), 1)]))
        
        if config['include_climate']:
            # get the rcp
            rcp = config['current_rcp']

            data = _reduce_weather_data(dataManager, name='cordex_coast', variable=vari, time='1Y', _filter=dict(RCP=rcp))
            data_ub = applySDM(wdata, data, meth='abs')

            dataUq = float(np.ceil(data_ub.max().quantile(0.76)))
            datamax = float(np.max([dataUq, np.round(data_ub.max().max(), 1)]))
        else:
            dataUq = float(np.ceil(allw.max().quantile(0.76)))
            datamax = float(np.max([dataUq,np.round(allw.max().max(), 1)]))

        datarng = control_right.slider('Adjust data range on x-axis of plot:', min_value=datamin, max_value=datamax, value=(dataLq, dataUq), step=0.1, key='drangew')

        # -------------------
        # start plotting plot
        if config['include_climate']:
            fig, ax = kde(wdata, data_ub.mean(axis=1), split_ts=3)
        else:
            fig, ax = kde(wdata, split_ts=3)

        ax.set_title(stat1 + ' Annual ' + navi_var)
        ax.set_xlabel('T (°C)')
        ax.set_xlim(datarng[0],datarng[1])
        plot_area.pyplot(fig)

        sndstat = st.checkbox('Show second station for comparison')

        if sndstat:
            stat2 = st.selectbox('Select second station:', [x for x in statios if x != config['selected_station']])
            wdata2 = _reduce_weather_data(dataManager, name='weather', station=stat2, variable=vari, time='1Y')

            fig, ax2 = kde(wdata2, split_ts=3)
            ax2.set_title(stat2 + ' Annual ' + navi_var)
            ax2.set_xlabel('T (°C)')
            ax2.set_xlim(datarng[0],datarng[1])
            st.pyplot(fig)

        # Re-implement this as a application wide service
        # expl_md = read_markdown_file('explainer/stripes.md')
        # st.markdown(expl_md, unsafe_allow_html=True)

    elif config['temporal_agg'] == 'Monthly':
        wdata = _reduce_weather_data(dataManager, name='weather', station=config['selected_station'], variable=vari, time='1M')

        ref_yr = control_right.slider('Reference period for anomaly calculation:', min_value=int(wdata.index.year.min()), max_value=2020,value=(max(1980, int(wdata.index.year.min())), 2000))

        if config['include_climate']:
            # get the rcp
            rcp = config['current_rcp']
            data = _reduce_weather_data(dataManager, name='cordex_coast', variable=vari, time='1M', _filter=dict(RCP=rcp))

            #ub = st.sidebar.checkbox('Apply SDM bias correction',True)
            ub = True # simplify here and automatically apply bias correction

            if ub:
                data_ub = applySDM(wdata, data, meth='abs')
                fig = yrplot_hm(pd.concat([wdata.loc[wdata.index[0]:data.index[0] - pd.Timedelta('1M')], data_ub.mean(axis=1)]), ref_yr, ag, li=2006, lang=config.lang)
            else:
                fig = yrplot_hm(pd.concat([wdata.loc[wdata.index[0]:data.index[0] - pd.Timedelta('1M')], data.mean(axis=1)]), ref_yr, ag, li=2006, lang=config.lang)

            # old matplotlib include
            #plt.title(stat1 + ' ' + navi_var + ' anomaly to ' + str(ref_yr[0]) + '-' + str(ref_yr[1]))
            #plot_area.pyplot(fig)
            fig.update_layout(title = f"{stat1} {navi_var} anomaly to {ref_yr[0]}-{ref_yr[1]}")
            plot_area.plotly_chart(fig, use_container_width=True)

            # compare to second station
            sndstat = st.checkbox('Compare to a second station?')
            
            if sndstat:
                stat2 = st.selectbox('Select second station:', [x for x in statios if x != stat1])
                wdata2 = _reduce_weather_data(dataManager, name='weather', station=stat2, variable=vari, time='1M')

                #ub = st.sidebar.checkbox('Apply SDM bias correction',True)
                ub = True # simplify here and automatically apply bias correction

                if ub:
                    data2_ub = applySDM(wdata2, data, meth='abs')
                    fig = yrplot_hm(pd.concat([wdata2.loc[wdata2.index[0]:data.index[0] - pd.Timedelta('1M')], data2_ub.mean(axis=1)]), ref_yr, ag, li=2006, lang=config.lang)
                else:
                    fig = yrplot_hm(pd.concat([wdata2.loc[wdata2.index[0]:data.index[0] - pd.Timedelta('1M')], data.mean(axis=1)]), ref_yr, ag, li=2006, lang=config.lang)
                
                fig.update_layout(title=f"{stat2} {navi_var} anomaly to {ref_yr[0]}-{ref_yr[1]}")
                plot_area.plotly_chart(fig, use_container_width=True)

        # TODO: break up this as well
        else:
            fig = yrplot_hm(sr=wdata, ref=ref_yr, ag=ag, lang=config.lang)
            fig.update_layout(title = f"{stat1} {navi_var} anomaly to {ref_yr[0]}-{ref_yr[1]}")
            plot_area.plotly_chart(fig, use_container_width=True)

            # old matplotlib include
            #plt.title(stat1 + ' ' + navi_var + ' anomaly to ' + str(ref_yr[0]) + '-' + str(ref_yr[1]))
            #plot_area.pyplot(fig)

            sndstat = st.checkbox('Compare to a second station?')

            if sndstat:
                stat2 = st.selectbox('Select second station:', [x for x in statios if x != stat1])
                data2 = _reduce_weather_data(dataManager, name='weather', station=stat2, variable=vari, time='1M')

                ref_yr2 = list(ref_yr)
                if ref_yr2[1]<data2.index.year.min():
                    ref_yr2[0] = data2.index.year.min()
                    ref_yr2[1] = ref_yr2[0]+10
                if ref_yr2[0]<data2.index.year.min():
                    ref_yr2[0] = data2.index.year.min()
                    if ref_yr2[1] - ref_yr2[0] < 10:
                        ref_yr2[1] = ref_yr2[0] + 10

                fig = yrplot_hm(sr=data2, ref=ref_yr2, ag=ag, lang=config.lang)
                fig.update_layout(title=f"{stat2} {navi_var} anomaly to {ref_yr2[0]}-{ref_yr2[1]}")
                plot_area.plotly_chart(fig, use_container_width=True)


def quick_access_buttons(config: Config, container = st.sidebar):
    """Add quick access button to skip parts of the Weather explorer"""
    # get the current stage
    stage = config.get('quick_access')

    # make columns
    l, r = container.columns(2)

    # make translations
    if config.lang == 'de':
        lab_weather = 'Wetterdaten Explorer'
        lab_climate = 'Klimamodell Explorer'
        lab_index = 'Klimadaten Indices'
    else:
        lab_weather = 'Weather explorer'
        lab_climate = 'Climate explorer'
        lab_index = 'Climate indices'

    # switch the cases
    if stage == 'climate':
        go_weather = l.button(lab_weather)
        go_climate = False
        go_idx = r.button(lab_index)
    elif stage == 'index':
        go_weather = l.button(lab_weather)
        go_climate = r.button(lab_climate)
        go_idx = False
    else:
        go_weather = False
        go_climate = l.button(lab_climate)
        go_idx = r.button(lab_index)

    # check if the Weather explorer is needed
    if go_weather:
        st.session_state.quick_access = 'weather'
        st.experimental_rerun()
    
    # check if the Climate explorer is needed
    if go_climate:
        if 'include_climate' in st.session_state:
            del st.session_state['include_climate']
        st.session_state.quick_access = 'climate'
        st.experimental_rerun()
    
    # check if the Climate indices are needed
    if go_idx:
        if 'include_climate' in st.session_state:
            del st.session_state['include_climate']
        st.session_state.quick_access = 'index'
        st.experimental_rerun()


def weather_stage(dataManager: DataManager, config: Config, data_expander=st.sidebar):
    # Story mode - go through each setting
    # update session state with current data settings
    data_expander = st.sidebar.expander('Data selection', expanded=True)
    data_select.data_select(dataManager, config, expander_container=data_expander, container=st)
    
    
    # build the app
    st.header('Weather Data Explorer')
    # TODO: move this text also into story mode?
    st.markdown('''In this section we provide visualisations to explore changes in observed weather data. Based on different variables and climate indices it is possible to investigate how climate change manifests itself in different variables, at different stations and with different temporal aggregation.''',unsafe_allow_html=True)

    warming_data_plotter(dataManager, config)

    # transition page
    st.markdown("""<hr style="margin-top: 4rem; margin-bottom: 2rem;" />""", unsafe_allow_html=True)
    st.success('Even about the present there is uncertainty! What about the future?')
    ok = st.button('LEARN MORE')

    if ok:
        st.session_state.quick_access = 'transition_climate'
        st.experimental_rerun()


def climate_stage(dataManager: DataManager, config: Config):
    # Story mode - go through each setting

    # update session state with current settings
    # get model scale
    option_container = st.sidebar.expander('OPTIONS', expanded=True)
    model_scale_select.model_scale_selector(dataManager, config, expander_container=option_container)
    
    # run main visualization
    climate_plots(dataManager, config, expander_container=option_container)

    # transition page
    st.markdown("""<hr style="margin-top: 4rem; margin-bottom: 2rem;" />""", unsafe_allow_html=True)
    st.success('How do we make sense of this? Can we identify trends?')
    ok = st.button('LEARN MORE')

    if ok:
        st.session_state.quick_access = 'transition_index'
        st.experimental_rerun()


def indices_stage(dataManager: DataManager, config: Config, data_expander=st.sidebar):
    # Story mode - go through each setting
    # update session state with current data settings
    data_expander = st.sidebar.expander('Data selection', expanded=True)
    data_select.selected_station_selector(dataManager, config, expander_container=data_expander)
    data_select.rcp_selector(dataManager, config, expander_container=data_expander)

    # run actual visualization
    climate_indices(dataManager, config)


def transition_page(config: Config) -> None:
    """
    This Transition is shown when the user switches from weather explorer to
    climate projections or further to climate indices, without using the quick access buttons.
    The page can be used to present a primer how the two topics are related.
    """
    # check with transition page is needed
    if config['quick_access'] == 'transition_climate':
        t = config.translator(de=_TRANSLATE_DE_CLIMATE, en=_TRANSLATE_EN_CLIMATE)
        next_stage = 'climate'
    elif config['quick_access'] == 'transition_index':
        t = config.translator(de=_TRANSLATE_DE_INDICES, en=_TRANSLATE_EN_INDICES)
        next_stage = 'index'
    
    # build the page
    st.header(t('title'))
    st.markdown(t('introduction'), unsafe_allow_html=True)

    # add continue button
    ok = st.button('WEITER' if config.lang=='de' else 'CONTINUE')

    if ok:
        st.session_state.quick_access = next_stage
        st.experimental_rerun()
    else:
        st.stop()


def main_app(**kwargs):
    """Describe the params in kwargs here

    The main app has three 'stages': 
      
      * learning about weather
      * learning about climate
      * condensing info into climate indices

    
    """
    # build the config and dataManager from kwargs
    url_params = st.experimental_get_query_params()
    config, dataManager = build_config(url_params=url_params, **kwargs)

    # set page properties and debug view    
    st.set_page_config(page_title='Weather Explorer', layout=config.layout)
    debug_view.debug_view(dataManager, config, debug_name='DEBUG - initial state')

    # check if a stage was set
    if not config.has_key('quick_access'):
        st.session_state.quick_access = 'weather'
    stage = config['quick_access']

    # add the skip buttons
    btn_expander = st.sidebar.expander('QUICK ACCESS', expanded=True)
    quick_access_buttons(config, container=btn_expander)


    # -------------
    # Weather Stage
    if stage == 'weather':
        weather_stage(dataManager, config)
    elif stage == 'climate':
        climate_stage(dataManager, config)
    elif stage == 'index':
        indices_stage(dataManager, config)
    elif stage.startswith('transition'):
        transition_page(config)
    else:
        st.error(f"We received weird data. A quick_access='{stage}' does not exist. Please contact the developer.")
        st.stop()
        
    # end state debug
    debug_view.debug_view(dataManager, config, debug_name='DEBUG - finished app')


if __name__ == '__main__':
    import fire
    fire.Fire(main_app)
