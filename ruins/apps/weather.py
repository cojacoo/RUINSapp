import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np

from ruins.plotting import plt_map



####
# OLD STUFF
# 
# TODO: replace with DataManager
def load_alldata():
    weather = xr.load_dataset('data/weather.nc')
    climate = xr.load_dataset('data/cordex_coast.nc')

    # WARNING - bug fix for now:
    # 'HadGEM2-ES' model runs are problematic and will be removed for now
    # The issue is with the timestamp and requires revision of the ESGF reading routines
    kys = [s for s in list(climate.keys()) if 'HadGEM2-ES' not in s] #remove all entries of HadGEM2-ES (6 entries)
    climate = climate[kys]

    return weather, climate


def weather_explorer():
    weather, _ = load_alldata()
    #weather = load_data('Weather')

    #aspects = ['Annual', 'Monthly', 'Season']
    #w_aspect = st.sidebar.selectbox('Temporal aggegate:', aspects)

    #cliproj = st.sidebar.checkbox('add climate projections',False)

    statios = list(weather.keys())
    stat1 = st.selectbox('Select station/group (see map in sidebar for location):', statios)

    aspects = ['Annual', 'Monthly']  # , 'Season']
    w_aspect = st.selectbox('Select temporal aggegate:', aspects)

    cliproj = st.checkbox('add climate projections (for coastal region)',False)
    if cliproj:
        plt_map(stat1, 'CORDEX')
        st.sidebar.markdown(
            '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>). The climate model grid is given in <span style="color:orange">orange</span> with the selected references as filled dots.''',
            unsafe_allow_html=True)
    else:
        plt_map(stat1)
        st.sidebar.markdown(
            '''Map with available stations (<span style="color:blue">blue dots</span>) and selected reference station (<span style="color:magenta">magenta highlight</span>).''',
            unsafe_allow_html=True)

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

        if w_aspect == 'Annual':
            wdata = weather[stat1].sel(vars=vari).resample(time='1Y').apply(afu).to_dataframe()[stat1]
            wdata = wdata[~np.isnan(wdata)]
            allw = weather.sel(vars=vari).resample(time='1Y').apply(afu).to_dataframe().iloc[:, 1:]

            dataLq = float(np.floor(allw.min().quantile(0.22)))
            datamin = float(np.min([dataLq, np.round(allw.min().min(), 1)]))
            if cliproj:
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

            if cliproj:
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

            expl_md = read_markdown_file('explainer/stripes.md')
            st.markdown(expl_md, unsafe_allow_html=True)

        elif w_aspect == 'Monthly':
            wdata = weather[stat1].sel(vars=vari).resample(time='1M').apply(afu).to_dataframe()[stat1]
            wdata = wdata[~np.isnan(wdata)]
            ref_yr = st.slider('Reference period for anomaly calculation:', min_value=int(wdata.index.year.min()), max_value=2020,value=(max(1980, int(wdata.index.year.min())), 2000))

            if cliproj:
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

            expl_md = read_markdown_file('explainer/stripes_m.md')
            st.markdown(expl_md, unsafe_allow_html=True)

    elif w_topic == 'Weather Indices':
        climate_indices(stat1,cliproj)

    if cliproj:
        st.markdown(
            '''RCPs are scenarios about possible greenhouse gas concentrations by the year 2100. RCP2.6 is a world in which little further greenhouse gasses are emitted -- similar to the Paris climate agreement from 2015. RCP8.5 was intendent to explore a rather risky, more worst-case future with further increased emissions. RCP4.5 is one candidate of a more moderate greenhouse gas projection, which might be more likely to resemble a realistic situation. It is important to note that the very limited differentiation between RCP scenarios have been under debate for several years. One outcome is the definition of Shared Socioeconomic Pathways (SSPs) for which today, however, there are not very many model runs awailable. For more information, please check with the [Climatescenario Primer](https://climatescenarios.org/primer/), [CarbonBrief](https://www.carbonbrief.org/explainer-how-shared-socioeconomic-pathways-explore-future-climate-change) and this [NatureComment](https://www.nature.com/articles/d41586-020-00177-3)''',
            unsafe_allow_html=True)


def main_app():
    # TODO refactor this
    weather_explorer()


if __name__ == '__main__':
    main_app()
