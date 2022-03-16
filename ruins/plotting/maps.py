import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

from ruins.core import DataManager
from ruins.core.cache import partial_memoize


@partial_memoize(hash_names=['sel', 'cm'])
def plt_map(dataManager: DataManager, sel='all', cm='none') -> go.Figure:
    # cordex_grid = xr.open_dataset('data/CORDEXgrid.nc')
    # cimp_grid = xr.open_dataset('data/CMIP5grid.nc')
    # stats = pd.read_csv('data/stats.csv', index_col=0)
    cordex_grid = dataManager['CORDEXgrid'].read()
    cimp_grid = dataManager['CMIP5grid'].read()
    stats = dataManager['stats'].read()

    stats['ms'] = 15.
    stats['color'] = 'gray'

    mapbox_access_token = 'pk.eyJ1IjoiY29qYWNrIiwiYSI6IkRTNjV1T2MifQ.EWzL4Qk-VvQoaeJBfE6VSA'
    px.set_mapbox_access_token(mapbox_access_token)

    nodexy = pd.DataFrame([cordex_grid.lon.values.ravel(), cordex_grid.lat.values.ravel()]).T
    nodexy.columns = ['lon', 'lat']
    nodexy['hov'] = 'CORDEX grid'

    def lin_grid(fig, dummy, chex='#d95f0e', opc=0.7):
        x1, x2 = np.shape(dummy.lon)
        lond = np.diff(dummy.lon.values)
        latd = np.diff(dummy.lat.values)
        lond1 = np.diff(dummy.lon.values, axis=0)
        latd1 = np.diff(dummy.lat.values, axis=0)

        for i in np.arange(x2)[1:-1]:
            fig.add_trace(go.Scattermapbox(
                mode='lines',
                lon=dummy.lon.values[:, i] - 0.5 * lond[:, i - 1],
                lat=dummy.lat.values[:, i] - 0.5 * latd[:, i - 1],
                line={'color': chex, 'width': 1},
                hoverinfo='skip',
                opacity=opc))

        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lon=dummy.lon.values[:, i] + 0.5 * lond[:, i - 1],
            lat=dummy.lat.values[:, i] + 0.5 * latd[:, i - 1],
            line={'color': chex, 'width': 1},
            hoverinfo='skip',
            opacity=opc))

        for i in np.arange(x1)[1:-1]:
            fig.add_trace(go.Scattermapbox(
                mode='lines',
                lon=dummy.lon.values[i, :] - 0.5 * lond1[i - 1, :],
                lat=dummy.lat.values[i, :] - 0.5 * latd1[i - 1, :],
                line={'color': chex, 'width': 1},
                hoverinfo='skip',
                opacity=opc))

        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lon=dummy.lon.values[i, :] + 0.5 * lond1[i - 1, :],
            lat=dummy.lat.values[i, :] + 0.5 * latd1[i - 1, :],
            line={'color': chex, 'width': 1},
            hoverinfo='skip',
            opacity=opc))

        return fig

    def add_stats(sel='all'):


        if sel == 'all':
            ac = '#2c7fb8'
            stats1 = stats
        else:
            ac = 'gray'

            fig.add_trace(go.Scattermapbox(
                lat=stats.lat,
                lon=stats.lon,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color=ac,
                    opacity=0.8
                ),
                text=stats['Station name']
            ))

            ac = '#df65b0'
            if sel == 'krummhoern':
                stats1 = stats.loc[stats.krummhoern == True]
            elif sel == 'coast':
                stats1 = stats.loc[stats.coast == True]
            elif sel == 'niedersachsen':
                stats1 = stats.loc[stats.niedersachsen == True]
            elif sel == 'inland':
                stats1 = stats.loc[stats.inland == True]
            else:
                try:
                    stats1 = pd.DataFrame(stats.loc[sel]).T
                except:
                    ac = 'gray'
                    stats1 = pd.DataFrame(stats.loc['Norderney']).T

        fig.add_trace(go.Scattermapbox(
            lat=stats1.lat,
            lon=stats1.lon,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color=ac,
                opacity=0.8
            ),
            text=stats1['Station name']
        ))

        return fig

    def add_cmpx(cm):
        maskcordex_coast = [[10, 4], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [10, 8], [9, 8], [8, 8], [8, 9],
                            [8, 10], [9, 10], [10, 10], [10, 11], [11, 12], [12, 12], [12, 13], [12, 14], [13, 14],
                            [14, 14], [15, 14], [16, 14], [17, 14], [18, 14], [19, 15], [20, 15], [21, 14], [22, 14],
                            [22, 13],
                            [9, 4], [10, 5], [10, 6], [10, 7], [10, 8], [9, 7], [8, 7], [7, 7], [7, 8], [7, 9], [9, 9],
                            [10, 12], [11, 13], [11, 14], [14, 15], [15, 15], [16, 15], [17, 15], [18, 15], [19, 16],
                            [20, 16], [21, 15], [22, 15],
                            [11, 4], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [11, 10], [11, 11], [13, 12],
                            [13, 13], [14, 13], [15, 13], [16, 13], [17, 13], [18, 13], [19, 14], [20, 14], [21, 13]]
        for cc in maskcordex_coast:
            fig.add_trace(go.Scattermapbox(
                lat=[cordex_grid.lat.values[tuple(cc)]],
                lon=[cordex_grid.lon.values[tuple(cc)]],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=8,
                    color='#fec44f',
                    opacity=0.8
                ),
                hoverinfo='skip'
            ))
        return fig

    # fig = go.Scattermapbox(lon = nodexy.lon, lat= nodexy.lat, mode='markers', marker_symbol='square', marker_size=15)
    #fig = px.scatter_mapbox(nodexy, lat='lat', lon='lon', center={'lat': 53.0, 'lon': 8.3}, zoom=5, opacity=0.1, hover_data=['hov'])
    fig = px.scatter_mapbox(stats, lat='lat', lon='lon', center={'lat': 53.0, 'lon': 8.6}, zoom=5, size='ms', opacity=0.8, color='color', hover_data=['Station name', 'lat', 'lon'], size_max=10)
    if cm != 'none':
        fig = lin_grid(fig, cordex_grid)
        fig = lin_grid(fig, cimp_grid, '#2c7fb8')
        fig = add_cmpx(cm)
    fig = add_stats(sel)
    fig.update_layout(showlegend=False,width=300, height=350,margin=dict(l=10, r=10, b=10, t=10))  # ,center={'lat':54.0,'lon':8.3}, zoom=7)
    # st.sidebar.plotly_chart(fig)
    return fig

