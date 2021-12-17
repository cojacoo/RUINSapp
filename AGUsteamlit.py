# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
sns.set_style('whitegrid', {'grid.linestyle': u'--'})

from sdm import SDM

#import statsmodels.api as sm
#import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

st.title('RUINS example for AGU')

def water_proj():
    #hsim = pd.read_csv('data/hsim.csv',index_col=0)
    #hsim.index = pd.to_datetime(hsim.index)
    hsim_collect = pd.read_csv('data/hsim_collect.csv', index_col=0)
    hsim_collect.index = pd.to_datetime(hsim_collect.index)
    all_vals = np.ravel(hsim_collect.values)[~np.isnan(np.ravel(hsim_collect.values))]

    perci = st.sidebar.slider('Adjust percentile of extreme events:', min_value=90., max_value=99.9999, value=99., key='perci')

    firstitem = True
    for i in hsim_collect.columns:
        hc = hsim_collect.loc[hsim_collect[i] > np.percentile(all_vals, perci), i].resample('1Y').count()
        hc.name = i
        if firstitem:
            h_count = hc
            firstitem = False
        else:
            h_count = pd.concat([h_count, hc], axis=1)

    fig = plt.figure(constrained_layout=True,figsize=(10,2.5))
    gs = fig.add_gridspec(1, 5)
    ax = fig.add_subplot(gs[0, :-1])
    ax1 = ax.twinx()
    ax2 = fig.add_subplot(gs[0, -1])

    all_vals = np.ravel(hsim_collect.values)[~np.isnan(np.ravel(hsim_collect.values))]
    sns.distplot(all_vals, ax=ax2)
    ax2.vlines(np.percentile(all_vals, perci), 0., 0.06, colors='red')
    ax2.text(np.percentile(all_vals, perci), 0.065, str(perci) + '%', c='red', ha='center')
    ax2.set_xlim(-3, np.percentile(all_vals, 99.98))

    for i in hsim_collect.columns:
        hsim_collect[i].plot(style='.', label='_nolegend_', c='gray', alpha=0.2, ax=ax)
        try:
            hsim_collect.loc[hsim_collect[i] > np.percentile(all_vals, perci), i].plot(style='.', label='_nolegend_',
                                                                                   c='red', alpha=0.2, ax=ax)
        except:
            pass

    (h_count.sum(axis=1) / len(h_count.columns)).rolling(7).mean().plot(ax=ax1, grid=False, ls='--', label='all')
    cc = [x for x in h_count.columns if x.split('.')[-1] == 'rcp45']
    (h_count[cc].sum(axis=1) / len(cc)).rolling(7).mean().plot(ax=ax1, grid=False, label='rcp45')

    cc = [x for x in h_count.columns if x.split('.')[-1] == 'rcp85']
    (h_count[cc].sum(axis=1) / len(cc)).rolling(7).mean().plot(ax=ax1, grid=False, label='rcp85')
    ax1.legend()
    ax.set_ylabel('Q (mm/day)')
    ax1.set_ylabel('days w/ Q>' + str(np.round(perci / 100., 3)) + ' percentile\n7 year rolling mean')
    st.pyplot()


ruinslogo = Image.open('RUINS_logo_small.png')

## sidebar header
st.sidebar.image(ruinslogo, caption='', use_column_width=True)
st.sidebar.header('Control Panel')
water_proj()
