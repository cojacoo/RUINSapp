"""
This module implements a stripes plot function that can create annual and monthly
stripes plots from given climate and weather data. There is a matplotlib and a 
plotly version. The streamlit app uses the plotly version by default.
The plotly plot is interactive and available in a German and English version, the
matplotlib version is static and only available in English.


"""
from typing import Tuple
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def yrplot_data(sr, ref=[1980, 2000], ag='sum', qa=0.95, cmxeq=True) -> Tuple[pd.DataFrame, float, float]:
    """
    Prepare the given data for monthly and annual stripes to be plotted in a heatmap.
    The data is grouped and the limits for plotting are returned, based on the quantile given.

    """
    # plot of heatmap with monthyl and annual stripes
    yrs = sr.index.year.unique()
    dummy = np.zeros((len(yrs), 14)) * np.nan
    dummy = pd.DataFrame(dummy)
    dummy.index = yrs
    dummy.columns = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D', '', 'Year']

    for i in sr.index:
        dummy.iloc[i.year - yrs[0], i.month - 1] = sr.loc[i]
    for i in yrs:
        if ag == 'sum':
            dummy.iloc[i - yrs[0], 13] = sr.loc[sr.index.year == i].sum()
            cmx = 'coolwarm_r'
        elif ag == 'min':
            dummy.iloc[i - yrs[0], 13] = sr.loc[sr.index.year == i].min()
        elif ag == 'max':
            dummy.iloc[i - yrs[0], 13] = sr.loc[sr.index.year == i].max()
        else:  # ag == 'mean'
            dummy.iloc[i - yrs[0], 13] = sr.loc[sr.index.year == i].mean()

    if ref == None:
        pass
    else:
        refx = dummy.loc[ref[0]:ref[1]].mean(axis=0)
        dummy = dummy - refx

    if cmxeq:
        vxU = dummy.abs().quantile(qa).quantile(qa)
        vxL = -1. * vxU
    else:
        vxU = dummy.quantile(qa).quantile(qa)
        vxL = dummy.quantile(1. - qa).quantile(1. - qa)

    if ag == 'sum':
        dummy.iloc[:, 13] = dummy.iloc[:, 13] / 12

    return dummy, vxU, vxL


def _yrplot_matplotlib(sr, ref=[1980, 2000], ag='sum', qa=0.95, cbar_title='Temperature anomaly (K)', cmx='coolwarm', cmxeq=True, li=False):
    # prepare the data
    dummy, vxU, vxL = yrplot_data(sr, ref=ref, ag=ag, qa=qa, cmxeq=cmxeq)
    yrs = dummy.index.values
    # build the figure
    fig = plt.figure(figsize=(8,len(dummy)/15.))
    ax = sns.heatmap(dummy, cmap=cmx, vmin=vxL, vmax=vxU, cbar_kws={'label': cbar_title})

    if ref == None:
        pass
    else:
        ax.add_patch(
            plt.Rectangle((0, ref[0] - yrs[0]), 12, ref[1] - ref[0], fill=True, edgecolor='red', facecolor='gray', lw=2,
                          alpha=0.3, clip_on=False))
        ax.add_patch(
            plt.Rectangle((13, ref[0] - yrs[0]), 1, ref[1] - ref[0], fill=True, edgecolor='red', facecolor='gray', lw=2,
                          alpha=0.3, clip_on=False))
        ax.annotate('Reference period', (0.5, ref[1] - yrs[0] - 2), color='white', weight='bold', ha='left',
                    va='bottom', alpha=0.6)

    if type(li) == int:
        ax.axhline(li - yrs[0], color='k', ls='--', lw=1, alpha=0.5)
        ax.annotate(' >> observed', (12.5, li - yrs[0] - 0.5), color='k', ha='center', va='bottom', alpha=0.6,
                    rotation=90.)
        ax.annotate('modelled << ', (12.5, li - yrs[0] + 0.5), color='k', ha='center', va='top', alpha=0.6,
                    rotation=90.)
        # ax.add_patch(plt.Rectangle((0, li-yrs[0]), 12, 0, fill=False, edgecolor='k', ls='--', lw=1, alpha=0.5, clip_on=False))

    ax.set_ylabel('Year')
    ax.set_xlabel('Month          ')
    
    return fig


def _yrplot_plotly(sr, ref=[1980, 2000], ag='sum', qa=0.95, cbar_title='Temperature anomaly (K)', cmx='coolwarm', cmxeq=True, li=False, lang='en'):
    # prepare the data
    dummy, vxU, vxL = yrplot_data(sr, ref=ref, ag=ag, qa=qa, cmxeq=cmxeq)
    
    # use full names
    if lang == 'de':
        dummy.columns = ['Janua', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember', '', 'Jahr']
        hv_tmpl = """<b>Jahr:</b> %{y}<br><b>Monat:</b> %{x}<br><b>Abweichung:</b> %{z:.1f} K"""
        ref_txt = f'<b>Reference Period</b><br>{ref[0]} - {ref[1]}'
    else:
        dummy.columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', '', 'Year']
        hv_tmpl = """<b>Year:</b> %{y}<br><b>Month:</b> %{x}<br><b>Deviation:</b> %{z:.1f} K"""
        ref_txt = f'<b>Reference Period</b><br>{ref[0]} - {ref[1]}'
    
    fig = go.Figure()
    
    if cmx == 'coolwarm':
        lo = min(0, abs(vxL / dummy.min().min()))
        up = min(1, vxU / dummy.max().max())
        cmx = [
            [lo, '#8FBDD3'],
            [(up + lo) / 2 , '#E4D1B9'],
            [up, '#A97155'],
        ]
    # add the trace
    fig.add_trace(go.Heatmap(
        z=dummy,
        x=dummy.columns,
        y=dummy.index,
        colorscale=cmx,
        hovertemplate=hv_tmpl,
        showlegend=False,
        name=cbar_title
    ))

    # add reference window
    fig.add_shape(type='rect', x0=-0.5, y0=ref[0], x1=11.5, y1=ref[1], xref='x', fillcolor='rgba(255, 0,0,0.3)', line=dict(width=3, color='red'))
    fig.add_shape(type='rect', x0=12.5, y0=ref[0], x1=13.5, y1=ref[1], xref='x', fillcolor='rgba(255, 0,0,0.3)', line=dict(width=3, color='red'))
    fig.add_annotation(x=12, y=np.mean(ref), showarrow=False, text=ref_txt, textangle=-90, font=dict(size=16, color='red'))

    # add colorbar title
    fig.add_annotation(x=1.1, align='right', valign='top', xref='paper', yref='paper', xanchor='right', yanchor='middle', textangle=-90, text=cbar_title, showarrow=False, font=dict(size=16, color='black'))
    
    # update the layout
    fig.update_layout(
        template='plotly_white',
        height=min(len(dummy) * 15, 650),
        margin=dict(t=25, b=100, l=80, r=160, autoexpand=False),
        xaxis=dict(tickangle=45, title='Monat' if lang == 'de' else 'Month'),
        yaxis=dict(title='Jahr' if lang == 'de' else 'Year'),
    )

    # return
    return fig


def yrplot_hm(sr, ref=[1980, 2000], ag='sum', qa=0.95, cbar_title='Temperature anomaly (K)', cmx='coolwarm', cmxeq=True, li=False, lang='en', backend='plotly'):
    """
    Yearly stripes plot.
    Creates a monthly resloved stripes plot for each year in the data.
    The data is plotted as a heatmap showing the Temperature anomaly to the passed reference period
    based on the adjustable quartile value.

    Parameters
    ----------
    sr : pandas.DataFrame
        Input data originating from one ground station
    ref : Tuple[int, int]
        Reference period to calculate the anomaly.
    ag : str
        Reduction function to use. Can be used to calculate
        the temperature anomaly in 'sum', 'min', 'max' or 'mean'
    qa : float
        Quartile to use. Has to be between 0 and 1.
    cbar_title : str
        Optional title for the colorbar
    cmx : str
        Colorbar scale. Can be any string accepted by the plotting backend.
    cmxeq : bool
        If True (default), the colorbar range is calculated for 
        the full dataset.
    lang : str
        Can be either ``'en'`` or ``'de'`. The language used for axis labels.
    backend : str
        Can be either ``matplotlib`` or ``plotly``.
        The plotting backend to use.

    """
    if backend.lower() == 'matplotlib':
        return _yrplot_matplotlib(sr, ref=ref, ag=ag, qa=qa, cbar_title=cbar_title, cmx=cmx, cmxeq=cmxeq, li=li)
    else:
        return _yrplot_plotly(sr, ref=ref, ag=ag, qa=qa, cbar_title=cbar_title, cmx=cmx, cmxeq=cmxeq, li=li, lang=lang)
