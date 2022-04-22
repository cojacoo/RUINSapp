import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def yrplot_hm(sr, ref=[1980, 2000], ag='sum', qa=0.95, cbar_title='Temperature anomaly (K)', cmx='coolwarm', cmxeq=True, li=False):
    """
    @deprecated - use stripes_heatmap.yrplot_hm instead
    
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


def monthlyx(dy, dyx=1, ylab='T (°C)', clab1='Monthly Mean in Year', clab2='Monthly Max in Year', pls='cividis_r') -> plt.Figure:
    cmap = plt.cm.get_cmap(pls)
    colors = cmap(np.linspace(0, 1, len(dy.index.year.unique())+1))
    colorsx = cmap(np.arange(cmap.N))

    idx1 = dy.index.year - dy.index.year.min()
    idx1m = dy.index.month

    if type(dyx) == int:
        pass
    else:
        idx2 = (dyx.index.year - dyx.index.year.min()).astype(int)
        idx2m = dyx.index.month

        cmap1 = plt.cm.get_cmap('gist_heat_r')
        colors1 = plt.cm.gist_heat_r(np.linspace(0, 1, len(dyx.index.year.unique()) + 1))
        colorsx1 = cmap1(np.arange(cmap1.N))

    for i in dy.columns:
        plt.scatter(idx1m + (np.random.rand(len(dy)) - 1.5), dy[i].values.astype(np.float), c=colors[idx1], alpha=0.6, s=2)

    if type(dyx) == int:
        pass
    else:
        for i in dyx.columns:
            plt.scatter(idx2m + (np.random.rand(len(dyx)) - 1.5), dyx[i].values.astype(np.float), c=colors1[idx2], alpha=0.6, s=2)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label=clab1, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.set_yticklabels(np.round(np.linspace(dy.index.year.min(), dy.index.year.max(), 6)).astype(int).astype(str))

    if type(dyx) == int:
        pass
    else:
        cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap1), label=clab2, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar1.ax.set_yticklabels(
            np.round(np.linspace(dyx.index.year.min(), dyx.index.year.max(), 6)).astype(int).astype(str))

    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.ylabel(ylab)

    fig = plt.gcf()
    return fig

