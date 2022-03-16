import numpy as np
import matplotlib.pyplot as plt


def kde(data, cmdata='none', split_ts=1, cplot=True, eq_period=True):
    # plot of kde with stripes
    from sklearn.neighbors import KernelDensity
    cxx = ['#E69F00', '#009E73', '#0072B2', '#D55E00', '#CC79A7']
    cxx2 = ['#8c6bb1', '#810f7c']

    data = data[~np.isnan(data)]
    x_d = np.linspace(np.min(data) * 0.9, np.max(data) * 1.1, len(data))

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(data[:, None])

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])

    if cplot & (split_ts == 1):
        plt.fill_between(x_d, np.exp(logprob), alpha=0.4, facecolor='grey')

    lp = np.exp(logprob)
    xd = x_d

    if type(cmdata) != str:
        fig, (ax, cax, cax2) = plt.subplots(ncols=3, figsize=(10.5, 2.3), gridspec_kw={"width_ratios": [1, 0.02, 0.02]})

        x_d2 = np.linspace(np.min(cmdata) * 0.9, np.max(cmdata) * 1.1, len(cmdata))
        # instantiate and fit second KDE model
        kde2 = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde2.fit(cmdata[:, None])

        # score_samples returns the log of the probability density
        logprob2 = kde2.score_samples(x_d2[:, None])

        if cplot & (split_ts == 1):
            ax.fill_between(x_d2, np.exp(logprob2), alpha=0.4, facecolor='grey')

        lp2 = np.exp(logprob2)
        xd2 = x_d2
    else:
        fig, (ax, cax) = plt.subplots(ncols=2, figsize=(10.5, 2.3), gridspec_kw={"width_ratios": [1, 0.06]})

    if split_ts > 1:
        if eq_period:
            spliti = [0, len(data) - 40, len(data) - 20, len(data)]
        else:
            spliti = np.linspace(0, len(data), split_ts + 1).astype(int)

        for i in np.arange(split_ts):
            datax = data.iloc[spliti[i]:spliti[i + 1]]
            x_d = np.linspace(np.min(datax) * 0.9, np.max(datax) * 1.1, len(datax))

            try:
                # instantiate and fit the KDE model
                kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
                kde.fit(datax[:, None])

                # score_samples returns the log of the probability density
                logprob = kde.score_samples(x_d[:, None])

                if cplot:
                    ax.fill_between(x_d, np.exp(logprob), alpha=0.4, facecolor=cxx[i],
                                    label='-'.join([str(datax.index.year.min()), str(datax.index.year.max())]))
            except:
                pass

    if type(cmdata) != str:
        # add climate model data
        csplit_ts = [2040, 2080]
        for i in np.arange(2):
            datax = cmdata.loc[(cmdata.index.year >= csplit_ts[i]) & (cmdata.index.year < csplit_ts[i] + 20)]
            x_d = np.linspace(np.min(datax) * 0.9, np.max(datax) * 1.1, len(datax))

            try:
                # instantiate and fit the KDE model
                kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
                kde.fit(datax[:, None])

                # score_samples returns the log of the probability density
                logprob = kde.score_samples(x_d[:, None])

                if cplot:
                    ax.fill_between(x_d, np.exp(logprob), alpha=0.4, facecolor=cxx2[i],
                                    label='-'.join([str(datax.index.year.min()), str(datax.index.year.max())]))

            except:
                pass

    ax.legend(loc=1, ncol=5)

    if cplot:
        # cmap = plt.cm.get_cmap('cividis_r')
        cmap = plt.cm.get_cmap('viridis_r')
        colors = plt.cm.cividis_r(np.linspace(0, 1, len(data)))
        colorsx = cmap(np.arange(cmap.N))

        for i in np.arange(len(data)):
            ax.plot([data.iloc[i], data.iloc[i]], [0, np.max(lp) * 0.9], c=colors[i])

        labcb = 'Year'
        if type(cmdata) != str:
            cmap2 = plt.cm.get_cmap('plasma')
            colors2 = plt.cm.plasma(np.linspace(0, 1, len(cmdata)))
            colorsx2 = cmap2(np.arange(cmap2.N))

            for i in np.arange(len(cmdata)):
                ax.plot([cmdata.iloc[i], cmdata.iloc[i]], [0, np.max(lp) * 0.9], c=colors2[i])

            cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap2), cax=cax2, label=labcb,
                                 ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], fraction=0.0027, anchor=(1.0, 0.1))
            cbar2.ax.set_yticklabels(
                np.round(np.linspace(cmdata.index.year.min(), cmdata.index.year.max(), 6)).astype(int).astype(str))
            labcb = ''

        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, label=labcb, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar.ax.set_yticklabels(
            np.round(np.linspace(data.index.year.min(), data.index.year.max(), 6)).astype(int).astype(
                str))  # vertically oriented colorbar

    ax.set_ylabel('Occurrence (KDE)')

    return fig, ax
