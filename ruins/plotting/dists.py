"""
Distribution plots as used by the uncertainty application.
"""
from typing import List
from itertools import cycle

import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.express.colors import unlabel_rgb, label_rgb, n_colors, sequential


def distribution_plot(*events: List[dict], fig: go.Figure = None, **kwargs) -> go.Figure:
    """Concept plot for PDFs of different events.
    This plot illustrates the interdependece of Kngihtian uncertainty
    and risk for a set of events. At least one event has to be passed.
    
    Parameters
    ----------
    events : List
        Each event is represented by a dict, that has to contain the following keys:
        outcomes: List[Tuple[float, float]]
            The mu and std of each outcome for this event
        Additionally, the following keys are optional:
        dist : str
            The distribution of the event. Default is 'norm'.
        coloscale : str
            The colorscale used to color the events. Default is a 
            cycle through ['Greens', 'Reds', 'Blues']
        name : str
            Name for this event
    
    Returns
    -------
    fig : go.Figure
        Result plot
    """
    # create a figure
    if fig is None:
        fig = go.Figure()

    # get the x-axis
    x = np.linspace(0, 10, 200)

    # get the first event
    if len(events) == 0:
        raise ValueError('At least one event has specified')

    # default colorscale names
    COLORS = cycle(['Greens', 'Reds', 'Blues', 'Oranges', 'Greys'])

    for event in events:
        # get the distribution
        dist = getattr(stats, event.get('dist', 'norm'))

        # get the coloscale
        cscale = getattr(sequential, event.get('colorscale', next(COLORS)))
        cmap = n_colors(unlabel_rgb(cscale[-1]), unlabel_rgb(cscale[0]), len(event['outcomes']) + 2)

        for i, outcome in enumerate(event['outcomes']):
            mu, std = outcome
            y = dist.pdf(x, loc=mu, scale=std)
            y_sum = y.sum()
            y /= y_sum

            # add the traces
            fig.add_trace(
                go.Scatter(x=x, y=y * 100, 
                mode='lines', 
                line=dict(color=label_rgb(cmap[i])), 
                name=event.get('name', f'Outcome #{i + 1}'), 
                fill='tozerox'
                )
            )
            fig.add_trace(
                go.Scatter(x=[mu, mu], y=[0, dist.pdf(mu, loc=mu, scale=std) / y_sum * 100], 
                mode='lines',
                line=dict(color=label_rgb(cmap[i]), width=3, dash='dash'),
                name=f"Mean {event.get('name', f'Outcome #{i + 1}')}",
                )
            )
    # adjust figure
    fig.update_layout(
        template='plotly_white',
        legend=dict(orientation='h')
    )

    # return
    return fig
