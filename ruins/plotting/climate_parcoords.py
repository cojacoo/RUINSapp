import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import pandas as pd


DIMENSIONS = {l: dict() for l in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}


def climate_projection_parcoords(data: pd.DataFrame, fig: go.Figure = None, align_range: bool = True, colorscale = 'electric', row: int = 1, col: int = 1, lang='en'):
    """
    Parallel coordinates plot for climate projections.
    This plot uses each month in the year as a coordinate dimension. By sorting the
    dimensions into the correct order, the cycle of annual temperature aggregates is preserved,
    while the full dataset can easily be compared.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the data to plot. The DataFrame has to be indexed by a
        Datetime Index and does accept more than one column (ground station, RCP scenario or grid cell).
    fig : plotly.graph_objects.Figure
        If not None, the given figure will be used to plot the data.
        Note, that subfigures need to use the ``'domain'`` type.
    align_range : bool
        If True (default) each dimension (aka month) will use the same value range, to
        focus the differences between the months. If False, the range will be adapted
        to span from min to max for each dimension, putting more focus on the differences
        between the years (decades).
    colorscale : str
        Name identifier of teh colorscale. See plotly to learn about available options.
    row : int
        If figure is not None, row and column can be used to plot into the 
        correct subplot.
    col: int
        If figure is not None, row and column can be used to plot into the
        correct subplot.
    lang : str
        Can either be ``'en'`` or ``'de'``. As of now, the language does not
        have any effect.

    """
    # create the dimensions dictionary
    dimensions = {k:v for k, v in DIMENSIONS.items()}

    # colormap container
    cmap = []
    vmin = data.min().min()
    vmax = data.max().max()

    # group by Month
    grp = data.groupby(data.index.map(lambda x: x.strftime('%B')))

    # create dimensions
    for label, df in grp:
        df = pd.melt(df, ignore_index=False).drop('variable', axis=1)
        dim = dict(label=label, values=df.values)
        if align_range:
            dim['range'] = (vmin, vmax)
        
        # append
        dimensions[label] = dim
        cmap.extend(df.index.map(lambda x: x.year).values.tolist())
    
    if fig is None:
        fig = make_subplots(1, 1, specs=[[{'type': 'domain'}]])

    # make plot
    fig.add_trace(go.Parcoords(
        line=dict(color=cmap, colorscale=colorscale, showscale=True),
        dimensions = [dict(label='Year', values=cmap)] + list(dimensions.values())
    ), row=row, col=col)

    return fig