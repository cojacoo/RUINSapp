import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import pandas as pd


DIMENSIONS = {l: dict() for l in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}


def climate_projection_parcoords(data: pd.DataFrame, fig: go.Figure = None, align_range: bool = True, colorscale = 'electric', row: int = 1, col: int = 1, lang='en'):
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