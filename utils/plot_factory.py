import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot as poff


def plot_historical(symbol_name: str, data: pd.DataFrame):
    """ plot historical data from data retriever """

    traces = list()
    plot_title = f"Stock: {symbol_name}"
    plot_filename = f"stock_{symbol_name}.html"

    for col in data.columns:

        traces.append(go.Scatter(
            x=data[col].index,
            y=data[col].values,
            name=col,
            showlegend=True
        ))

    fig = go.Figure(traces)

    fig.update_layout(
        title_text=plot_title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend_title="Legend Title")
    poff(fig, filename=plot_filename, auto_open=True)


def plot_multiple_histocals(data):
    raise NotImplementedError