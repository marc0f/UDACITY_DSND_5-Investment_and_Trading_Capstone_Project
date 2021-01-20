import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot as poff


def plot_historical(symbol_name: str, data: pd.DataFrame, open_file=False, return_fig=False):
    """ plot historical data from data retriever """

    traces = list()
    plot_title = f"Stock: {symbol_name}"
    plot_filename = f"stock_{symbol_name}.html"

    for col in data.columns:

        traces.append(go.Scatter(
            x=data[col].index,
            y=data[col].values,
            name=col,
            showlegend=True)
        )

    fig = go.Figure(traces)

    fig.update_layout(
        title_text=plot_title,
        xaxis_title="Date",
        yaxis_title="Price ($)")
        # legend_title="Legend Title")

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    if return_fig:
        return fig

    poff(fig, filename=plot_filename, auto_open=open_file)


def plot_historical_with_predicted_data(symbol_name: str, data: pd.DataFrame, predicted_data: pd.DataFrame,
                                        open_file=False, return_fig=False):
    """ plot historical data from data retriever """

    traces = list()
    plot_title = f"Stock: {symbol_name}"
    plot_filename = f"stock_{symbol_name}.html"

    for col in data.columns:

        traces.append(go.Scatter(
            x=data[col].index,
            y=data[col].values,
            name=col,
            showlegend=True)
        )

    for col in predicted_data.columns:

        traces.append(go.Scatter(
            x=predicted_data[col].dropna().index,
            y=predicted_data[col].dropna().values,
            name=col,
            line=dict(color='red', dash='dot'),
            showlegend=True)
        )

    fig = go.Figure(traces)

    fig.update_layout(
        title_text=plot_title,
        xaxis_title="Date",
        yaxis_title="Price ($)")
        # legend_title="Legend Title")

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    if return_fig:
        return fig

    poff(fig, filename=plot_filename, auto_open=open_file)


def plot_multiple_historical(data):
    raise NotImplementedError