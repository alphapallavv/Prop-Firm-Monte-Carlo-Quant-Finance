# utils.py
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_equity_cloud(equity_matrix, sample_n=200, title="Equity Simulation"):
    """
    equity_matrix: np.array shape (n_sims, days)
    sample_n: number of sample lines to plot for visual clarity
    """
    n_sims, days = equity_matrix.shape
    # pick a random subset to plot (to avoid massive traces)
    rng = np.random.default_rng(123)
    idx = rng.choice(n_sims, size=min(sample_n, n_sims), replace=False)
    x = np.arange(1, days+1)
    fig = go.Figure()
    for i in idx:
        fig.add_trace(go.Scatter(x=x, y=equity_matrix[i], mode='lines',
                                 line=dict(width=1, color='rgba(200,200,200,0.12)'),
                                 showlegend=False, hoverinfo='none'))
    # Plot median and some quantiles
    median = np.median(equity_matrix, axis=0)
    q10 = np.percentile(equity_matrix, 10, axis=0)
    q90 = np.percentile(equity_matrix, 90, axis=0)
    fig.add_trace(go.Scatter(x=x, y=median, mode='lines', name='Median',
                             line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=x, y=q90, mode='lines', name='90%', line=dict(color='rgba(255,200,100,0.6)', width=0)))
    fig.add_trace(go.Scatter(x=x, y=q10, mode='lines', name='10%', line=dict(color='rgba(255,200,100,0.6)', width=0),
                             fill='tonexty', fillcolor='rgba(255,200,100,0.12)'))
    fig.update_layout(title=title, template='plotly_dark', xaxis_title='Day', yaxis_title='Equity ($)', height=420)
    return fig

def plot_histogram(final_equity, start_equity, bins=50):
    fig = go.Figure()
    pnl = final_equity - start_equity
    fig.add_trace(go.Histogram(x=pnl, nbinsx=bins, marker_color='lightseagreen'))
    fig.update_layout(template='plotly_dark', title='Final P&L Distribution', xaxis_title='Profit / Loss ($)', yaxis_title='Count', height=300)
    return fig

def compute_metrics(results, start_equity):
    n = results['equity_matrix'].shape[0]
    passes = results['pass_mask'].sum()
    pass_rate = passes / n * 100.0
    fail_reasons = results['fail_reason']
    unique, counts = np.unique(fail_reasons, return_counts=True)
    freq = dict(zip(unique, counts))
    metrics = {
        'n_sims': n,
        'pass_rate': pass_rate,
        'passes': int(passes),
        'fail_counts': freq
    }
    return metrics
