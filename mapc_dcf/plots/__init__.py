from typing import Optional

from mapc_research.plots.config import *
from mapc_dcf.constants import CW_EXP_MIN, CW_EXP_MAX

set_style()

def calculate_ema(data, span=None, alpha=None):
    """
    Calculate Exponential Moving Average (EMA) for a numpy array.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array of values to smooth
    span : int, optional
        Number of periods to use for smoothing. 
        Either span or alpha must be provided.
    alpha : float, optional
        Smoothing factor between 0 and 1. 
        Either span or alpha must be provided.
        
    Returns:
    --------
    numpy.ndarray
        Array of smoothed values
        
    Notes:
    ------
    If span is provided, alpha is calculated as: 2 / (span + 1)
    The larger the span, the smoother the result.
    """
    if span is None and alpha is None:
        raise ValueError("Either span or alpha must be provided")
        
    if span is not None:
        alpha = 2 / (span + 1)
    
    # Initialize the output array with the first value
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    
    # Calculate EMA
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
    return ema


def plot_backoff_hist(backoff_hist: dict, ap: Optional[int] = None):
    """
    Plot the backoff histogram.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot of backoff times
    axes[0].scatter(backoff_hist.keys(), backoff_hist.values())
    axes[0].set_yscale('log')
    axes[0].set_ylim(9e-1, ymax=max(np.max(list(backoff_hist.values())), 3e2))
    axes[0].set_xlim(0, 1100)

    # Histogram of the backoff times
    backoffs = list(backoff_hist.keys())
    frequencies = list(backoff_hist.values())
    cw_ranges = np.array(([0] + list(2 ** np.arange(CW_EXP_MIN, CW_EXP_MAX + 1))))
    counts, bin_edges = np.histogram(backoffs, bins=cw_ranges, weights=frequencies)

    xs = range(len(cw_ranges) - 1)
    axes[1].bar(xs, counts)
    axes[1].set_xticks(xs, [f'[{cw_ranges[i]}, {cw_ranges[i+1]})' for i in xs])
    axes[1].set_yscale('log')
    axes[1].set_ylim(ymin=9e-1, ymax=max(np.max(counts), 4e3))
    axes[1].set_xlabel('Selected Backoff')
    axes[1].set_ylabel('Frequency')
    plt.title('2 APs, MCS 11')
    plt.savefig(f'backoff_ap{ap if ap is not None else ""}.pdf', bbox_inches='tight')
    plt.close()
