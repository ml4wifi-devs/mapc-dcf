from typing import Optional

from mapc_mab.plots.config import *
from mapc_dcf.constants import CW_EXP_MIN, CW_EXP_MAX

set_style()


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
