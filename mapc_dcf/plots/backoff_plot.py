import json
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

from mapc_research.plots.config import *
from mapc_dcf.plots import set_style, get_cmap

set_style()

LABELS = [
    "[0, 16)",
    "[16, 32)",
    "[32, 64)",
    "[64, 128)",
    "[128, 256)",
    "[256, 512)",
    "[512, 1024)"
]
BINS = [0, 16, 32, 64, 128, 256, 512, 1024]


def plot(df_results: pd.DataFrame, warmup: float, save_path: str, accumulate_aps: bool) -> None:

    # Filter out warmup period
    df = df_results[df_results["SimTime"] > warmup]

    # Plot the mean backoff
    mean_backoff = df["Backoff"].mean()

    # Set color map
    aps = sorted(df_results["Src"].unique())
    colors = get_cmap(len(aps))

    plt.figure(figsize=(5, 3))
    if accumulate_aps:
        # Accumulate all APs results
        backoffs = df["Backoff"]
        hist, _ = np.histogram(backoffs, bins=BINS)
        plt.bar(LABELS, hist, color=colors[0])
    else:
        # Plot each AP separately
        for ap, color in zip(aps, colors):
            backoffs = df[df["Src"] == ap]["Backoff"]
            hist, _ = np.histogram(backoffs, bins=BINS)
            plt.bar(LABELS, hist, color=color, alpha=0.5, label=f"AP {ap}")

    # Configure plot
    plt.title(f'Mean backoff: {mean_backoff:.4f}')
    plt.xlabel('Backoff Interval')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.yscale('log')
    plt.ylim(bottom=0.5)
    plt.grid(axis='y')
    plt.legend() if not accumulate_aps else None
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-c', '--csv',    type=str,   help='Path to the results CSV file', required=True)
    args.add_argument('-j', '--json',   type=float, help='Path to the results JSON file')
    args.add_argument('-a', '--accumulate_aps',     help='Flag to accumulate APs results in a single histogram'\
                      , action='store_true')
    args = args.parse_args()

    if args.json is None:
        args.json = args.csv.split('.')[0] + '.json'

    save_path = args.csv.split('.')[0] + '_backoffs.pdf'

    # Load warmup period from the JSON file
    with open(args.json, 'r') as f:
        json_data = json.load(f)
    warmup = json_data["Config"]["warmup_length"]

    # Load the results CSV file
    df_results = pd.read_csv(args.csv)

    # Plot the backoff distribution
    plot(df_results, warmup, save_path, args.accumulate_aps)
    