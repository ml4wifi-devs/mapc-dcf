import json
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

from mapc_research.plots.config import *
from mapc_dcf.plots import set_style, get_cmap

set_style()

BINS = [16, 32, 64, 128, 256, 512, 1024, 1024]
LABELS = [str(b) for b in BINS]


def plot(df_results: pd.DataFrame, warmup: float, save_path: str, accumulate_aps: bool) -> None:

    # Filter out warmup period
    df = df_results[df_results["SimTime"] > warmup]

    # Set color map
    aps = sorted(df_results["Src"].unique())
    colors = get_cmap(len(aps))

    plt.figure(figsize=(5, 3))
    if accumulate_aps:
        # Accumulate all APs results
        cws = df["CW"]
        hist, _ = np.histogram(cws, bins=BINS)
        print(hist)
        plt.bar(LABELS[:len(hist)], hist, color=colors[0])
    else:
        # Plot each AP separately
        for ap, color in zip(aps, colors):
            cws = df[df["Src"] == ap]["CW"]
            hist, _ = np.histogram(cws, bins=BINS)
            plt.bar(LABELS[:len(hist)], hist, color=color, alpha=0.5, label=f"AP {ap}")

    # Configure plot
    plt.xlabel('CW Value')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.yscale('log')
    plt.ylim(bottom=0.5)
    plt.grid(axis='y')
    if not accumulate_aps and len(aps) <= 10:
        plt.legend()
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

    save_path = args.csv.split('.')[0] + '_CWs.pdf'

    # Load warmup period from the JSON file
    with open(args.json, 'r') as f:
        json_data = json.load(f)
    warmup = json_data["Config"]["warmup_length"]

    # Load the results CSV file
    df_results = pd.read_csv(args.csv)

    # Plot the CW distribution
    plot(df_results, warmup, save_path, args.accumulate_aps)
    