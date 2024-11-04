from argparse import ArgumentParser

import json
import pandas as pd
import matplotlib.pyplot as plt

from mapc_dcf.constants import DATA_RATES
from mapc_research.plots.utils import confidence_interval


def plot(
    df_results: pd.DataFrame,
    scenario_name: str,
    mcs: int,
    num_bins: int = None
) -> None:

    # Aggregate the data rate from all streams
    df = df_results.groupby(['Run', 'Time'])['DataRate'].sum().reset_index()

    # Bin the data to avoide temporal fluctuations
    num_bins = num_bins if num_bins else df.groupby('Run')['Run'].count().min()
    df['TimeBin'] = pd.cut(df['Time'], bins=int(num_bins), labels=False)

    # Average the data rate in each bin
    df = df.groupby(['Run', 'TimeBin']).mean().reset_index()

    # Replace the time with the normalized time (mean time in the bin)
    normalized_time = df.groupby(['TimeBin'])['Time'].mean().reset_index()
    df = df.drop(columns=['Time'])
    df = df.merge(normalized_time, on='TimeBin').sort_values(by=['Run', 'Time']).reset_index(drop=True)
    df = df.drop(columns=['TimeBin'])

    # Calculate the confidence intervals
    df_data_rate = df.pivot(index='Run', columns='Time', values='DataRate').reset_index().drop(columns=['Run'])
    time = df_data_rate.columns.values.astype(float)
    time = time - time[0]
    mean, ci_low, ci_high = confidence_interval(df_data_rate.values)

    # Plot the coordinated single TX data rate
    plt.axhline(DATA_RATES[mcs].item(), linestyle='--', color='gray', label='Single TX')

    # Plot the DCF data rate
    plt.plot(time, mean, label='DCF')
    plt.fill_between(time, ci_low, ci_high, alpha=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Effective data rate [Mb/s]')
    plt.xlim((time[0], time[-1]))
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rate-{scenario_name}.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-d', '--csv_data',       type=str, required=True)
    args.add_argument('-c', '--config_path',    type=str, required=True)
    args.add_argument('-b', '--num_bins',       type=int, default=None)
    args = args.parse_args()

    df_results = pd.read_csv(args.csv_data)

    with open(args.config_path, 'r') as file:
        config = json.load(file)
    
    scenario_name = config['scenario']
    mcs = config['scenario_params']['mcs']

    plot(df_results, scenario_name, mcs, args.num_bins)


    