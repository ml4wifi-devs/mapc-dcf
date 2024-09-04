from typing import Optional

import os
import json
import logging
import pandas as pd
import numpy as np

from mapc_mab.plots.utils import confidence_interval


class Logger:

    def __init__(
        self,
        results_path: str,
        n_runs: int,
        warmup_length: float,
        logging_freq: float,
        log_collisions: bool
    ) -> None:
        self.n_runs = n_runs
        self.warmup_length = warmup_length
        self.logging_freq = logging_freq
        self.column_names = ['SimTime', 'Src', 'Dst', 'Payload', 'MCS', 'Collision']
        self.logs_per_run = {run: [0., []] for run in range(1, self.n_runs + 1)}
        self.acumulators = {}
        self.results_path_csv = results_path.split('.')[0] + '.csv'
        self.results_path_json = results_path.split('.')[0] + '.json'
        self.log_collisions = log_collisions

        # Create the results files
        if os.path.exists(self.results_path_csv):
            logging.warning(f"logger: Overwriting file {self.results_path_csv}!")
            os.remove(self.results_path_csv)
        if os.path.exists(self.results_path_json):
            logging.warning(f"logger: Overwriting file {self.results_path_json}!")
            os.remove(self.results_path_json)


    def _savepoint(self, run: int, sim_time: float, time_delta: float) -> None:

        logging.info(f"logger: Saving results for run {run} at time {sim_time}")
        
        # Load the logs
        logs = self.logs_per_run[run][1]
        logs_df = pd.DataFrame(logs, columns=self.column_names)

        # Calculate the results
        src = logs_df.groupby('Dst')['Src'].first()
        data_volume = logs_df[logs_df["Collision"] == False].groupby('Dst')['Payload'].sum() / 1e6
        success_rate = 1 - logs_df.groupby('Dst')['Collision'].mean()

        # Build the results dataframe
        results_df = pd.DataFrame({'Src': src, 'DataVolume': data_volume, 'SuccessRate': success_rate})
        results_df.reset_index(inplace=True)
        results_df['Run'] = run
        results_df['Time'] = sim_time
        results_df['DataRate'] = results_df['DataVolume'] / time_delta
        
        # Reorder the columns
        results_df = results_df[['Run', 'Time', 'Src', 'Dst', 'DataVolume', 'DataRate', 'SuccessRate']]

        # Save the results
        results_df.to_csv(self.results_path_csv, mode='a', header=not os.path.exists(self.results_path_csv), index=False)

        # Reset the logs
        self.logs_per_run[run] = [sim_time, []]


    def log(self, run: int, sim_time: float, src: int, dst: int, payload: int, mcs: int, collision: bool) -> None:

        if sim_time <= self.warmup_length:
            return
        
        # Log accumulators
        if run not in self.acumulators:
            self.acumulators[run] = {"DataVolume": 0, "TotalFrames": 0, "SuccessFrames": 0}
        self.acumulators[run]["DataVolume"] += payload / 1e6
        self.acumulators[run]["TotalFrames"] += 1
        self.acumulators[run]["SuccessFrames"] += 1 - collision

        # If log_collisions flag is True, log all transmissions. Otherwise, log only successful transmissions
        if self.log_collisions or not collision:

            # Every logging_freq seconds, aggregate, save and reset the logs structure
            time_delta = sim_time - self.logs_per_run[run][0]
            if time_delta >= self.logging_freq:
                self._savepoint(run, sim_time, time_delta)
                self.logs_per_run[run][0] = sim_time
            
            # Log the event
            self.logs_per_run[run][1].append([sim_time, src, dst, payload, mcs, collision])
    

    def save_accumulators(self, simulation_length: float) -> None:

        # Aggregate the dictionaries
        data_rate = [self.acumulators[run]["DataVolume"] / simulation_length for run in self.acumulators]
        success_rate = [self.acumulators[run]["SuccessFrames"] / self.acumulators[run]["TotalFrames"] for run in self.acumulators]

        # Calculate the confidence intervals
        data_rate_mean, data_rate_low, data_rate_high = confidence_interval(np.array(data_rate))
        success_rate_mean, success_rate_low, success_rate_high = confidence_interval(np.array(success_rate))

        # Save the results
        results = {
            "DataRate": {"Mean": data_rate_mean, "Low": data_rate_low, "High": data_rate_high},
            "SuccessRate": {"Mean": success_rate_mean, "Low": success_rate_low, "High": success_rate_high}
        }
        with open(self.results_path_json, 'w') as file:
            json.dump(results, file, indent=4)
