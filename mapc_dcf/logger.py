from typing import Optional

import os
import logging
import pandas as pd


class Logger:

    def __init__(
        self,
        results_path: str,
        n_runs: int,
        logging_freq: float,
        log_collisions: bool
    ) -> None:
        self.n_runs = n_runs
        self.logging_freq = logging_freq
        self.column_names = ['SimTime', 'Src', 'Dst', 'Payload', 'MCS', 'Collision']
        self.logs_per_run = {run: [0., []] for run in range(1, self.n_runs + 1)}
        self.results_path = results_path + '.csv' if not results_path.endswith('.csv') else results_path
        self.log_collisions = log_collisions

        if os.path.exists(self.results_path):
            logging.warning(f"logger: Overwriting file {self.results_path}!")
            os.remove(self.results_path)


    def save_results(self, run: int, sim_time: float):

        logging.info(f"logger: Saving results for run {run} at time {sim_time}")
        
        # Load the logs
        logs = self.logs_per_run[run][1]
        logs_df = pd.DataFrame(logs, columns=self.column_names)

        # Calculate the results
        data_volume = logs_df[logs_df["Collision"] == False].groupby('Dst')['Payload'].sum() / 1e6
        success_rate = 1 - logs_df.groupby('Dst')['Collision'].mean()

        # Build the results dataframe
        results_df = pd.DataFrame({'DataVolume': data_volume, 'SuccessRate': success_rate})
        results_df.reset_index(inplace=True)
        results_df['Run'] = run
        results_df['Time'] = sim_time
        results_df['DataRate'] = results_df['DataVolume'] / sim_time
        
        # Reorder the columns
        results_df = results_df[['Run', 'Time', 'Dst', 'DataVolume', 'DataRate', 'SuccessRate']]

        # Save the results
        results_df.to_csv(self.results_path, mode='a', header=not os.path.exists(self.results_path), index=False)

        # Reset the logs
        self.logs_per_run[run] = [sim_time, []]


    def log(self, run: int, sim_time: float, src: int, dst: int, payload: int, mcs: int, collision: bool):

        if self.log_collisions or not collision:

            # Every logging_freq seconds, aggregate, save and reset the logs structure
            if sim_time - self.logs_per_run[run][0] >= self.logging_freq:
                self.save_results(run, sim_time)
                self.logs_per_run[run][0] = sim_time
            
            # Log the event
            self.logs_per_run[run][1].append([sim_time, src, dst, payload, mcs, collision])
