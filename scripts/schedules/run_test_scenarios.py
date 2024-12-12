from typing import List, Tuple
import logging
import os
os.environ['JAX_ENABLE_X64'] = 'True'
from time import time
from datetime import datetime
from typing import Dict
from argparse import ArgumentParser

import simpy
import jax
import jax.numpy as jnp
from chex import PRNGKey
from joblib import Parallel, delayed

from mapc_research.envs.scenario import Scenario
from mapc_research.envs.test_scenarios import *
from mapc_dcf.channel import Channel
from mapc_dcf.nodes import AccessPoint
from mapc_dcf.logger import Logger


logging.basicConfig(level=logging.WARNING)
LOGGER_DUMP_SIZE = 100


def flatten_scenarios(scenarios: List[Scenario]) -> List[Tuple[Scenario, float, str]]:

    scenarios_flattened = []
    for scenario in scenarios:
        str_repr = scenario.__str__()
        list_of_scenarios = scenario.split_scenario()
        for i, s in enumerate(list_of_scenarios):
            suffix = f"_{chr(ord('a') + i)}" if len(list_of_scenarios) > 1 else ""
            scenarios_flattened.append((s[0], s[1], f"{str_repr}{suffix}"))
    return scenarios_flattened


def single_run(key: PRNGKey, run: int, scenario: Scenario, sim_time: float, logger: Logger):
    key, key_channel = jax.random.split(key)
    des_env = simpy.Environment()
    channel = Channel(key_channel, scenario.pos, walls=scenario.walls)
    aps: Dict[int, AccessPoint] = {}
    for ap in scenario.associations:

        key, key_ap = jax.random.split(key)
        clients = jnp.array(scenario.associations[ap])
        tx_power = scenario.tx_power[ap].item()
        mcs = scenario.mcs
        aps[ap] = AccessPoint(key_ap, ap, scenario.pos, tx_power, mcs, clients, channel, des_env, logger)
        aps[ap].start_operation(run)
    
    des_env.run(until=(logger.warmup_length + sim_time))
    logger.dump_acumulators(run)
    del des_env


def run_test_scenarios(key: PRNGKey, results_dir: str, n_runs: int, warmup: float, scenarios_type: str):

    # TODO TO DELETE: Temporary subscenario selection
    if scenarios_type == '1st':
        scenarios = RANDOM_SCENARIOS[:8]
    elif scenarios_type == '2nd':
        scenarios = RANDOM_SCENARIOS[8:16]
    elif scenarios_type == '3rd':
        scenarios = RANDOM_SCENARIOS[16:]
    else:
        scenarios = ALL_SCENARIOS
    
    scenarios = flatten_scenarios(scenarios)
    n_scenarios = len(scenarios)
    for i, (scenario, sim_time, scenario_name) in enumerate(scenarios, 1):
        logger = Logger(sim_time, warmup, os.path.join(results_dir, scenario_name), dump_size=LOGGER_DUMP_SIZE)
        
        logging.warning(f"{datetime.now()}\t Running scenario {scenario_name} ({i}/{n_scenarios})")
        start_time = time()
        Parallel(n_jobs=n_runs)(
            delayed(single_run)(k, r, scenario, sim_time, logger)
            for k, r in zip(jax.random.split(key, n_runs), range(1, n_runs + 1))
        )
        logger.shutdown({
            "name": scenario_name,
            "simulation_length": sim_time,
            "warmup_length": warmup,
            "n_runs": n_runs

        })
        logging.warning(f"Execution time: {time() - start_time:.2f} seconds")


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-r', '--results_dir',    type=str, required=True)
    args.add_argument('-s', '--seed',           type=int, default=42)
    args.add_argument('-n', '--n_runs',         type=int, default=10)
    args.add_argument('-w', '--warmup',         type=float, default=0.)

    # TODO TO DELETE: Temporary subscenario selection
    args.add_argument('-t', '--scenarios_type', type=str, default='all', choices=['all', '1st', '2nd', '3rd'])
    
    args = args.parse_args()
    
    key = jax.random.PRNGKey(args.seed)
    run_test_scenarios(key, args.results_dir, args.n_runs, args.warmup, args.scenarios_type)
    

