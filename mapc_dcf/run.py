import json
import logging
import os
from time import time
from typing import Dict
from argparse import ArgumentParser

os.environ['JAX_ENABLE_X64'] = 'True'

import simpy
import jax
from chex import PRNGKey
from joblib import Parallel, delayed
from tqdm import tqdm

from mapc_mab.envs.static_scenarios import *
from mapc_dcf.channel import Channel
from mapc_dcf.nodes import AccessPoint
from mapc_dcf.logger import Logger


def single_run(
        key: PRNGKey,
        run: int,
        simulation_length: float,
        warmup_length: float,
        scenario: StaticScenario,
        logger: Logger
) -> None:
    
    key, key_channel = jax.random.split(key)
    des_env = simpy.Environment()
    channel = Channel(key_channel, scenario.pos, walls=scenario.walls)
    aps: Dict[int, AccessPoint] = {}
    for ap in scenario.associations:

        key, key_ap = jax.random.split(key)
        clients = jnp.array(scenario.associations[ap])
        tx_power = scenario.tx_power[ap].item()
        mcs = scenario.mcs[ap].item()
        aps[ap] = AccessPoint(key_ap, ap, scenario.pos, tx_power, mcs, clients, channel, des_env, logger)
        aps[ap].start_operation(run)
    
    des_env.run(until=warmup_length + simulation_length)
    logger.dump_acumulators(run)

    # TODO to be removed once debugged or improve logger
    total = 0
    collisions = 0
    for ap in aps.keys():
        total_ap = aps[ap].dcf.total_attempts
        collisions_ap = aps[ap].dcf.total_collisions
        logging.warning(f"Run{run}:Collisions:AP{ap}: {collisions_ap / total_ap:.3f} (of {total_ap})")
        total += total_ap
        collisions += collisions_ap
    logging.warning(f"Run{run}:Collisions: {collisions / total:.3f} (of {total})")

    del des_env


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-c', '--config_path',    type=str, default='default_config.json')
    args.add_argument('-r', '--results_path',   type=str, default=os.path.join('out', 'results'))
    args.add_argument('-l', '--log_level',      type=str, default='warning')
    args = args.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log_level.upper()))

    with open(args.config_path, 'r') as file:
        config = json.load(file)
    
    key = jax.random.PRNGKey(config['seed'])

    logger = Logger(config['simulation_length'], config['warmup_length'], args.results_path, **config['logger_params'])
    scenario = globals()[config['scenario']](**config['scenario_params'])

    start_time = time()
    n_runs = config['n_runs']
    Parallel(n_jobs=n_runs)(
        delayed(single_run)(key, run, config['simulation_length'], config['warmup_length'], scenario, logger)
        for key, run in zip(jax.random.split(key, n_runs), range(1, n_runs + 1))
    )
    logger.shutdown(config)
    logging.warning(f"Execution time: {time() - start_time:.2f} seconds")
