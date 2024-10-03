import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
import logging
from argparse import ArgumentParser

import jax
from chex import PRNGKey
import simpy
from tqdm import tqdm
from typing import Dict

from mapc_mab.envs.static_scenarios import *
from mapc_dcf.channel import Channel
from mapc_dcf.nodes import AccessPoint
from mapc_dcf.logger import Logger

def run_scenario(
        key: PRNGKey,
        n_runs: int,
        simulation_length: float,
        warmup_length: float,
        scenario: StaticScenario,
        logger: Logger
):  
    
    run_keys = jax.random.split(key, n_runs)

    for run, _ in enumerate(tqdm(range(n_runs), desc='Repetition'), start=1):
        logging.info(f"Run {run}/{n_runs}")

        run_key = run_keys[run - 1]
        run_key, key_channel = jax.random.split(run_key)
        des_env = simpy.Environment()
        channel = Channel(key_channel, scenario.pos, walls=scenario.walls)
        aps: Dict[int, AccessPoint] = {}
        for ap in scenario.associations:

            run_key, key_ap = jax.random.split(run_key)
            clients = jnp.array(scenario.associations[ap])
            tx_power = scenario.tx_power[ap].item()
            mcs = scenario.mcs[ap].item()
            aps[ap] = AccessPoint(key_ap, ap, scenario.pos, tx_power, mcs, clients, channel, des_env, logger)
            aps[ap].start_operation(run)
        
        des_env.run(until=warmup_length + simulation_length)
        logger._save_accumulators()

        # TODO to be removed once debugged or improve logger
        total = 0
        collisions = 0
        for ap in aps.keys():
            total_ap = aps[ap].dcf.total_frames
            collisions_ap = aps[ap].dcf.total_collisions
            print(f"Collisions:AP{ap}: {collisions_ap / total_ap:.3f} (of {total_ap})")
            total += total_ap
            collisions += collisions_ap
        print(f"Collisions: {collisions / total:.3f} (of {total})")

        del des_env

    logger.shutdown()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-c', '--config_path',    type=str, default='default_config.json')
    args.add_argument('-r', '--results_path',   type=str, default='all_results')
    args.add_argument('-l', '--log_level',      type=str, default='warning')
    args = args.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log_level.upper()))

    with open(args.config_path, 'r') as file:
        config = json.load(file)
    
    key = jax.random.PRNGKey(config['seed'])

    logger = Logger(args.results_path, config['n_runs'], config['simulation_length'], config['warmup_length'], **config['logger_params'])
    scenario = globals()[config['scenario']](**config['scenario_params'])
    run_scenario(key, config['n_runs'], config['simulation_length'], config['warmup_length'], scenario, logger)
