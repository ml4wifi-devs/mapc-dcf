import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
import logging
from argparse import ArgumentParser

import jax
import simpy
from tqdm import tqdm

from mapc_mab.envs.static_scenarios import *
from mapc_dcf.channel import Channel
from mapc_dcf.nodes import AccessPoint

def run_scenario(
        scenario: StaticScenario,
        n_reps: int,
        time: float,
        seed: int
):
    key = jax.random.PRNGKey(seed)
    channel = Channel()
    aps = list(scenario.associations.keys())
    for ap in aps:
        clients = jnp.array(scenario.associations[ap])
        ap = AccessPoint(ap, channel, scenario.pos, scenario.mcs[ap].item(), clients, key)
        ap.start_operation()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-c', '--config', type=str, default='config.json')
    args.add_argument('-o', '--output', type=str, default='all_results.json')
    args.add_argument('-l', '--log', type=str, default='warning')
    args = args.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log.upper()))

    with open(args.config, 'r') as file:
        config = json.load(file)

    for scenario_config in tqdm(config['scenarios'], desc='Scenarios'):
        scenario = globals()[scenario_config['scenario']](**scenario_config['params'])
        logging.info(f"Running: {scenario_config['name']}")

        run_scenario(scenario, config['n_reps'], scenario_config['time'], config['seed'])