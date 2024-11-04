import json
import logging
import os
from time import time
from typing import Dict
from argparse import ArgumentParser

import simpy
import jax
import jax.numpy as jnp
from chex import PRNGKey
from joblib import Parallel, delayed

from mapc_research.envs.scenario import Scenario
from mapc_research.envs.dynamic_scenario import DynamicScenario
from mapc_research.envs.scenario_impl import *
# from mapc_research.envs.test_scenarios import *
from mapc_dcf.channel import Channel
from mapc_dcf.nodes import AccessPoint
from mapc_dcf.logger import Logger

logging.basicConfig(level=logging.WARNING)

# TODO DEBUG Remove this and fix imports from mapc_research
SHORT_SIM_STEPS = 50
ALL_SCENARIOS = [
    small_office_scenario(d_ap=10.0, d_sta=2.0, n_steps=SHORT_SIM_STEPS),
    DynamicScenario.from_static_scenarios(
        small_office_scenario(d_ap=20.0, d_sta=2.0, n_steps=300),
        small_office_scenario(d_ap=20.0, d_sta=3.0, n_steps=300),
        switch_steps=[SHORT_SIM_STEPS // 2], n_steps=SHORT_SIM_STEPS
    ),
    residential_scenario(seed=100, n_steps=SHORT_SIM_STEPS, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=10.0),
    random_scenario(seed=100, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=SHORT_SIM_STEPS),
]


def run_test_scenarios(key: PRNGKey):
    pass


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
    
    des_env.run(until=sim_time)
    logger.dump_acumulators(run)
    del des_env


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-r', '--results_dir',    type=str, required=True)
    args.add_argument('-s', '--seed',           type=int, default=42)
    args.add_argument('-n', '--n_runs',         type=int, default=10)
    args.add_argument('-w', '--warmup',         type=float, default=0.)
    args = args.parse_args()

    # The results directory should exist and be empty
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    else:
        assert len(os.listdir(args.results_dir)) == 0, f"Results dir {args.results_dir} is not empty." +\
            "Please empty it manually before running this script."
    

