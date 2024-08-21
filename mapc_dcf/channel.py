from typing import Set, Optional

import logging
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from chex import Array, Scalar, PRNGKey
from intervaltree import Interval, IntervalTree

from mapc_dcf.constants import *
from mapc_sim.utils import logsumexp_db, tgax_path_loss

tfd = tfp.distributions


class WiFiFrame():

    def __init__(self, src: int, dst: int, mcs: int, size: int = FRAME_LEN_INT) -> None:
        self.src = src
        self.dst = dst
        self.mcs = mcs
        self.size = size
        self.duration = self.size / (DATA_RATES[mcs].item() * 1e6)
    

    def materialize(self, start_time: float, tx_power: float):
        """
        Materialize the WiFi frame by setting its start time, end time, and transmission power.
        End time is calculated based on the predefined frame duration. After materialization,
        the frame can be sent over the channel.

        Parameters
        ----------
        start_time : float
            Transmission start time.
        tx_power : float
            Transmission power.
        """
        
        self.start_time = start_time
        self.end_time = start_time + self.duration
        self.tx_power = tx_power


class Channel():

    def __init__(self, key: PRNGKey, pos: Array, walls: Optional[Array] = None) -> None:
        self.key = key
        self.pos = pos
        self.n_nodes = self.pos.shape[0]
        self.walls = walls if walls is not None else jnp.zeros((self.n_nodes, self.n_nodes))
        self.frames_history = IntervalTree()


    def is_idle(self, time: float, ap: int) -> bool:
        """
        Check if the signal level at the AP is below the CCA threshold, in other words, check if the channel is idle.

        Parameters
        ----------
        time : float
            Time at which to check the channel.
        ap : int
            AP index for which to check the channel.

        Returns
        -------
        bool
            Whether the channel is idle or not.
        """

        self.key, key_idle = jax.random.split(self.key)
        return jax.random.uniform(key_idle).item() < 0.95 # TODO Remove this line

        self.key, key_sinr = jax.random.split(self.key)

        overlapping_frames = self.frames_history[time]

        tx_matrix_at_time = jnp.zeros((self.n_nodes, self.n_nodes))
        tx_power_at_time = jnp.zeros((self.n_nodes,))
        for frame_interval in overlapping_frames:

            overlapping_frame = frame_interval.data
            tx_matrix_at_time = tx_matrix_at_time.at[overlapping_frame.src, overlapping_frame.dst].set(1)
            tx_power_at_time = tx_power_at_time.at[overlapping_frame.src].set(overlapping_frame.tx_power)
        
        sinr_at_ap = self._get_sinr(key_sinr, tx_matrix_at_time, tx_power_at_time)[ap].item()
        return sinr_at_ap < CCA_THRESHOLD


    def is_idle_for(self, time: float, duration: float, ap: int) -> bool:
        """
        Check if the signal level at the AP is below the CCA threshold for a given duration. In other words,
        check if the channel is idle for a given duration.

        Parameters
        ----------
        time : float
            Time at which to check the channel.
        duration : float
            Required duration for the channel to be idle.
        ap : int
            AP index for which to check the channel.

        Returns
        -------
        bool
            Whether the channel is idle for the given duration or not.
        """

        overlapping_frames = self.frames_history.overlap(max(0., time - duration), time)
        logging.debug(f"AP{ap}: Overlapping frames: {overlapping_frames}")

        middlepoints = self._get_middlepoints(overlapping_frames)
        for middlepoint in middlepoints:
            if not self.is_idle(middlepoint, ap):
                return False
        return True

    
    def send_frame(self, frame: WiFiFrame, start_time: float, tx_power: float) -> None:
        """
        Send a WiFi frame over the channel.

        Parameters
        ----------
        frame : WiFiFrame
            The 802.11 frame to send.
        start_time : float
            The simulation time at which the frame transmission starts.
        tx_power : float
            The transmission power at which the frame is sent.
        """
        frame.materialize(start_time, tx_power)
        self.frames_history.add(Interval(start_time, frame.end_time, frame))


    def is_succesfully_transmitted(self, frame: WiFiFrame) -> bool:
        """
        Check if a WiFi frame was transmitted successfully. In other words, check if the frame was received without errors.

        Parameters
        ----------
        frame : WiFiFrame
            The frame to check.

        Returns
        -------
        bool
            Whether the frame was transmitted successfully or not.
        """

        self.key, key_per = jax.random.split(self.key)
        
        frame_start_time, frame_end_time = frame.start_time, frame.end_time
        overlapping_frames = self.frames_history.overlap(frame_start_time, frame_end_time)
        overlapping_frames_tree = IntervalTree(overlapping_frames)

        max_per = 0
        middlepoints = self._get_middlepoints(overlapping_frames)
        for middlepoint in middlepoints:

            self.key, key_per = jax.random.split(self.key)
            middlepoint_overlapping_frames = overlapping_frames_tree[middlepoint]
            tx_matrix_at_middlepoint = jnp.zeros((self.n_nodes, self.n_nodes))
            mcs_at_middlepoint = jnp.zeros((self.n_nodes,), dtype=int)
            tx_power_at_middlepoint = jnp.zeros((self.n_nodes,))
            for frame_interval in middlepoint_overlapping_frames:
                
                overlapping_frame = frame_interval.data
                tx_matrix_at_middlepoint = tx_matrix_at_middlepoint.at[overlapping_frame.src, overlapping_frame.dst].set(1)
                mcs_at_middlepoint = mcs_at_middlepoint.at[overlapping_frame.src].set(overlapping_frame.mcs)
                tx_power_at_middlepoint = tx_power_at_middlepoint.at[overlapping_frame.src].set(overlapping_frame.tx_power)
            
            per_middlepoint = self._get_per(
                key_per,
                tx_matrix_at_middlepoint,
                mcs_at_middlepoint,
                tx_power_at_middlepoint,
                frame.src
            )

            # TODO Can we agregate the PERs in a better way? Maybe we can weight them by the time they are overlapping?
            max_per = max(max_per, per_middlepoint)
        
        return jax.random.uniform(key_per).item() < max_per
    

    def _get_middlepoints(self, overlapping_frames: Set[Interval]) -> Array:
        start_times = {interval.data.start_time for interval in overlapping_frames}
        end_times = {interval.data.end_time for interval in overlapping_frames}
        timepoints = jnp.array(sorted(list(start_times.union(end_times))))
        return (timepoints[:-1] + timepoints[1:]) / 2

    
    def _get_sinr(self, key: PRNGKey, tx: Array, tx_power: Array) -> Array:

        distance = jnp.sqrt(jnp.sum((self.pos[:, None, :] - self.pos[None, ...]) ** 2, axis=-1))
        distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

        signal_power = tx_power[:, None] - tgax_path_loss(distance, self.walls)

        interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=1, keepdims=True) * (1 - tx)
        a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
        b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
        interference = jax.vmap(logsumexp_db, in_axes=(1, 1))(a, b)

        sinr = signal_power - interference
        sinr = sinr + tfd.Normal(loc=jnp.zeros_like(signal_power), scale=DEFAULT_SIGMA).sample(seed=key)
        sinr = (sinr * tx).sum(axis=1)

        return sinr
    

    def _get_per(self, key: PRNGKey, tx: Array, mcs: Array, tx_power: Array, ap_src: int) -> Scalar:

        sinr = self._get_sinr(key, tx, tx_power)
        sdist = tfd.Normal(loc=MEAN_SNRS[mcs], scale=2.)
        logit_success_prob = sdist.log_cdf(sinr) - sdist.log_survival_function(sinr)
        logit_success_prob = jnp.where(sinr > 0, logit_success_prob, -jnp.inf)
        success_prob = jnp.exp(logit_success_prob)/(1 + jnp.exp(logit_success_prob))

        return 1 - success_prob[ap_src].item()
        