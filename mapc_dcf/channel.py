from typing import Set, Optional, Tuple

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

    def __init__(self, id: int, src: int, dst: int, tx_power: float, mcs: int, size: int = FRAME_LEN_INT) -> None:
        self.id = id
        self.src = src
        self.dst = dst
        self.tx_power = tx_power
        self.mcs = mcs
        self.size = size
        self.duration = self.size / (DATA_RATES[mcs].item() * 1e6) # ~84 us for MCS 11
    

    def materialize(self, start_time: float, retransmission: int) -> None:
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
        self.retransmission = retransmission


class Channel():

    def __init__(self, key: PRNGKey, pos: Array, walls: Optional[Array] = None) -> None:
        self.key = key
        self.pos = pos
        self.n_nodes = self.pos.shape[0]
        self.walls = walls if walls is not None else jnp.zeros((self.n_nodes, self.n_nodes))
        self.frames_history = IntervalTree()


    def is_idle(self, time: float, ap: int, sender_tx_power: float) -> bool:
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

        # Get frames that occupy the channel at the given time
        overlapping_frames = self.frames_history[time]

        if not overlapping_frames:
            return True

        # Set the transmission matrix and transmission power from the current frames in the channel
        tx_matrix_at_time = jnp.zeros((self.n_nodes, self.n_nodes))
        tx_power_at_time = jnp.zeros((self.n_nodes,))
        for frame_interval in overlapping_frames:

            overlapping_frame = frame_interval.data
            tx_matrix_at_time = tx_matrix_at_time.at[overlapping_frame.src, overlapping_frame.dst].set(1)
            tx_power_at_time = tx_power_at_time.at[overlapping_frame.src].set(overlapping_frame.tx_power)
        
        # If ap was transmitting at the given time, the channel is busy
        if tx_matrix_at_time[ap].sum() > 0:
            return False
        
        # Set the transmission from AP to itself, to be used in the signal level calculation
        tx_matrix_at_time = tx_matrix_at_time.at[ap, ap].set(1)
        tx_power_at_time = tx_power_at_time.at[ap].set(sender_tx_power)
        
        # Channel is idle if the signal level at the AP is below the CCA threshold
        idle = self._get_signal_level(tx_matrix_at_time, tx_power_at_time, ap) < CCA_THRESHOLD

        return idle


    def is_idle_for(self, time: float, duration: float, ap: int, sender_tx_power: float) -> bool:
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

        # Transform time and duration to low and high times
        low_time, high_time = time - duration, time

        # If the simulation does not last long enough, assume the channel busy
        if low_time < 0:
            return False

        # Get the overlapping frames within the given time interval
        overlapping_frames = self.frames_history.overlap(low_time, high_time)

        # We asses the channel as idle if it is idle in all the middlepoints of the given interval
        middlepoints, _ = self._get_middlepoints_and_durations(overlapping_frames, low_time, high_time)
        logging.debug(f"AP{ap}:t{time:.9f}\t Overlapping frames: {overlapping_frames}\n" + "\t"*8 + f"Middlepoints: {middlepoints}")
        for middlepoint in middlepoints:

            if not self.is_idle(middlepoint, ap, sender_tx_power):
                return False
        
        return self.is_idle(high_time, ap, sender_tx_power)

    
    def send_frame(self, frame: WiFiFrame, start_time: float, retransmission: int) -> None:
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
        frame.materialize(start_time, retransmission)
        self.frames_history.add(Interval(start_time, frame.end_time, frame))


    def is_colliding(self, frame: WiFiFrame) -> bool:
        """
        Check if a frame transmission resulted in collision.

        Parameters
        ----------
        frame : WiFiFrame
            The transmitted frame.

        Returns
        -------
        bool
            Whether there was a collision or not.
        """
        
        # Get the overlapping frames with the transmitted frame
        frame_start_time, frame_end_time, frame_duration = frame.start_time, frame.end_time, frame.duration
        overlapping_frames = self.frames_history.overlap(frame_start_time, frame_end_time) - {Interval(frame_start_time, frame_end_time, frame)}
        overlapping_frames_tree = IntervalTree(overlapping_frames)

        # Calculate the middlepoints and durations in reference to the transmitted frame
        middlepoints, durations = self._get_middlepoints_and_durations(overlapping_frames, frame_start_time, frame_end_time)
        
        # Calculate the success probability at the middlepoints
        middlepoints_collision_probs = []
        for middlepoint, duration in zip(middlepoints, durations):

            # Get the concurrent frames at the middlepoint
            middlepoint_overlapping_frames = overlapping_frames_tree[middlepoint]
            middlepoint_overlapping_frames = middlepoint_overlapping_frames.union({Interval(frame_start_time, frame_end_time, frame)})

            # Build the transmission matrix, MCS, and transmission power at the middlepoint
            tx_matrix_at_middlepoint = jnp.zeros((self.n_nodes, self.n_nodes))
            mcs_at_middlepoint = jnp.zeros((self.n_nodes,), dtype=int)
            tx_power_at_middlepoint = jnp.zeros((self.n_nodes,))
            for frame_interval in middlepoint_overlapping_frames:
                
                iter_frame = frame_interval.data
                tx_matrix_at_middlepoint = tx_matrix_at_middlepoint.at[iter_frame.src, iter_frame.dst].set(1)
                mcs_at_middlepoint = mcs_at_middlepoint.at[iter_frame.src].set(iter_frame.mcs)
                tx_power_at_middlepoint = tx_power_at_middlepoint.at[iter_frame.src].set(iter_frame.tx_power)
            
            # Calculate the success probability at the middlepoint
            self.key, key_per = jax.random.split(self.key)
            success_prob_middlepoint = self._get_success_probability(
                key_per,
                tx_matrix_at_middlepoint,
                mcs_at_middlepoint,
                tx_power_at_middlepoint,
                frame.src
            )

            # Calculate the collision probability at the middlepoint, weighted by the overlapping ratio
            collision_prob_middlepoint = 1 - success_prob_middlepoint
            middlepoints_collision_probs.append(collision_prob_middlepoint)
        
        # Aggregate the collision probabilities
        middlepoints_collision_probs = jnp.array(middlepoints_collision_probs)
        self.key, key_collision = jax.random.split(self.key)
        collision = jnp.any(jax.random.uniform(key_collision, shape=middlepoints_collision_probs.shape) < middlepoints_collision_probs).item()
        
        return collision
    

    def _get_middlepoints_and_durations(
            self,
            overlapping_frames: Set[Interval],
            low_time: float,
            high_time: float
    ) -> Tuple[Array, Array]:
        
        start_times = {interval.data.start_time for interval in overlapping_frames if interval.data.start_time > low_time}
        start_times = start_times.union({low_time})
        end_times = {interval.data.end_time for interval in overlapping_frames if interval.data.end_time < high_time}
        end_times = end_times.union({high_time})
        timepoints = jnp.array(sorted(list(start_times.union(end_times))))
        durations = timepoints[1:] - timepoints[:-1]

        return (timepoints[:-1] + timepoints[1:]) / 2, durations
    

    def _get_signal_power_and_interference(self, tx: Array, tx_power: Array) -> Tuple[Array, Array]:

        distance = jnp.sqrt(jnp.sum((self.pos[:, None, :] - self.pos[None, ...]) ** 2, axis=-1))
        distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

        signal_power = tx_power[:, None] - tgax_path_loss(distance, self.walls)

        interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=1, keepdims=True) * (1 - tx)
        a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
        b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
        interference = jax.vmap(logsumexp_db, in_axes=(1, 1))(a, b)

        return signal_power, interference

    
    def _get_signal_level(self, tx: Array, tx_power: Array, ap: int) -> Scalar:
        _, interference = self._get_signal_power_and_interference(tx, tx_power)
        return interference[ap].item()
    

    def _get_success_probability(self, key: PRNGKey, tx: Array, mcs: Array, tx_power: Array, ap_src: int) -> Scalar:

        signal_power, interference = self._get_signal_power_and_interference(tx, tx_power)

        sinr = signal_power - interference
        sinr = sinr + tfd.Normal(loc=jnp.zeros_like(signal_power), scale=DEFAULT_SIGMA).sample(seed=key)
        sinr = (sinr * tx).sum(axis=1)

        sdist = tfd.Normal(loc=MEAN_SNRS[mcs], scale=2.)
        logit_success_prob = sdist.log_cdf(sinr) - sdist.log_survival_function(sinr)
        logit_success_prob = jnp.where(sinr > 0, logit_success_prob, -jnp.inf)
        success_prob = jnp.exp(logit_success_prob)/(1 + jnp.exp(logit_success_prob))

        return success_prob[ap_src].item()
        