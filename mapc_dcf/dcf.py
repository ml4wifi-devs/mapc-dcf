from typing import Callable, Optional
from chex import PRNGKey

import logging
import jax
import jax.numpy as jnp
import simpy

from mapc_dcf.constants import *
from mapc_dcf.channel import Channel, WiFiFrame
from mapc_dcf.logger import Logger


class DCF():

    def __init__(
            self,
            key: PRNGKey,
            ap: int,
            des_env: simpy.Environment,
            channel: Channel,
            logger: Logger,
            frame_generator: Callable[[], WiFiFrame],
    ) -> None:
        self.key = key
        self.ap = ap
        self.des_env = des_env
        self.channel = channel
        self.logger = logger
        self.frame_generator = frame_generator
        self.retry_limit = RETRY_LIMIT if RETRY_LIMIT is not None else jnp.inf
        self.cw = 2**CW_EXP_MIN

        # TODO Temporary, to be removed
        self.total_attempts = 0
        self.total_collisions = 0
    

    def start_operation(self, run_number: int):
        self.run_number = run_number
        self.des_env.process(self._run())
    

    def _wait_for_one_slot(self):
        yield self.des_env.timeout(SLOT_TIME)
    

    def _wait_for_difs(self, frame: WiFiFrame):
        
        idle = False
        while not idle:
            yield self.des_env.timeout(SLOT_TIME)
            idle = self.channel.is_idle_for(self.des_env.now, DIFS, frame.src, frame.tx_power)


    def _try_sending(self, frame: WiFiFrame, retry_count: int):

        logging.info(f"AP{self.ap}:t{self.des_env.now}\t Attempting to send frame. Retry count: {retry_count}")
        
        # If the retry limit is reached, the frame is dropped
        if retry_count > self.retry_limit:
            logging.info(f"AP{self.ap}:t{self.des_env.now}\t Retry limit reached, dropping frame")
            return

        # Initialize a random backoff interval
        key_backoff, self.key = jax.random.split(self.key)
        time_to_backoff = jax.random.randint(key_backoff, shape=(1,), minval=0, maxval=self.cw).item()
        self.logger.log_backoff(self.des_env.now, time_to_backoff, self.ap)
        logging.info(f"AP{self.ap}:t{self.des_env.now}\t TTB initialized as {time_to_backoff} from [0, {self.cw}) interval")

        # The bakoff countdown with the freeze-and-reactivation mechanism
        while time_to_backoff > 0:
            logging.info(f"AP{self.ap}:t{self.des_env.now}\t TTB = {time_to_backoff}")

            # The backoff time counter is decremented as long as the channel is sensed idle.
            if self.channel.is_idle(self.des_env.now, frame.src, frame.tx_power):
                yield self.des_env.process(self._wait_for_one_slot())
                time_to_backoff -= 1
            
            # It is frozen when activities (i.e. packet transmissions) are detected on the channel
            else:
                logging.info(f"AP{self.ap}:t{self.des_env.now}\t Channel busy, freezing backoff at TTB = {time_to_backoff}")
                yield self.des_env.process(self._wait_for_difs(frame))
                logging.info(f"AP{self.ap}:t{self.des_env.now}\t Channel idle, reactivating backoff at TTB = {time_to_backoff}")
                # and reactivated after the channel is sensed idle again for a guard period.
        
        # The frame is sent to the channel
        logging.info(f"AP{self.ap}:t{self.des_env.now}\t Sending frame to {frame.dst}")
        self.channel.send_frame(frame, self.des_env.now)
        yield self.des_env.timeout(frame.duration + SIFS) # The SIFS is the lower bound of the ACK timeout

        # If the packet transmission is unsuccessful, the size of the contention window is doubled
        if collision := self.channel.is_colliding(frame):
            self.cw = min(2*self.cw, 2**CW_EXP_MAX)
            yield self.des_env.process(self._try_sending(frame, retry_count + 1))
            self.total_collisions += 1
            logging.info(f"AP{self.ap}:t{self.des_env.now}\t Collision, increasing CW to {self.cw}")
        
        # and reset, if successful
        else:
            self.cw = 2**CW_EXP_MIN
            logging.info(f"AP{self.ap}:t{self.des_env.now}\t TX successfull, resetting CW to {self.cw}")

        # Log the transmission attempt
        self.total_attempts += 1
        self.logger.log(self.run_number, self.des_env.now, frame.src, frame.dst, frame.size, frame.mcs, collision)


    def _run(self):
        """
        The simplified 802.11 DCF algorithm. Diagram of the algorithm can be found in
        the documentation `\\docs\\diagrams\\DCF_simple.pdf`.
        """

        logging.info(f"AP{self.ap}:t{self.des_env.now}\t DCF running")

        # Network is assumed to be saturated, there is always a frame to send
        while True:

            # Generate a frame
            frame = self.frame_generator()

            # Whenever a station has a packet to send it should defer its transmission for a DIFS guard period
            yield self.des_env.process(self._wait_for_difs(frame))
            yield self.des_env.process(self._try_sending(frame, 0))
