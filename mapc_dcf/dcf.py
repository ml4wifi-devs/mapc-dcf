from typing import Callable
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
        self.cw = 2**CW_EXP_MIN

        # TODO Temporary, to be removed
        self.total_frames = 0
        self.total_collisions = 0
    
    def start_operation(self, run_number: int):
        self.run_number = run_number
        self.des_env.process(self.run())
    
    def wait_for_one_slot(self):
        yield self.des_env.timeout(SLOT_TIME)

    def run(self):
        """
        The simplified 802.11 DCF algorithm. Diagram of the algorithm can be found in
        the documentation `\\docs\\diagrams\\DCF_simple.pdf`.
        """

        logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t DCF running")

        # While ready to send frames
        while True:
            frame = self.frame_generator()

            # Try sending the frame until transmitted successfully
            retry_count = 0
            frame_sent_successfully = False
            while not frame_sent_successfully:

                logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t Attempting to send frame. Retry count: {retry_count}")
                
                # First condition: channel is idle
                channel_idle = False
                while not channel_idle:

                    # Wait for DIFS
                    while not channel_idle:
                        yield self.des_env.timeout(SLOT_TIME)
                        channel_idle = self.channel.is_idle_for(self.des_env.now, DIFS, frame.src, frame.tx_power)
                    logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t Channel idle for a DIFS guard period")
                    
                    # Initialize backoff counter
                    key_backoff, self.key = jax.random.split(self.key)
                    time_to_backoff = jax.random.randint(key_backoff, shape=(1,), minval=0, maxval=self.cw).item()
                    self.logger.log_backoff(self.des_env.now, time_to_backoff, self.ap)
                    logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t TTB initialized as {time_to_backoff} from [0, {self.cw}) interval")

                    # Corner case: Selected TTB is zero (in this situatiion, we ommit the next while loop)
                    if time_to_backoff == 0:
                        channel_idle = self.channel.is_idle(self.des_env.now, frame.src, frame.tx_power)

                    # Second condition: backoff counter is zero
                    while channel_idle and time_to_backoff > 0:
                        logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t TTB = {time_to_backoff}")

                        # If not, wait for one slot and check again both conditions
                        yield self.des_env.process(self.wait_for_one_slot())
                        time_to_backoff -= 1
                        channel_idle = self.channel.is_idle(self.des_env.now, frame.src, frame.tx_power)
                    
                    # Log the channel status
                    if not channel_idle:
                        logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t Channel busy, waiting for idle channel")
                
                logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t TTB reached zero (TTB = {time_to_backoff}) and the channel is idle")
                logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t Sending frame to {frame.dst}")
                
                # If both conditions are met, send the frame
                yield self.des_env.timeout(SIFS)            # TODO Try to remove this line
                self.channel.send_frame(frame, self.des_env.now)
                yield self.des_env.timeout(frame.duration)  # TODO Include ACK time
                collision = self.channel.is_colliding(frame)
                yield self.des_env.timeout(SIFS)            # TODO Try to remove this line

                # Act according to the transmission result
                self.total_frames += 1
                if collision:
                    self.total_collisions += 1
                    retry_count += 1
                    self.cw = min(2*self.cw, 2**CW_EXP_MAX)
                    logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t Collision, increasing CW to {self.cw}")
                else:
                    frame_sent_successfully = True
                    self.cw = 2**CW_EXP_MIN
                    logging.info(f"AP{self.ap}:t{self.des_env.now:.9f}\t TX successfull, resetting CW to {self.cw}")
                
                # Log the transmission
                self.logger.log(self.run_number, self.des_env.now, frame.src, frame.dst, frame.size, frame.mcs, collision)
    