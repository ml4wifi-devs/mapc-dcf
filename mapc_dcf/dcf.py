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
        self.tx_power = DEFAULT_TX_POWER
    
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

        logging.info(f"AP{self.ap}: DCF running")

        # While ready to send frames
        while True:
            frame = self.frame_generator()

            # Try sending the frame until transmitted successfully
            frame_sent_successfully = False
            while not frame_sent_successfully:
                
                # First condition: channel is idle
                channel_idle = False
                while not channel_idle:
                    logging.info(f"AP{self.ap}: Channel busy, waiting for idle channel")

                    # Wait for DIFS
                    while not channel_idle:
                        yield self.des_env.timeout(SLOT_TIME)
                        channel_idle = self.channel.is_idle_for(self.des_env.now, DIFS, frame.src)
                    logging.info(f"AP{self.ap}: Channel idle for DIFS")
                    
                    # Initialize backoff counter
                    key_backoff, self.key = jax.random.split(self.key)
                    backoff_counter = jax.random.randint(key_backoff, shape=(1,), minval=0, maxval=self.cw+1).item()
                    logging.info(f"AP{self.ap}: Backoff counter initialized to {backoff_counter}")

                    # Second condition: backoff counter is zero
                    while channel_idle and backoff_counter > 0:
                        logging.info(f"AP{self.ap}: Backoff counter: {backoff_counter}")

                        # If not, wait for one slot and check again both conditions
                        yield self.des_env.process(self.wait_for_one_slot())
                        backoff_counter -= 1
                        channel_idle = self.channel.is_idle(self.des_env.now, frame.src)
                
                logging.info(f"AP{self.ap}: Backoff counter: 0")
                logging.info(f"AP{self.ap}: Channel is idle and backoff counter is zero. Sending frame...")
                
                # If both conditions are met, send the frame
                yield self.des_env.timeout(SIFS)            # TODO Try to remove this line
                self.channel.send_frame(frame, self.des_env.now, self.tx_power)
                yield self.des_env.timeout(frame.duration)  # TODO Include ACK time
                frame_sent_successfully = self.channel.is_succesfully_transmitted(frame)
                yield self.des_env.timeout(SIFS)            # TODO Try to remove this line

                # Act according to the transmission result
                if frame_sent_successfully:
                    logging.info(f"AP{self.ap}: Frame sent successfully! Resetting CW to {2**CW_EXP_MIN}")
                    self.logger.log(self.run_number, self.des_env.now, frame.src, frame.dst, frame.size, frame.mcs, False)
                    frame_sent_successfully = True
                    self.cw = 2**CW_EXP_MIN
                else:
                    logging.info(f"AP{self.ap}: Collision detected! Increasing CW")
                    self.logger.log(self.run_number, self.des_env.now, frame.src, frame.dst, frame.size, frame.mcs, True)
                    self.cw = min(2*self.cw, 2**CW_EXP_MAX)
                    logging.info(f"AP{self.ap}: Retrying with CW={self.cw}")
    