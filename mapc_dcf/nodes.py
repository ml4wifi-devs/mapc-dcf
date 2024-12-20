import jax
from chex import Array, PRNGKey

import simpy

from mapc_dcf.channel import Channel, AMPDU
from mapc_dcf.dcf import DCF
from mapc_dcf.logger import Logger


class AccessPoint():

    def __init__(
            self,
            key: PRNGKey,
            id: int,
            position: Array,
            tx_power: float,
            mcs: int,
            clients: Array,
            channel: Channel,
            des_env: simpy.Environment,
            logger: Logger
    ) -> None:
        self.key, key_dcf = jax.random.split(key)
        self.id = id
        self.position = position
        self.tx_power = tx_power
        self.mcs = mcs
        self.clients = clients
        self.channel = channel
        self.des_env = des_env
        self.dcf = DCF(key_dcf, self.id, self.des_env, self.channel, logger, self.frame_generator)
        self.frame_id = 0
    

    def frame_generator(self) -> AMPDU:
        """
        Generate an 802.11 frame to be sent by the AP.

        Returns
        -------
        WiFiFrame
            An 802.11 frame to be sent by the AP.
        """

        self.key, key_frame = jax.random.split(self.key)
        dst = jax.random.choice(key_frame, self.clients).item()
        frame = AMPDU(self.frame_id, self.id, dst, self.tx_power, self.mcs, self.channel.is_sr_on)
        self.frame_id += 1

        return frame
    

    def start_operation(self, run_number: int):
        self.dcf.start_operation(run_number)
