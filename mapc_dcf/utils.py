import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import t

from mapc_dcf.constants import *


def timestamp(time: float) -> str:
    t = f"{10**6 * time:.9f} us"
    leading_zeros = max(6 - len(t.split(".")[0]), 0)
    return "0" * leading_zeros + t


def tgax_path_loss(distance: jax.Array, walls: jax.Array) -> jax.Array:
    r"""
    Calculates the path loss according to the TGax channel model [1]_.

    Parameters
    ----------
    distance: Array
        Distance between nodes
    walls: Array
        Adjacency matrix describing walls between nodes (1 if there is a wall, 0 otherwise).

    Returns
    -------
    Array
        Path loss in dB

    References
    ----------
    .. [1] https://www.ieee802.org/11/Reports/tgax_update.htm#:~:text=TGax%20Selection%20Procedure-,11%2D14%2D0980,-TGax%20Simulation%20Scenarios
    """

    return (40.05 + 20 * jnp.log10((jnp.minimum(distance, BREAKING_POINT) * CENTRAL_FREQUENCY) / 2.4) +
            (distance > BREAKING_POINT) * 35 * jnp.log10(distance / BREAKING_POINT) + WALL_LOSS * walls)


def logsumexp_db(a: jax.Array, b: jax.Array) -> jax.Array:
    r"""
    Computes :func:`jax.nn.logsumexp` for dB i.e. :math:`10 * \log_{10}(\sum_i b_i 10^{a_i/10})`

    This function is equivalent to

    .. code-block:: python

        interference_lin = jnp.power(10, a / 10)
        interference_lin = (b * interference_lin).sum()
        interference = 10 * jnp.log10(interference_lin)


    Parameters
    ----------
    a: Array
        Parameters are the same as for :func:`jax.nn.logsumexp`
    b: Array
        Parameters are the same as for :func:`jax.nn.logsumexp`

    Returns
    -------
    Array
        ``logsumexp`` for dB
    """

    LOG10DIV10 = jnp.log(10.) / 10.
    return jax.nn.logsumexp(a=LOG10DIV10 * a, b=b) / LOG10DIV10


def confidence_interval(data: np.ndarray, ci: float = 0.95) -> tuple:
    measurements = data.shape[0]
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    alpha = 1 - ci
    z = t.ppf(1 - alpha / 2, measurements - 1)

    ci_low = mean - z * std / np.sqrt(measurements)
    ci_high = mean + z * std / np.sqrt(measurements)

    return mean, ci_low, ci_high
