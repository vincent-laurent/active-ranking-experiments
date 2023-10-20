import numba
import numpy as np


@numba.njit
def beta(t, c, c_prime, delta):
    return c * np.log(t / delta) + c_prime * np.log(np.log(t))


@numba.njit
def kl_bernoulli(p: float, q: np.array):
    if p == 0:
        return (1 - p) * np.log((1 - p) / (1 - q))
    elif p == 1:
        return p * np.log(p / q)
    else:
        loc3 = q == 0
        val3 = 1.

        loc4 = q == 1
        val4 = 1.

        val5 = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

        ret = np.where(loc3, val3, np.where(loc4, val4, val5))
    return ret


def lcb_index(mu, c, beta_, n):
    q_dist = np.linspace(0., 1., num=10000)
    if n == 0:
        return 0
    select = q_dist <= mu
    if np.sum(select) == 0:
        return 0
    if c * beta_ / n > 1:
        return 0
    q = q_dist[select]
    kl_dist = kl_bernoulli(mu, q)
    select2 = kl_dist <= c * beta_ / n
    if np.sum(select2) == 0:
        return 0
    i = np.where(select2)[0][0]
    return q[i]


def ucb_index(mu, c, beta_, n):
    q_dist = np.linspace(0., 1., num=10000)
    if n == 0:
        return 1
    select = q_dist >= mu
    if np.sum(select) == 0:
        return 1

    q = q_dist[select]
    kl_dist = kl_bernoulli(mu, q)
    select = kl_dist <= c * beta_ / n
    if np.sum(select) == 0:
        return 0
    i = np.argmax(kl_dist[select])
    return q[i]


@numba.njit
def compute_w_d_tracking_algorithm(delta: np.array):
    ret = np.zeros(delta.shape)
    for i, da in enumerate(delta):
        select = np.delete(delta, i)

        s = np.sum(da / select)
        ret[i] = 1. / (1. + s)
    return ret


@numba.njit
def compute_stopping_rule(mu: np.array, t: np.array, crit: float):
    for a, mu_a in enumerate(mu):
        for b, mu_b in enumerate(mu):
            if a != b:
                mu_ab = (t[a] * mu_a + t[b] * mu_b) / max((t[a] + t[b]), 0.1)
                z = t[a] * kl_bernoulli(mu_a, np.array([mu_ab]))[0]
                z += t[b] * kl_bernoulli(mu_b, np.array([mu_ab]))[0]
                if z > crit:
                    return True
    return False


@numba.njit
def _init(upper, lower):
    u = [np.array([0])[1:] for _ in range(len(upper))]
    d_dist = np.zeros(len(upper))
    for i in range(len(upper)):
        for j in range(len(upper)):
            if i != j:
                __c = upper[i] < lower[j]
                __c |= upper[j] < lower[i]

                if not __c:
                    u[i] = np.concatenate((u[i], np.array([j])))
                    m = max((
                        upper[j] - lower[i],
                        upper[i] - lower[j],
                        d_dist[i]))
                d_dist[i] = m
    return u, d_dist


# @numba.njit
# def _update(i, upper, lower):
#     ret = np.array([0])[1:]
#     for j in range(len(upper)):
#         __c = upper[i] < lower[j]
#         __c |= upper[j] > lower[i]
#         if not __c:
#             ret = np.concatenate((np.array([j]), ret))
#     return ret


class BoundsIntersection:
    """
    \tild U in the paper
    """

    def __init__(self, upper, lower):
        self.upper = upper
        self.lower = lower
        assert len(self) == len(self.lower)
        self.u, self.dist = _init(self.upper, self.lower)
        self.card = np.array([len(i) for i in self.u])

    def __len__(self):
        return len(self.upper)

    def __call__(self, i):
        return self.u[i]


upper = np.array([0.1, 0.2, 0.1, 0.001, 0.8])
lower = np.array([0, 0, 0, 0.0001, 0.5])
bi = BoundsIntersection(upper, lower)
self = bi
bi.dist
bi.u

_init(upper=np.array([1, 0.8]),
      lower=np.array([0, 0.3]))
compute_stopping_rule(mu=np.array([0.1, 0.2]), t=np.array([10, 15]), crit=100)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from importlib import reload

    reload(config)

    c = config.c
    c_prime = config.c_prime
    delta = config.delta
    mu = 0.
    b = beta(2, c, c_prime, delta)

    kl_bernoulli(0.01, np.array([0.2]))
    lcb_index(0.1, .2, 3, 10)
    ucb_index(0.1, .2, 3, 10)
    lcb_index(0.00000000000001, .2, 3, 10)

    range_ = np.linspace(1, 100, 1000)
    ucb = np.array([ucb_index(mu, c, b, i) for i in range_])
    lcb = np.array([lcb_index(mu, c, b, i) for i in range_])

    plt.figure()
    plt.plot(range_, ucb, label=f"$UCB(\mu ={mu})$")
    plt.plot(range_, lcb, label=f"$LCB(\mu ={mu})$")
    plt.legend()
    plt.ylim((-0.01, 1.01))
    plt.xlabel("Number of sample")


    ucb = np.array([ucb_index(mu, c, i, 100) for i in range_])
    lcb = np.array([lcb_index(mu, c, i, 100) for i in range_])

    plt.figure()
    plt.plot(range_, ucb, label=f"$UCB(\mu ={mu})$")
    plt.plot(range_, lcb, label=f"$LCB(\mu ={mu})$")
    plt.legend()
    plt.ylim((-0.01, 1.01))
    plt.xlabel("Number of sample")


    # DEGENERATED CASES
    ucb_index(0, 1, 1, 0)
    lcb_index(
        mu=1.0,
        c=1,
        beta_=9.536974631954463,
        n=1)
    lcb_index(mu=0.25, c=1, beta_=9.536974631954463, n=8)

    range_ = np.linspace(0, 1, 200)
    plt.figure()
    range_sample = [10, 50, 75, 100, 200, 10000]
    for i, n in enumerate(range_sample):
        c_ = config.cmap(i / (len(range_sample) - 1))
        ucb = np.array([ucb_index(mu, c, b, n) for mu in range_])
        lcb = np.array([lcb_index(mu, c, b, n) for mu in range_])

        plt.plot(range_, ucb - lcb, label=f"n sample = {n}")
    plt.legend()
    plt.ylabel("UCB($\mu$) - LCB($\mu$)")

    plt.xlabel("$\mu$")
