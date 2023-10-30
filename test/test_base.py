import numpy as np

from base import Space


def test_space():
    space = Space([0, 1])
    assert all(
        space.isin(np.array([0, 0.1, 2])) == np.array(
            [True, True, False]))
    assert space.sample(10).__len__() == 10
