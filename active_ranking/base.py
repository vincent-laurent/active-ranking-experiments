import numpy as np
import pandas as pd


class SamplerProcess:
    def __init__(self, first_point: float):
        self.__x0 = first_point
        self.__points = np.empty(int(1e6)) * np.nan
        self.__i = 0
        self.__points[0] = first_point

    def add(self, x: np.ndarray):
        i = self.__i
        self.__points[i:i + len(x)] = x
        self.__i += len(x)


class Space:
    def __init__(self, bounds: iter):
        self.__bounds = bounds
        self.__min = min(bounds)
        self.__max = max(bounds)

    def isin(self, x: np.ndarray):
        return (self.__min <= x) & (x <= self.__max)

    def sample(self, n: int) -> np.ndarray:
        return np.random.uniform(
            self.__min, self.__max, size=n)

    def __add__(self, other):
        return self.__init__(other.bounds)

    @property
    def bounds(self):
        return self.__bounds


class ActiveApproach:
    def __init__(self, x0, learning_space):
        self.__t = 0
        self.active_set = Space(learning_space)
        self.sampler = SamplerProcess(x0)
        self.x = pd.Series({0: x0})

    def sample(self):
        pass

    @property
    def t(self):
        return self.__t


class ActiveRanking(ActiveApproach):

    def sampling_distribution(self):
        pass
