from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np


class Space:
    def __init__(self, bounds:iter):
        self.__bounds = bounds

    def __add__(self, other):
        return self.__init__(other.bounds)

    @property
    def bounds(self):
        return self.__bounds


class ActiveApproach:
    def __init__(self, x0, learning_space:Space):
        self.__t = 0
        self.active_set = learning_space
        self.x = pd.DataFrame({0: x0})

    def sample(self):
        pass

    @property
    def t(self):
        return self.__t


class ActiveRanking(ActiveApproach):

    def sampling_distribution(self):
        pass
