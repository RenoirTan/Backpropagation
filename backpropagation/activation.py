import numpy as np


class BaseActivation(object):
    def __init__(self):
        pass
    
    def activate(self, zs: np.array) -> np.array:
        raise NotImplemented
    
    def gradient(self, zs: np.array) -> np.array:
        raise NotImplemented


class SigmoidActivation(BaseActivation):
    def activate(self, zs: np.array) -> np.array:
        assert(zs.ndim == 1)
        return 1 / (1 + np.exp(-zs))
    
    def gradient(self, zs: np.array) -> np.array:
        a = self.activate(zs)
        return a * (1 - a)


class ReluActivation(BaseActivation):
    def activate(self, zs: np.array) -> np.array:
        assert(zs.ndim == 1)
        a = np.array(zs, copy=True)
        a[a < 0] = 0
        return a
    
    def gradient(self, zs: np.array) -> np.array:
        assert(zs.ndim == 1)
        a = np.ones(zs.shape)
        a[zs <= 0] = 0
        return a