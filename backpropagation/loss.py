import numpy as np


class BaseLoss(object):
    def __init__(self):
        pass
    
    def loss(self, y_hats: np.array, ys: np.array) -> np.array:
        raise NotImplemented
    
    def gradient(self, y_hats: np.array, ys: np.array) -> np.array:
        raise NotImplemented


class MeanSquareError(BaseLoss):
    def loss(self, y_hats: np.array, ys: np.array) -> np.array:
        # assert(y_hats.ndim == 1)
        assert(y_hats.shape == ys.shape)
        return 0.5 * ((y_hats - ys.shape) ** 2)
    
    def gradient(self, y_hats: np.array, ys: np.array) -> np.array:
        assert(y_hats.shape == ys.shape)
        return y_hats - ys