import numpy as np


class BaseLoss(object):
    def __init__(self):
        pass
    
    def loss(self, y_hats: np.array, ys: np.array) -> np.array:
        raise NotImplemented
    
    def mass_loss(self, y_hatss: np.array, yss: np.array) -> np.array:
        return np.array([self.loss(y_hats, ys) for y_hats, ys in zip(y_hatss.T, yss.T)]).T
    
    def gradient(self, y_hats: np.array, ys: np.array) -> np.array:
        raise NotImplemented
    
    def mass_gradient(self, y_hatss: np.array, yss: np.array) -> np.array:
        return np.array([self.gradient(y_hats, ys) for y_hats, ys in zip(y_hatss.T, yss.T)]).T


class MeanSquareError(BaseLoss):
    def loss(self, y_hats: np.array, ys: np.array) -> np.array:
        # assert(y_hats.ndim == 1)
        assert(y_hats.shape == ys.shape)
        return 0.5 * ((y_hats - ys) ** 2)
    
    def gradient(self, y_hats: np.array, ys: np.array) -> np.array:
        assert(y_hats.shape == ys.shape)
        return y_hats - ys