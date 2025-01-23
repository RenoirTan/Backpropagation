from dataclasses import dataclass
from typing import List
import sys

import numpy as np

from backpropagation.activation import BaseActivation
from backpropagation.loss import BaseLoss


def sum_outs(activated: np.array) -> np.array:
    activated = np.atleast_2d(activated)
    return np.atleast_2d(np.sum(activated, axis=1)).T


class NeuralNetwork(object):
    def __init__(
        self,
        insize: int,
        layers: List[int],
        activations: List[BaseActivation],
        loss: BaseLoss,
        learning_rate: float
    ):
        self.insize = insize
        self.layers = layers
        self.loss = loss
        self.alpha = learning_rate
        self.F = activations
        # list of weight matrices where each matrix are of dimensions (m, n)
        # where n is the number of input neurons and m is the number of output
        # neurons
        self.W = []
        # list of bias matrices where each matrix is of dimension (m, 1)
        # i.e. a column vector of size m
        self.B = []
        # Given an input matrix L of 2 inputs each of size 4, (shape = (4, 2)),
        # a weight matrix W of dimension (3, 4)
        # and a bias matrix B of dimension (3, 1)
        # The output matrix is given by W @ L + B
        # Although, W @ L is of dimension (3, 2), it can be added to B as if
        # the bias layer is being added to each output layer separately
        # (B behaves like it is shape (3, 2) where B[i][0] == B[i][1])
        for outsize in layers:
            self.W.append(np.random.randn(outsize, insize))
            self.B.append(np.random.randn(outsize, 1))
            insize = outsize
    
    def forward(self, X: np.array) -> "ForwardResult":
        X = np.atleast_2d(X)
        Z = []
        A = [X]
        # print(f"{X.shape=}")
        for li in range(len(self.W)):
            # print(f"{self.W[li].shape=} {A[-1].shape=}")
            Z.append(self.W[li] @ A[-1] + self.B[li])
            A.append(self.F[li].mass_activate(Z[-1]))
        return ForwardResult(self, Z, A)
    
    def backprop(self, forward_result: "ForwardResult", ys: np.array):
        ys = np.atleast_2d(ys)
        Z = list(forward_result.Z)
        Z.insert(0, None) # bit of padding
        A = forward_result.A
        D = [self.F[-1].mass_gradient(A[-1]) * self.loss.mass_gradient(A[-1], ys)]
        for li in range(len(self.layers) - 1, 0, -1):
            # print(f"{self.W[li].shape=} {D[0].shape=}")
            # print(f"{self.F[li-1].mass_gradient(Z[li]).shape=}")
            D.insert(0, self.F[li-1].mass_gradient(Z[li]) * (self.W[li].T @ D[0]))
        for li in range(len(self.W)):
            # print(f"{(D[li] @ A[li].T).shape=}")
            self.W[li] -= self.alpha * (D[li] @ A[li].T)
            self.B[li] -= self.alpha * sum_outs(D[li])
    
    def fit(self, X: np.array, y: np.array, train_size: int=0, epochs: int=100):
        train_size = train_size or X.shape[1] // 10
        for epoch in range(epochs):
            indices = np.random.randint(0, X.shape[1], size=train_size)
            loss = 0
            for i in indices:
                sx = X[:, [i]]
                sy = y[:, [i]]
                self.backprop(self.forward(sx), sy)
                loss += np.sum(self.forward(sx).loss(sy))
            loss /= train_size
            print(f"{epoch=} {loss=}")


@dataclass
class ForwardResult(object):
    nn: NeuralNetwork
    Z: List[np.array]
    A: List[np.array]
    
    @property
    def prediction(self) -> np.array:
        return self.A[-1]
    
    def loss(self, ys: np.array) -> np.array:
        return self.nn.loss.mass_loss(self.A[-1], ys)