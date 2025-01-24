from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys

import numpy as np
from backpropagation.nn import NeuralNetwork, sum_outs
from backpropagation.activation import SigmoidActivation, ReluActivation, SoftmaxActivation
from backpropagation.loss import MeanSquareError, CrossEntropyLoss


ap = ArgumentParser(description="Neural Network in Numpy")
ap.add_argument("dataset")
ap.add_argument("-m", "--model")
ap.add_argument("-o", "--out")
ap.add_argument("-T", "--train-size", type=int)
ap.add_argument("-t", "--test-size", type=int)
ap.add_argument("-e", "--epochs", type=int, default=100)
ap.add_argument("-a", "--alpha", type=float, default=0.1)
args = ap.parse_args()

with Path(args.dataset).open("rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

def flatten_images(X: np.array) -> np.array:
    return np.array([x.flatten() for x in X]).T

X_train = flatten_images(X_train)
X_test = flatten_images(X_test)

def one_hot(y: np.array) -> np.array:
    oh = np.zeros((y.size, y.max() + 1))
    oh[np.arange(y.size), y] = 1
    return oh.T

y_train = one_hot(y_train)
y_test = one_hot(y_test)

train_size = args.train_size or len(X_train)
test_size = args.test_size or len(X_test)
epochs = args.epochs
alpha = args.alpha

if args.model:
    with Path(args.model).open("rb") as f:
        nn = pickle.load(f)
else:
    nn = NeuralNetwork(
        insize=784,
        layers=[128, 64, 10],
        activations=[
            SigmoidActivation(),
            SigmoidActivation(),
            SoftmaxActivation()
        ],
        loss=MeanSquareError(),
        learning_rate=alpha
    )
    nn.fit(X_train, y_train, train_size, epochs)

indices = np.random.choice(X_test.shape[1], size=test_size, replace=False)
X_sample = X_test[:, indices]
y_sample = y_test[:, indices]
forward_result = nn.forward(X_sample)
prediction = forward_result.prediction
mse_loss = np.sum(forward_result.loss(y_sample)) / test_size
predicted_answers = np.argmax(prediction, axis=0)
expected_answers = np.argmax(y_sample, axis=0)
correct = np.sum(predicted_answers == expected_answers)
print(f"{correct=} {mse_loss=}")

if args.out:
    with Path(args.out).open("wb") as f:
        pickle.dump(nn, f)