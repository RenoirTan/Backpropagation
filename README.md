# Backpropagation

This is my attempt at creating a deep learning model using just numpy. This is based on the demonstration found in [this article by Adrian Rosebrock](https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/). Key modifications are the transposing of the weight and input matrices in order to match the notation in the [Backpropagation wiki page](https://en.wikipedia.org/wiki/Backpropagation), and separating the biases from the weights so that the inputs to each bias is always 1. The examples shown below use the fashion MNIST data from [this repo](https://github.com/zalandoresearch/fashion-mnist). Since each image is 28 by 28 pixels, the total number of inputs per image is 784.

## Setup

Clone the `fashion-mnist` repository and follow the instructions on their README.md to get the training and test data.

```bash
git clone https://github.com/zalandoresearch/fashion-mnist
cd fashion-mnist/utils
python -c $(echo "
import mnist_reader
X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

from pathlib import Path
import pickle
with Path('../fashion.pkl').open('wb') as f:
    pickle.dump((X_train, y_train, X_test, y_test), f)
")
cd ..
ls fashion.pkl
```

The training and test data can then be found in fashion.pkl.

Run the test model with:

```bash
git clone https://github.com/RenoirTan/Backpropagation
poetry install
poetry shell
python -m backpropagation /path/to/fashion.pkl -o /path/to/output_model.pkl -T 10 -t 10000 -e 10000 -a 0.01
```

## Neural Network

Creation of the neural network:

```python
from backpropagation.nn import NeuralNetwork
from backpropagation.activation import SigmoidActivation, SoftmaxActivation
from backpropagation.loss import MeanSquareError

nn = NeuralNetwork(
    insize=784,
    layers=[128, 64, 10],
    activations=[
        SigmoidActivation(),
        SigmoidActivation(),
        SoftmaxActivation()
    ],
    loss=MeanSquareError(),
    learning_rate=0.01
)
```

This creates 3 weight layers of shapes (128, 784), (64, 128) and (10, 64) and 3 bias layers (128, 1), (64, 1) and (10, 1). The weights and biases are all initialised to random numbers using `np.random.randn`.

The activation functions are defined in the `backpropagation.activation` module. These functions are classes with 2 methods: `activate` and `gradient`. The activation functions are stored in a field named `F`. Similarly, the loss function has `loss` and `gradient` functions.

The `NeuralNetwork.forward` method looks like this:

```python
    def forward(self, X: np.array) -> "ForwardResult":
        X = np.atleast_2d(X)
        Z = []
        A = [X]
        for li in range(len(self.W)):
            Z.append(self.W[li] @ A[-1] + self.B[li])
            A.append(self.F[li].mass_activate(Z[-1]))
        return ForwardResult(self, Z, A)
```

`X` is the input matrix. Let's say for each epoch, 100 images are inputted. `X` will have a shape of (784, 100). `Z` is a list of inputs to the activation functions while `A` is a list of outputs from the activation functions. For the first layer of weights (li == 0):

$Z_1 = W_1 \cdot X + B_1$

$A_1 = F_1(Z_1)$

The new inputs are then passed into the next layer and so on:

$Z_2 = W_2 \cdot A_1 + B_2$

$A_2 = F_2(Z_2)$

The outputs are then stored in a `ForwardResult` object, which along with the target outputs `ys` are passed into the `backprop` method:

```python
    def backprop(self, forward_result: "ForwardResult", ys: np.array):
        ys = np.atleast_2d(ys)
        Z = list(forward_result.Z)
        Z.insert(0, None) # bit of padding
        A = forward_result.A
        D = [self.F[-1].mass_gradient(Z[-1]) * self.loss.mass_gradient(A[-1], ys)]
        for li in range(len(self.layers) - 1, 0, -1):
            D.insert(0, self.F[li-1].mass_gradient(Z[li]) * (self.W[li].T @ D[0]))
        for li in range(len(self.W)):
            self.W[li] -= self.alpha * (D[li] @ A[li].T)
            self.B[li] -= self.alpha * sum_outs(D[li])
```

The line

```python
        D = [self.F[-1].mass_gradient(Z[-1]) * self.loss.mass_gradient(A[-1], ys)]
```

represents the error associated with the last layer of the neural network: $\delta_3 = (F_3)' \odot \nabla C$,
with the error for preceding layers calculated with the following line:

```python
            D.insert(0, self.F[li-1].mass_gradient(Z[li]) * (self.W[li].T @ D[0]))
```

or $\delta_{l-1} = (F_{l-1}) \odot (W_l)^T \cdot \delta_l$

For each layer $l$, the amount that each weight should be altered by can calculated by $\Delta w_{ij} = -\alpha \delta_j a_i$ where $a_i$ is the activated output of the previous layer. Hence, change in for each weight in that layer can be obtained using $\Delta W_l = -\alpha (\delta \cdot A_{l-1})$. Likewise, for the biases, the change in bias is $\Delta b_j = -\alpha (\delta_j \cdot 1) = -\alpha \delta_j$. The function `sum_outs` sums the `error` calculated over all input samples for each neuron.

```python
        for li in range(len(self.W)):
            self.W[li] -= self.alpha * (D[li] @ A[li].T)
            self.B[li] -= self.alpha * sum_outs(D[li])
```

With the logic out of the way, the input to the neural network should be a column vector of 784 numbers and the output a column vector of 10 numbers. Multiple inputs and outputs can be fed through the model at the same time by placing all of the column vectors into one matrix. For example, if you want to pass in 100 images at the same time, the input matrix should have 784 rows and 100 columns and likewise for the outputs, 10 rows and 100 columns.

## Evaluation

It's terrible. For reference, on tensorflow, you should be getting accuracies of around 98%.

But hey! At least it's better than random chance.

| Samples Per Epoch (-T) | Epochs (-e) | Learning Rate (-a) | Accuracy |
| ---------------------- | ----------- | ------------------ | -------- |
| 300 | 1000 | 0.01 | 56.87% |
| 100 | 3000 | 0.01 | 64.71% |
| 32 | 12288 | 0.01 | 64.52% |
| 32 | 12288 | 0.03 | 46.31% |