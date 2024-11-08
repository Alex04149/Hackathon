import numpy as np
from scipy.special import softmax
from typing import Literal

from util import extract_features, values_init, values_update, get_label, SiLU, SiLU_prime, sigmoid, sigmoid_prime
from data_loader import load_data
from errors import ModeError, LossFunctionError


class MLP:
    def __init__(self, mode: str, sizes: list[int]=None) -> None:
        _get_modes = lambda: ('rand', 'init')

        match mode:
            case 'rand':
                self.num_layers = len(sizes)
                self.sizes = sizes
                self._default_weight_initializer(sizes)
            case 'init':
                self.sizes, self.weights, self.biases = values_init()
                self.num_layers = len(self.sizes)
            case _:
                raise ModeError(f"Incorrect mode. Should be {_get_modes()}")

        self.losses_and_evaluates_list = []
        

    def forward(self, a):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = sigmoid(np.dot(w, a) + b)
        w_L, b_L = self.weights[-1], self.biases[-1]  # last layer
        z_L = np.dot(w_L, a) + b_L
        return np.round(softmax(z_L), 3)

    def SGD(self, training_data, test_data, epochs: int, mini_batch_size: int, learning_rate: float, weight_decay: float=0.0) -> None:
        eta = learning_rate
        lmbda = weight_decay

        losses = 0

        for _ in range(epochs):
            losses = 0 
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] 
                for k in range(0, len(training_data), mini_batch_size)
                ]

            for mini_batch in mini_batches:
                losses += self._update_mini_batch(mini_batch, eta, lmbda)
            self.losses_and_evaluates_list.append((losses/len(mini_batches),self.evaluate(test_data)))   
                
             

    def _update_mini_batch(self, mini_batch, eta, lmbda) -> float:
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]

        l = 0

        #region calculate nablas
        for x, y in mini_batch:
            delta_nabla_weights, delta_nabla_biases, loss = self._backprop(x, y)

            nabla_weights = [nw + dnw for nw, dnw in zip(nabla_weights, delta_nabla_weights)]
            nabla_biases = [nb + dnb for nb, dnb in zip(nabla_biases, delta_nabla_biases)]
            l += loss
            

        regulization_term = 1 - (eta * lmbda / 1158)  # 1158 - the number of test data
        self.weights = [regulization_term * w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_weights)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_biases)]
        return l/len(mini_batch)
        #endregion

    def _backprop(self, x, y) -> tuple[list, list, float]:
        def loss(x):
            j = np.argmax(y)
            return -np.log(x[j])

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        zs = []
        activation = x
        activations = [x]

        #region forward pass
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        w_L, b_L = self.weights[-1], self.biases[-1]

        z_L = np.dot(w_L, activation) + b_L
        zs.append(z_L)

        activation = softmax(z_L)
        activations.append(activation)
        #endregion

        #region backward pass
        delta = activations[-1] - y  # cross-entropy delta

        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2])

        for l in range(2, self.num_layers):
            w = self.weights[-l+1]

            z = zs[-l]
            spz = sigmoid_prime(z)

            delta = np.dot(w.T, delta) * spz  # hadamard product
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1])
        #endregion

        return (nabla_w, nabla_b, loss(activations[-1]))

    def evaluate(self, test_data) -> float:
        detections = 0

        for signal, label in test_data:
            prediction = np.argmax(self.forward(signal))
            detections += 1 if prediction == np.argmax(label) else 0

        acc = detections / 1158  # 1158 - the number of test data
        return acc

    def _default_weight_initializer(self, sizes) -> None:
        self.biases = [np.random.randn(y, ) for y in sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x) 
            for x, y in zip(sizes[:-1], sizes[1:])
            ]

    def _large_weight_initializer(self, sizes) -> None:
        self.biases = [np.random.randn(y, ) for y in sizes[1:]]
        self.weights = [
            np.random.randn(y, x) 
            for x, y in zip(sizes[:-1], sizes[1:])
            ]


if __name__ == "__main__":
    net = MLP(mode='rand', sizes=[938, 20, 20, 10, 2])

    training_data = load_data('training')

    print(net.forward(training_data[0][0]))
    print(net.forward(training_data[7][0]))
    print(net.forward(training_data[12][0]))

    # net.backprop(training_data[0][0], np.array([0, 1]))
    # print(temp)
    # temp = net.forward(training_data[1][0])
    # print(temp)
    # print(temp)
