import math

import numpy as np


class Linear:

    def __init__(self, num_inp: int, num_out: int, activation: str):
        xavier_range = 1/math.sqrt(num_inp)
        self.weights = np.random.uniform(low=-xavier_range, high=xavier_range, size=(num_out, num_inp+1))
        if activation == "tanh":
            self.activation = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x)**2
        else:
            raise ValueError(f"Activation function {activation} is unknown.")

        self.net = None
        self.output = None
        self.weight_adjust = None

    def forward(self, x: np.ndarray):
        # x: (batch, input)
        x_bias = np.pad(x, ((0, 0), (0, 1)), 'constant', constant_values=(1,))  # (batch, num_inp+1)
        self.net = x_bias @ self.weights.T  # (batch, num_out)
        self.output = self.activation(self.net)  # (batch, num_out)
        return self.output

    def backward(self, error: np.ndarray, prev_output: np.ndarray, lr: float):
        # todo: implement optimizer
        # error: (batch, num_out)
        # prev_output: (batch, num_inp) o_i
        delta = self.activation_derivative(self.net) * error  # (batch, num_out) delta_j
        prev_output_bias = np.pad(prev_output, ((0, 0), (0, 1)), 'constant', constant_values=(1,))  # (batch, num_inp+1)
        self.weight_adjust = - lr * np.einsum('ij,ik->ijk', delta, prev_output_bias, optimize=True)  # (batch, num_out, num_inp) w_ij
        error_new = (delta @ self.weights)[:, :-1]  # (batch, num_inp)
        return error_new

    def adjust_weights(self):
        self.weights += self.weight_adjust.mean(axis=0)
