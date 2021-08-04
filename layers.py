import math

import numpy as np

from activations import ActivationFunction


class Linear:

    def __init__(self, num_inp: int, num_out: int, activation_function: ActivationFunction):
        xavier_range = 1/math.sqrt(num_inp)
        self.weights = np.random.uniform(low=-xavier_range, high=xavier_range, size=(num_out, num_inp+1))
        self.activation_function = activation_function

        self.previous_layer = None
        self.next_layer = None

        self.net = None
        self.output = None
        self.weight_adjust = None
        self.error_derivative_o = None

    def forward(self, x: np.ndarray):
        # x: (batch, input)
        x_bias = np.pad(x, ((0, 0), (0, 1)), 'constant', constant_values=(1,))  # (batch, num_inp+1)
        self.net = x_bias @ self.weights.T  # (batch, num_out)
        self.output = self.activation_function.forward(self.net)  # (batch, num_out)
        return self.output

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float):
        # todo: implement optimizer
        # error: (batch, num_out)
        # prev_output: (batch, num_inp) o_i

        local_gradient = self.activation_function.local_gradient(self, y)  # (batch, num_out) delta_j

        if self.previous_layer is None:  # first layer
            prev_output_bias = np.pad(x, ((0, 0), (0, 1)), 'constant', constant_values=(1,))  # (batch, num_inp+1)
        else:
            prev_output_bias = np.pad(self.previous_layer.output, ((0, 0), (0, 1)), 'constant', constant_values=(1,))  # (batch, num_inp+1)
            self.previous_layer.error_derivative_o = (local_gradient @ self.weights)[:, :-1]  # (batch, num_inp)

        error_derivative_w = np.einsum('ij,ik->ijk', local_gradient, prev_output_bias, optimize=True)  # (batch, num_out, num_inp)
        self.weight_adjust = - lr * error_derivative_w  # (batch, num_out, num_inp) w_ij

    def adjust_weights(self):
        self.weights += self.weight_adjust.mean(axis=0)
