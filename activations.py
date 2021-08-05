import numpy as np


class ActivationFunction:

    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def derivative(x):
        raise NotImplementedError

    @staticmethod
    def local_gradient(layer, y):
        raise NotImplementedError


class Tanh(ActivationFunction):

    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def local_gradient(layer, y):
        if layer.next_layer is None:  # ouput layer
            local_gradient = Tanh.derivative(layer.net) * (layer.output - y)  # todo: implement losses
        else:
            local_gradient = Tanh.derivative(layer.net) * layer.error_derivative_o
        return local_gradient


class Softmax(ActivationFunction):

    @staticmethod
    def forward(x):
        shift_x = np.max(x)
        exp_x = np.exp(x - shift_x)
        sum_exp_x = np.sum(exp_x, axis=1)
        return exp_x / sum_exp_x.reshape(-1, 1)

    @staticmethod
    def local_gradient(layer, y):
        assert layer.next_layer is None, "Softmax activation should only be used on last layer."  # todo: remove
        local_gradient = layer.output - y  # todo: incorrect
        return local_gradient


class LogSoftmax(ActivationFunction):

    @staticmethod
    def forward(x):
        shift_x = np.max(x)
        exp_x = np.exp(x - shift_x)
        sum_exp_x = np.sum(exp_x, axis=1)
        return np.log(exp_x / sum_exp_x.reshape(-1, 1))

    @staticmethod
    def local_gradient(layer, y):
        assert layer.next_layer is None, "Softmax activation should only be used on last layer."  # todo: remove
        local_gradient = np.exp(layer.output) - y
        return local_gradient