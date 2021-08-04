import numpy as np


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.x = None

    def forward(self, x: np.ndarray):
        self.x = x
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y: np.ndarray, lr: float):
        for layer in self.layers[::-1]:
            layer.backward(self.x, y, lr)

    def adjust_weights(self):
        for layer in self.layers:
            layer.adjust_weights()

    def add_layer(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-2].next_layer = self.layers[-1]
            self.layers[-1].previous_layer = self.layers[-2]

    def train(self, x_train, y_train, num_epochs, batch_size, lr, shuffle):
        num_train_samples = len(x_train)
        assert num_train_samples >= batch_size, "Batch size is smaller than training set."
        assert len(x_train.shape) == 2 and len(y_train.shape) == 2, "Training data has invalid shape."
        order = np.arange(num_train_samples)
        if shuffle:
            np.random.shuffle(order)
        epoch = 0
        idx = 0
        while epoch < num_epochs:
            if idx + batch_size <= num_train_samples:
                x_batch = x_train[order[idx: idx + batch_size]]
                y_batch = y_train[order[idx: idx + batch_size]]
            else:
                x_batch = x_train[np.append(order[idx:], order[:num_train_samples - idx])]
                y_batch = y_train[np.append(order[idx:], order[:num_train_samples - idx])]
            assert len(x_batch) == batch_size  # todo: remove
            y_pred = self.forward(x_batch)
            self.backward(y_batch, lr)
            self.adjust_weights()
            idx += batch_size
            if idx >= num_train_samples:
                idx -= num_train_samples
                print("error:", np.mean((y_pred - y_batch)**2))  # todo: criterion
                epoch += 1
