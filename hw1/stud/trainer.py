import torch

from typing import Dict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


class Trainer:
    """
    This class can be used to facilitate the training of a torch model.

    It implements a training loop wrapped in the `train` method, with the optional monitoring of the loss over a
    validation `DataLoader` and an early stopping callback with optional patience.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_history = list()
        self.valid_history = list()
        self.global_epoch = 0

    def train(self, train_dataloader: DataLoader, dev_dataloader: DataLoader = None, epochs: int = 5,
              early_stopping: bool = False, early_stopping_patience: int = 0) -> dict:
        """
        Starts a training loop over the `train_dataloader`, monitoring the loss.

        :param train_dataloader: `DataLoader` for training
        :param dev_dataloader: `DataLoader` for validation and early stopping
        :param epochs: number of maximum training epochs
        :param early_stopping: `True` if the training should stop earlier in case of loss increase, `False` otherwise
        :param early_stopping_patience: number of epochs, in which loss increases, to wait before early stopping
        :return: a dictionary of the loss histories for training and validation
        """

        early_stopping_patience_counter = 0

        for epoch in range(epochs):
            losses = []
            self.global_epoch = epoch
            self.model.train()

            for batch in train_dataloader:
                self.optimizer.zero_grad()
                result = self.__predict(batch)
                loss = result['loss']
                losses.append(loss)

                loss.backward()
                self.optimizer.step()

            mean_loss = sum(losses) / len(losses)
            self.train_history.append(mean_loss.item())

            if dev_dataloader:
                validation_loss = self.evaluate(dev_dataloader)
                self.valid_history.append(validation_loss.item())

                if early_stopping and epoch > 0 and self.valid_history[-1] > self.valid_history[-2]:
                    if early_stopping_patience_counter < early_stopping_patience:
                        early_stopping_patience_counter += 1
                    else:
                        break
        return {
            'train_history': self.train_history,
            'valid_history': self.valid_history,
        }

    def __predict(self, batch) -> Dict[str, torch.Tensor]:
        """
        Computes the prediction based on the type of batch.

        In the case of RNN the `DataLoader` will return 3 values, instead of the default x, y pair
        (see also --> :func:`~utils.rnn_collate_fn`).

        :param batch: a batch element from a `DataLoader` instance
        :return: the model prediction
        """
        if len(batch) == 3:
            x, x_summary_position, y = batch
            return self.model(x, x_summary_position, y)
        elif len(batch) == 2:
            x, y = batch
            return self.model(x, y)

    def evaluate(self, validation_dataloader: DataLoader) -> torch.Tensor:
        """
        Evaluates the losses over a validation `DataLoader` and returns their mean

        :param validation_dataloader: the validation `DataLoader`
        :return: the mean of the validation `DataLoader` losses
        """
        losses = []
        self.model.eval()

        with torch.no_grad():
            for batch in validation_dataloader:
                result = self.__predict(batch)
                loss = result['loss']
                losses.append(loss)

        return sum(losses) / len(losses)

    def plot_losses(self):
        """
        Shows a plot of the training and validation losses histories
        """
        plt.figure(figsize=(8, 6))
        plt.plot(list(range(len(self.train_history))), self.train_history, label='Training loss')

        if len(self.valid_history) > 0:
            plt.plot(list(range(len(self.valid_history))), self.valid_history, label='Validation loss')
            plt.title('Training vs Validation loss')
        else:
            plt.title('Training loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()
