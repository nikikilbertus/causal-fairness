import torch
import torch.nn as nn
from torch.autograd import Variable
import collections


class MLP(nn.Module):
    """A simple fully connected feed forward network."""

    def __init__(self, sizes, final=None):
        """
        Initialize the network.

        A variable size network with only fully connected layers and SELU
        activations after all but the last layer.

        Arguments:

        sizes: A list of the numbers of neurons in the layers.
               len(sizes)-1 is the number of layers.
               First and last entries are input and output dimension.

        final: What to use as a final layer, e.g. torch.nn.Sigmoid()
               None (default) means no final layer.

        Example:
            A network with 2-dimensional input, one hidden layer with 128
            neurons and 1-dimensional output for regression:

            >>> net = MLP([2, 128, 1])

            A network with 10-dimensional input, two hidden layers of 128 and
            256 neurons and 1-dimensional output for classification:

            >>> net = MLP([10, 128, 256, 1], final=torch.nn.Sigmoid())
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        # If there is only one input dimension, everything is fine
        if sizes[0] == 1:
            self.layers.append(nn.Linear(sizes[0], sizes[1]))

        # For multiple input dimensions, each one has a separate following
        # hidden layer.
        # This is necessary for the partial training later on.
        else:
            self.layers.append(nn.ModuleList([nn.Linear(1, sizes[1])
                                              for _ in range(sizes[0])]))

        # Add the remaining layers with selu activations
        for i in range(len(sizes) - 1)[1:]:
            if i != (len(sizes) - 1):
                self.layers.append(nn.SELU())
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        if final is not None:
            self.layers.append(final)

    def forward(self, x):
        """The forward pass."""
        # If there are multiple inputs, add up their hidden layers
        if isinstance(self.layers[0], collections.Iterable):
            y = self.layers[0][0](x[:, 0, None])
            for i in range(1, len(self.layers[0])):
                y += self.layers[0][i](x[:, i, None])
            return nn.Sequential(*[self.layers[i]
                                   for i in range(1, len(self.layers))])(y)
        # Otherwise just build a simple sequential model
        else:
            return nn.Sequential(*self.layers)(x)


def train(net, x, y, loss_func=nn.MSELoss(), epochs=50, batchsize=32):
    """
    Train a network on data.

    Arguments:

        net:       A network module

        x:         Training input data

        y:         Training labels

        loos_func: Loss function (default nn.MSELoss())

        n_epochs:  Number of training epochs.
    """
    opt = torch.optim.Adam(net.parameters())
    n_samples = x.size(0)
    for epoch in range(epochs):
        # Shuffle training data
        p = torch.randperm(n_samples).long()
        xp = x[p]
        yp = y[p]

        for i1 in range(0, n_samples, batchsize):
            # Extract a batch
            i2 = min(i1 + batchsize, n_samples)
            xi, yi = xp[i1:i2], yp[i1:i2]

            # Reset gradients
            opt.zero_grad()

            # Forward pass
            loss = loss_func(net(Variable(xi)), Variable(yi))

            # Backward pass
            loss.backward()

            # Parameter update
            opt.step()
    return net
