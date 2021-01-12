import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import math
import time
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


class EHRNN(nn.Module):
    """ Elman Network for classification that keeps track of the statistics of a pooled version of the hidden states
    across time. """

    def __init__(self, seq_len, num_classes, input_size, hidden_size,
                 num_layers, EHeps, EHlengths, useMean=True,
                 useVariance=False, inputToLinear='all'):

        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.Softmax = nn.Softmax(dim=1)
        self.EHeps = EHeps
        self.EHlengths = EHlengths
        self.useVariance = useVariance
        self.useMean = useMean
        self.inputToLinear = inputToLinear

        # each hidden size will be reduced to size sqrt(hidden_size). Then,
        # each element in it will go to an EH.
        self.avgKernelSize = int(np.floor(np.sqrt(hidden_size)))
        # from https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html
        self.numEHs = math.floor(((self.hidden_size - self.avgKernelSize) / self.avgKernelSize) + 1)
        self.avgPool = nn.AvgPool1d(kernel_size=self.avgKernelSize)

        if not useMean and not useVariance:
            raise Exception("At least one type of estimate must be used.")
        else:
            self.numberOfEstimates = 2 if useMean and useVariance else 1

        # EHs[i][j] is EH over pooled element i of some length EHlengths[j]
        self.EHs = [[VarEH(len, eps=EHeps, maxValue=1) for len in EHlengths] for _ in range(self.numEHs)]

        # linear: its input size depends on hidden size, how many EH we maintain
        # and how many estimates we query
        if self.inputToLinear == 'all':
            self.linear = nn.Linear(hidden_size + self.numEHs * len(EHlengths) * self.numberOfEstimates, num_classes)
        elif self.inputToLinear == 'estimates':
            self.linear = nn.Linear(self.numEHs * len(EHlengths) * self.numberOfEstimates, num_classes)
        else:
            raise Exception("Input to linear must be either 'all' or 'estimates'")

    def forward(self, x):
        linearInput = self.hidden_states(x)
        linear_out = self.linear(linearInput)
        return self.Softmax(linear_out)

    def hidden_states(self, x):
        batch_size = x.size()[0]
        # assuming batch_first = True for RNN cells
        hidden = self._init_hidden(batch_size)
        hidden = hidden.to(hparams['device'])
        x = x.view(batch_size, self.seq_len, self.input_size)

        # apart from the output, rnn also gives us the hidden
        # cell, this gives us the opportunity to pass it to
        # the next cell if needed; we won't be needing it here
        # because the nn.RNN already computed all the time steps
        # for us. rnn_out will of size [batch_size, seq_len, hidden_size]
        # rnn_out: B x 1 x H
        rnn_out, _ = self.rnn(x, hidden)

        # add hidden states to EHs, getting the mean each time so as to not have
        # future hidden states.
        allMeans = torch.tensor([])
        rnn_pooled = torch.squeeze(self.avgPool(rnn_out))  # B x numEH
        for pointIndex, point in enumerate(rnn_pooled):
            for i, element in enumerate(point):
                for j in range(len(self.EHs[i])):
                    self.EHs[i][j].add(element.item())

            # get the estimates at this point so as to not look into the future
            if self.useVariance and self.useMean:
                estimates = []
                for i in range(self.numEHs):
                    for j in range(len(self.EHs[i])):
                        estimates.append(self.EHs[i][j].get_mean_estimate())
                        estimates.append(self.EHs[i][j].get_var_estimate())
            elif self.useMean:
                estimates = [self.EHs[i][j].get_mean_estimate() for i in range(self.numEHs) for j in
                             range(len(self.EHs[i]))]
            elif self.useVariance:
                estimates = [self.EHs[i][j].get_var_estimate() for i in range(self.numEHs) for j in
                             range(len(self.EHs[i]))]

            allMeans = torch.cat((allMeans, torch.tensor(estimates)))

        allMeans = allMeans.to(hparams['device'])

        if self.inputToLinear == 'all':
            # rnn_out: B x (H + self.numEHs * len(EHlengths) * self.numberOfEstimates)
            linearInput = torch.cat((torch.squeeze(rnn_out), allMeans.view(batch_size, self.numEHs * len(
                self.EHlengths) * self.numberOfEstimates)), 1)
            linearInput = linearInput.view(batch_size, self.hidden_size + self.numEHs * len(
                self.EHlengths) * self.numberOfEstimates)
        else:
            # rnn_out: B x (self.numEHs * len(EHlengths) * self.numberOfEstimates)
            linearInput = allMeans.view(batch_size, self.numEHs * len(self.EHlengths) * self.numberOfEstimates)

        return linearInput

    def _init_hidden(self, batch_size):
        """
        Initialize hidden cell states, assuming
        batch_first = True for RNN cells
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
