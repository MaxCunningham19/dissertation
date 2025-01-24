import random
import math
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Linear,
    ReLU,
    CrossEntropyLoss,
    Sequential,
    Conv2d,
    MaxPool2d,
    Module,
    Softmax,
    BatchNorm2d,
    Dropout,
)


class QN(nn.Module):
    """
    Basic Model of a Q network
    """

    def __init__(self, state_number, hidlyr_nodes, action_number):
        super(QN, self).__init__()
        self.fc1 = nn.Linear(state_number, hidlyr_nodes)  # first conected layer
        self.fc2 = nn.Linear(hidlyr_nodes, hidlyr_nodes * 2)  # second conected layer
        self.fc3 = nn.Linear(hidlyr_nodes * 2, hidlyr_nodes * 4)  # third conected layer
        self.fc4 = nn.Linear(hidlyr_nodes * 4, hidlyr_nodes * 8)  # fourth conected layer
        self.out = nn.Linear(hidlyr_nodes * 8, action_number)  # output layer

    def forward(self, state):
        x = F.relu(self.fc1(state))  # relu activation of fc1
        x = F.relu(self.fc2(x))  # relu activation of fc2
        x = F.relu(self.fc3(x))  # relu activation of fc3
        x = F.relu(self.fc4(x))  # relu activation of fc4
        x = self.out(x)  # calculate output
        return x
