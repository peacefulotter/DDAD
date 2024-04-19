import torch
import os
import torch.nn as nn
from dataset import *

from dataset import *
from loss import *


def trainer(model, category, config):
    """
    Training the UNet model
    :param model: the UNet model
    :param category: the category of the dataset
    """
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=config.model.learning_rate)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
