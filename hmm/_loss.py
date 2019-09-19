import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class HingeLoss(torch.nn.Module):
    """
    Multiclass Hinge loss.
    """
    def __init__(self, margin=0.2):
        super(HingeLoss, self).__init__()
        self._margin = margin
    
    def forward(self, beliefs, labels):
        batch_size = labels.size()[0]
        
        zero = torch.Tensor([0])
        beliefs_margin = (beliefs - 0.5) * 2
        labels_sign = (labels - 0.5) * 2   # turn 0, 1 to -1, 1
        
        loss = torch.max(zero, self._margin - torch.mul(beliefs_margin, labels_sign))
        loss = torch.sum(loss)
        return torch.div(loss, batch_size)

