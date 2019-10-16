import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class Chain(torch.nn.Module):
    def __init__(self, length):
        super(Chain, self).__init__()
        self._length = length

    def forward(self, j, b, observations):
        batch_size = observations.size()[0]
        loss = 0
        
        beliefs_pred = torch.Tensor()
        for b_idx in range(batch_size):

            values = torch.Tensor([0, 1]) - 0.5 
            pairs = torch.mul(values.view(-1, 2).t(), values)
            unit_msg = torch.zeros([2, 1])
            
            forward_messages = unit_msg
            msg_left = unit_msg # no information traveling left to x1
            
            # forward pass    
            for i in range(0, self._length - 1):
                
                phi = torch.mul(b[int(observations[b_idx][i])], values).view(-1, 1)
                psi = torch.mul(j, pairs)
      
                msg, _ = torch.max(phi + psi + msg_left, dim=0)
                msg = msg.view(-1 , 1)
        
                forward_messages = torch.cat((forward_messages, msg), dim=1)    
                msg_left = msg
            
            
            backward_messages = unit_msg
            msg_right = unit_msg # no information traveling right to x_n
            
            # backward pass    
            for i in range(self._length - 1, 0, -1):

                phi = torch.mul(b[int(observations[b_idx][i])], values).view(-1, 1)
                psi = torch.mul(j, pairs)
                
                msg, _ = torch.max(phi + psi + msg_right, dim=0)
                msg = msg.view(-1 , 1)
        
                backward_messages = torch.cat((msg, backward_messages), dim=1)    
                msg_right = msg

            # add data term
            data_term = torch.Tensor()
            for i in range(0, self._length):
                phi = torch.mul(b[int(observations[b_idx][i])], values)
                data_term = torch.cat((data_term, phi.view(-1, 1)), dim=1)
            
            # calculate beliefs
            beliefs = data_term + forward_messages + backward_messages            
            beliefs_pred = torch.cat((beliefs_pred, beliefs.expand(1, 2, -1)), dim=0)
        
        return beliefs_pred
    