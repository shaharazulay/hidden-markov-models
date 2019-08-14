import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def belief_propagation_cross_entropy_loss(j, b, labels):
    batch_size = labels.size()[0]
    len_ = b.size().numel() # length of chain
    
    values = torch.Tensor([0, 1]) - 0.5
    pairs = torch.mul(values.view(-1, 2).t(), values)
    unit_msg = torch.ones((2, 1))
        
    forward_messages = unit_msg
    msg_left = unit_msg # no information traveling left to x1
    # forward pass    
    for i in range(0, len_ - 1):
        
        phi = torch.exp(torch.mul(b[i], values))
        psi = torch.exp(torch.mul(j[i], pairs))
        step1 = torch.mul(psi, msg_left)
        step2 = torch.mul(step1, psi)
        step3, _ = torch.max(step2, dim=1)
        
        msg = step3.view(-1 ,1)
        forward_messages = torch.cat((forward_messages, msg), dim=1)    
        msg_left = msg

    backward_messages = unit_msg
    msg_right = unit_msg # no information traveling right to x_n
    # backward pass    
    for i in range(len_ - 1, 0, -1):
        
        phi = torch.exp(torch.mul(b[i], values))
        psi = torch.exp(torch.mul(j[i - 1], pairs))
        step1 = torch.mul(psi, msg_right)
        step2 = torch.mul(step1, psi)
        step3, _ = torch.max(step2, dim=1)
        
        msg = step3.view(-1 ,1)
        backward_messages = torch.cat((backward_messages, msg), dim=1)    
        msg_right = msg
        
    # calculate beliefs
    messages = torch.mul(forward_messages, backward_messages)
    data_term = torch.exp(torch.mul(b, values.view(-1, 1)))
    beliefs = torch.mul(data_term, messages)
    beliefs_norm = torch.softmax(beliefs, dim=0)
    
    loss = F.binary_cross_entropy(beliefs_norm[1, :], labels)
    return loss
    