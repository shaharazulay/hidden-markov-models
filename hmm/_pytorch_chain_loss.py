import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def belief_propagation_cross_entropy_loss(j, b, observations, labels, chain_len, train=True):
    batch_size = labels.size()[0]
    
    loss = 0
    
    labels_out = []
    for b_idx in range(batch_size):

        values = torch.Tensor([0, 1]) - 0.5
        pairs = torch.mul(values.view(-1, 2).t(), values)
        unit_msg = torch.ones([2, 1]) / 2
        
        forward_messages = unit_msg
        msg_left = unit_msg # no information traveling left to x1
        
        # forward pass    
        for i in range(0, chain_len - 1):
            
            phi = torch.exp(torch.mul(b[b_idx][int(observations[b_idx][i])], values))
            psi = torch.exp(torch.mul(j[b_idx], pairs))
  
            step1 = torch.mul(phi, msg_left.t())
            step2 = torch.mul(step1, psi)
            step3, _ = torch.max(step2, dim=1)

            msg = step3.view(-1 ,1)
            norm_ = torch.norm(msg, p=1, dim=0)  # L1 norm
            msg = torch.div(msg, norm_)
            
            forward_messages = torch.cat((forward_messages, msg), dim=1)    
            msg_left = msg
        
        
        backward_messages = unit_msg
        msg_right = unit_msg # no information traveling right to x_n
        
        # backward pass    
        for i in range(chain_len - 1, 0, -1):

            phi = torch.exp(torch.mul(b[b_idx][int(observations[b_idx][i])], values))
            psi = torch.exp(torch.mul(j[b_idx], pairs))
            step1 = torch.mul(phi, msg_right.t())
            step2 = torch.mul(step1, psi)
            step3, _ = torch.max(step2, dim=1)

            msg = step3.view(-1 ,1)
            norm_ = torch.norm(msg, p=1, dim=0)  # L1 norm
            msg = torch.div(msg, norm_)
            
            backward_messages = torch.cat((msg, backward_messages), dim=1)    
            msg_right = msg

        # calculate message propagation
        messages = torch.mul(forward_messages, backward_messages)
        # add data term
        data_term = torch.Tensor()
        for i in range(0, chain_len):
            phi = torch.exp(torch.mul(b[b_idx][int(observations[b_idx][i])], values))
            data_term = torch.cat((data_term, phi.view(-1, 1)), dim=1)
        
        # calculate beliefs
        beliefs = torch.mul(data_term, messages)
        norm_ = torch.norm(beliefs, p=1, dim=0)  # L1 norm
        beliefs_norm = torch.div(beliefs, norm_)
        
        beliefs_softmax = torch.softmax(beliefs_norm, dim=0)
        loss += F.binary_cross_entropy(beliefs_softmax[1, :].float(), labels[b_idx]) ###
        
        labels_out.append(beliefs_norm[1, :] > 0.5)
        
    loss = torch.div(loss, batch_size)
    
    if train:
        return loss
    else:
        return labels_out