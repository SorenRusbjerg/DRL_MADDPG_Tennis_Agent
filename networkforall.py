import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False, action_size=0):
        super(Network, self).__init__()

        self.input_norm = nn.BatchNorm1d(input_dim, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)

        if actor: 
            self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        else: # critic uses output from fc1 + action input as input for fc2
            self.fc1 = nn.Linear(input_dim,hidden_in_dim-action_size)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        self.nonlin = f.leaky_relu
        self.actor = actor
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x, x_action_critic=0):
        if self.actor:
            # return a vector of the force
            bn = self.input_norm(x) 
            h1 = self.nonlin(self.fc1(bn))
            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))            
            return f.tanh(h3)
        
        else:
            # critic network simply outputs a number
            bn = self.input_norm(x) 
            h1 = self.nonlin(self.fc1(bn))
            h1_cat = torch.cat((h1, x_action_critic), dim=1)
            h2 = self.nonlin(self.fc2(h1_cat))
            h3 = (self.fc3(h2))
            return h3