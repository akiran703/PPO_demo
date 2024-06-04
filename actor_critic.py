import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical



class actornetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,fc1_dims = 256,fc2_dims=256):
        
        super(actornetwork,self).__init__()
        
        
        self.actor = nn.Sequential(
            nn.Linear(*input_dims,fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,n_actions),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'CPU')
        self.to(self.device)
    
    #calculate a series of probablities to draw a distribution to get a actual action, then use the action to get log prob for the ratio to update 
    def forward(self,state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

class criticnetwork(nn.Module):
    def __init__(self, input_dims,alpha, fc1_dims = 256, fc2_dims=256):
        super(criticnetwork,self).__init__
        
        self.critic = nn.Sequential(
            nn.Linear(*input_dims,fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,1)
        )
        
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'CPU')
        self.to(self.device)
        
    def forward(self,state):
        value = self.critic(state)
        return value
        
        
        