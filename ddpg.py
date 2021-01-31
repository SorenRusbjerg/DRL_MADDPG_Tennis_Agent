# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np



# add OU noise for exploration
from OUNoise import OUNoise

class DDPGAgent:
    def __init__(self, NN_pars_actor, NN_pars_critic, device, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0):
        super(DDPGAgent, self).__init__()

        self.device = device
        in_actor, hidden_in_actor, hidden_out_actor, out_actor = NN_pars_actor
        in_critic, in_agents_action, hidden_in_critic, hidden_out_critic = NN_pars_critic
        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, actor=False, action_size=in_agents_action).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, actor=False, action_size=in_agents_action).to(device)

        self.noise = OUNoise(out_actor, scale=1.0 )

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)


    def act(self, obs, noise=0.0, train=False):
        if train:
            self.actor.train()
        else:
            self.actor.eval()
        action = self.actor(obs.to(self.device)) + noise*self.noise.noise().to(self.device)
        return action

    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs.to(self.device)) + noise*self.noise.noise().to(self.device)
        return action
