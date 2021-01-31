# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, buffer_state_list_to_tensor, buffer_list_to_tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
from buffer import ReplayBuffer
from tensorboardX import SummaryWriter
import os

BUFFER_SIZE         = 1e5
BATCH_SIZE          = 128
STEPS_PER_UPDATE    = 2
LR_ACTOR            = 0.5e-4
LR_CRITIC           = 0.5e-3
TAU                 = 0.001
DISCOUNT_FACTOR     = 0.99

log_path = os.getcwd()+"/log" # tensorBoard log

class MADDPG:
    def __init__(self, discount_factor=DISCOUNT_FACTOR, tau=TAU):
        super(MADDPG, self).__init__()

        self.N_actions = 2
        self.N_agents = 2
        self.all_states = 48
        # critic input = obs_full + actions = 2*24+2*2=52
        # actor input = obs_full = 2*24
        # critic out = 1 Hardcoded        
        n_acts = self.N_actions*self.N_agents
        self.maddpg_agent = [DDPGAgent(NN_pars_actor=(self.all_states, 256, 128, self.N_actions), NN_pars_critic=(self.all_states, n_acts, 256, 128), device=device, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC), 
                             DDPGAgent(NN_pars_actor=(self.all_states, 256, 128, self.N_actions), NN_pars_critic=(self.all_states, n_acts, 256, 128),device=device, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

        self.logger = SummaryWriter(log_dir=log_path)

        self.t_step = 0
       
        # Create replay buffer
        self.buffer = ReplayBuffer(int(BUFFER_SIZE))

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs_all_agents, noise) for agent in self.maddpg_agent]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs_all_agents, noise) for ddpg_agent in self.maddpg_agent]
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # unpack samples
        obs, action, reward, next_obs, done = samples

        # Convert samples to tensors
        obs = buffer_state_list_to_tensor(obs).to(device)
        next_obs = buffer_state_list_to_tensor(next_obs).to(device)
        action = buffer_list_to_tensor(action).to(device)
        reward = buffer_list_to_tensor(reward).to(device)
        done = buffer_list_to_tensor(done).to(device)
        
        # get agent from agent list
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        with torch.no_grad():
            q_next = agent.target_critic(next_obs, target_actions)
          
        y = reward[:,agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:,agent_number].view(-1, 1))
        critic_input = torch.cat((obs, action.view(-1,self.N_actions*self.N_agents)), dim=1)
        action = action.view(-1, self.N_actions*self.N_agents)
        q = agent.critic(obs, action)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        for i in range(self.N_agents):
            self.maddpg_agent[i].actor.train()
            self.maddpg_agent[i].critic.train()

        q_act_input = [ self.maddpg_agent[i].actor(obs) if i == agent_number else self.maddpg_agent[i].actor(obs).detach()
                   for i in range(self.N_agents) ]

        # combine agents actions input to critic     
        q_act_input = torch.cat(q_act_input, dim=1)
         
        # get the policy gradient
        actor_loss = -agent.critic(obs, q_act_input).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
    # do training step
    def step(self, states, actions, rewards, next_states, dones):
        self.t_step += 1

        # add data to buffer
        transition = (states, actions, rewards, next_states, dones)
        self.buffer.push(transition)        

        if len(self.buffer) > BATCH_SIZE and self.t_step % STEPS_PER_UPDATE == 0:
            # Reset OU noise 
            for ddpg_agent in self.maddpg_agent:
                ddpg_agent.noise.reset()            
                 
            for agent_i in range(self.N_agents): # loop agents
                samples = self.buffer.sample(BATCH_SIZE)
                self.update(samples, agent_i, self.logger)
            self.update_targets() #soft update the target network towards the actual networks 
            
    # Save entire model
    def save_model(self, filename, episode):
        #saving model
        model_dir= os.getcwd()+"/model_dir"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        save_dict_list =[]
        for i in range(self.N_agents):
            save_dict = {'actor_params' : self.maddpg_agent[i].actor.state_dict(),
                            'actor_optim_params': self.maddpg_agent[i].actor_optimizer.state_dict(),
                            'critic_params' : self.maddpg_agent[i].critic.state_dict(),
                            'critic_optim_params' : self.maddpg_agent[i].critic_optimizer.state_dict()}
            save_dict_list.append(save_dict)
            torch.save(save_dict_list, os.path.join(model_dir, filename + '-episode-{}.pt'.format(episode)))

    # load saved model
    def load_model(self, filename):
        torch.load(filename)