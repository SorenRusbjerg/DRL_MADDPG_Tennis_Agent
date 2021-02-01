# Initial agent
In this project a MADDPG agent is implemented. The agent consist of two DDPG agents with critics that share experiences, states and actions from both models. 

The algorithm is based on the [MADDPG paper](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf), and the implementation have borrowed code from the Udacity "MADDPG - Lab" exercise. 

# Final implementation
## MADDPG Code layout
The code consist of 
- `train.py`: Main file for training agent using an outer training loop and running the tennis environment.  
- `maddpg.py`: Implements the MADDPG agents. Set up the the two tennis DDPG agents network sizes. Runs the training step function, and implements an action function to get agent actions. Also contains the shared replay buffer.
- `ddpg.py`: Contains the implementation of a single DDPG actor-critic agent, containing both active and target networks. It also contains the OU-noise model used to provide exploration for the agents.
- `networkforall.py`: Contains the actual Torch NN-networks used by the DDPG agents.         
- `utilities.py`: Contains helper functions, primarily to convert the actions and states to the right format used by the MADDPG agents and the tennis environment. 

## MADDPG Actor-Critic network
The DDPG agent Actor-critic network is set up as follows: 
* First Layer is a batch normalization layer to normalize input signals variance
* Both Actor and Critic networks have 2 hidden Linear layers of 256 and 128 neurons with Leaky-Relu activation 
* Output layer of Actor has a tanh-activation function to make the output range [-1,1]
* Critic does not use activation on the output layer
* Gradient clipping is used in the critic network to prevent exploding gradients. (Note: implemented in `maddpg.py`)

## Agent parameters
Below the parameters used for training the final model:
```python
BUFFER_SIZE         = 1e5       # replay buffer size
BATCH_SIZE          = 128       # minibatch size
STEPS_PER_UPDATE    = 2         # Nr of steps before agent update
LR_ACTOR            = 0.5e-4    # learning rate of the actor
LR_CRITIC           = 0.5e-3    # learning rate of the critic
TAU                 = 0.001     # for soft update of target parameters
DISCOUNT_FACTOR     = 0.99      # or Gamma
ACTOR_FC1           = 256       # 1 actor layer neurons
ACTOR_FC2           = 128       # 2 actor layer neurons
CRITIC_FC1          = 256       # 1 critic layer neurons
CRITIC_FC2          = 128       # 2 critic layer neurons
```

The DDPG uses noise from a Ornstein-Uhlenbeck process added to the action outputs to add exploration to the agent. The gaussion input noise to this process is decreased over episodes until a minimum level of 5% initial value.

The agent parameters are updated from a `Replay buffer` 1 time every 2 samples using batches of 128 samples.


## Agent description
Two actor network is used to estimate action values for the two agents together with two critic network that uses both the action values as input together with the state values from both agents combined to estimate the Q-function (action-value function). 
This model should make the environment seen from the individual DDPG agent become stationary, as each agent gets to know the other agents actions, in contrast to having decoupled training of two seperate DDPG agents where each agent changes over time, ie. a non-stationary environment.   
The Actor and critic networks is divided into a target and a local network. The local networks parameters are trained using gradient steps with their respective loss functions. The target networks are used to estimate the next Q-value when training the critic network. 
The target networks are slowly updated towards the local networks using a soft-update function to get the target network parameters. 

All the agent state-action-reward-nextstate are saved into a replay buffer during training  and sampled randomly in the agent update step.

The MADDPG agent uses an actor loss function of:

**Loss = -mean(Qi(state,PI(state)))**

to train for highest reward given a specific state.

and a critic loss function of:

**Loss = MSE(Qi(state,actions) - y)**

**where:
 y=reward + gamma*Qi_target(next-state, next-actions)**

**where:
 next-actions = PI_target(next-state)**

**PI=Actor-policy**

to get better Q-estimates. 

Note: MSE loss is replaced by Huber-loss in the code, which could lead to better stability. 

In this version of MADDPG, the actor models gets the combined states from both agents, to get an improved knowledge of the environment. 
The critic model, in contrast to the paper description, recieves its action inputs from the two agents into the second layer and the states into the first layer. This should give a better and more stable training, suggested by several people.  

## Agent results

In the end the agent got an average score of 0.53 after 2900 episodes and 1.13 after 3700 episodes of tennis games.
A training example is given in `./Tennis_training.html`.



## Improvement steps done during project 
Below is a list of updates and steps made gradually to produce the final agent implementation:
* Include Batch normalization layers (don't know if this helped)
* only updating agent after N steps (2 was found to give better results than higher numbers)
* Test with high process noise (improved training performance when using an amplitude of 2 or more)
* Test with low process noise (OU-noise under two decereased training performance)
* Include Gradient clipping on Critic (don't know if this helped)
* Implement Reducing exploration, by gradually decreasing process noise
* Test with bigger Network size (did not seem to improve performance)
* Test with smaller Network size
* Test with different Learning rates (the final values seemed to give the best training)
* Test using Leaky relus (don't know if this helped)

It was very difficult to get the agent training initially, and only scores of 0.03 in average over 2000 episodes were seen. Therefore a lot of changes in the hyper parameters and the model layout was done to agent score improving. 
In the end the reduced learning rate together with a smaller network size and running fever steps per agent update seemed to make the agent train and gets average scores above 0.5. Moving the action input to the critic to the 2nd layer seemed to help. 
The OU-noise ampltiude also had a part in getting the model to train. 

It was not possible to get agent started learning in much less than 3000 episodes, which was a litlle surpricing, as it seems others have been able to do it in half the training amount.    

## Future improvements

Future improvements to help getting the model to learn faster is using a priority replay buffer as also done in DDPG. 
I am quite sure the hyper parameters could be improved also, to get faster learning. The soft update TAU value I didn't change, and also different GAMMA value I didn't get to test either. 

An improvement idea is to remove redundant state information from the two agents, so that the state input size decreases.

Also split the actor input, so only the local state is used by each actor, to have a more decoupled model. 

I also didn't spend much time in tuning the parameters for the OU-noise memory, which might be able to give improved exploration for the model, making it more diverce. It seems to repeat the same basic patterns in the trained model at the moment.











