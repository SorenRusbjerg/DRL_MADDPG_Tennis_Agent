from maddpg import MADDPG
import torch
import numpy as np
import os
from utilities import *
import time
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment


def train_agents(agents, env, brain_name, n_episodes, target_score=0.5):
    # Local train parameters
   
    # amplitude of OU noise
    # this slowly decreases to 0
    noiseAmp = 2.5
    noise_reduction = 0.9997
    noise_lim = noiseAmp*0.05
    print_steps = 100

    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment   
    # number of agents 
    num_agents = len(env_info.agents)
    # initialize the score (for each agent)
    scores = []
    start = time.time()
    for i_episode in range(1, n_episodes):                                      # play game for n episodes
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations                 # get the current state (for each agent)
        score = np.zeros(num_agents)                          # initialize the score (for each agent)

        while True:
            states_T = states_to_tensor(states)
            actions = agents.act(states_T, noise=noiseAmp)
            actions = action_to_environment(actions)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = np.asarray(env_info.rewards, dtype=np.float32)   # get reward (for each agent)
            dones = np.asarray(env_info.local_done, dtype=np.float32)  # see if episode finished
            agents.step(states, actions, rewards, next_states, dones)
            score += env_info.rewards                          # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):
                break        

        # decrease noise level
        if noiseAmp > noise_lim:
            noiseAmp *= noise_reduction   

        scores.append(np.max(score))
        print('\rEpisode {}\tScore: {:.3f} \tNoise ampl.: {:.3f}'.format(i_episode, np.max(score), noiseAmp), end="")
            
        if i_episode % print_steps == 0:
            end = time.time()    
            mean_score = np.mean(scores[-print_steps:])
            print('\rEpisode {}\tAverage Score: {:.3f} \tTime/episode: {:.2f}'.format(i_episode, mean_score, (end-start)/print_steps, end=""))
            start = end
            if mean_score >target_score:
                name = "MADDPG_V1_Score_{:.2f}".format(mean_score)
                agents.save_model(name, i_episode)
                target_score = mean_score      
                print('Agent saved! file="MADDPG_V1_Score_{:.2f}"'.format(mean_score))

    return scores

if __name__ == "__main__":  

    # get environment
    env = UnityEnvironment(file_name="p3_collab-compet/Tennis_Linux/Tennis.x86_64" )

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

       # train parameters
    n_episodes = 3500
    save_score = 0.5

    # create agent
    maddpg = MADDPG()

    load = False    # Load saved model
    load_filename = "p3_collab-compet/model_dir/MADDPG_V1_Score_0.98-episode-3400.pt"
    # Load model
    if load:
        maddpg.load_model(load_filename)

    scores = train_agents(maddpg, env, brain_name, n_episodes, target_score=save_score)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


    
