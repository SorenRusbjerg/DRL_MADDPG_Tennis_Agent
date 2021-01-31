# Task List

* Find env shapes
    * State shape = (agent, states) = np(2, 24) 
    * action shape = (agent, actions) = np(2, 2) 
    * reward shape = (rewards) = list(2,)
* MADDPG shapes
    * obs_full State shape = (envs,state) =  list(arrays(state)) 
    * obs State shape = (envs,agents,state) = list(list(arrays(state))) 
    * obs to act input = (agent,env,state) = list(tensor(env,state))
     new = (agent,batch,state) = list(tensor(batch,state))
    * action shape = (agent,env,action) = list(tensor(env,actions))
    * reward shape = 
* ~~Update NN layers in Maddpg~~
* ~~Cleanout parallel env in maddpg file~~
* ~~Setup training environemnt in tennis notebook~~
* ~~Setup buffer to input relevant states~~
* ~~Create input for mddpg.act: (agent,1,state) = list(tensor(1,obs_full)) = [tensor(1,obs_full), tensor(1,obs_full)]
Create function to map (agent, states) = np(2, 24)  -> [tensor(1,obs_full), tensor(1,obs_full)]~~
* ~~Fix MADDPG.update function~~
* ~~Insert action into critic 2nd layer~~
* ~~Insert batchnorm layer~~ No effect seen
* ~~Leaky relu~~ No effect seen
* ~~Remove output normaliztion~~ No effect seen
* Appply parameters from https://github.com/rhemon/p3_CollaborationCompetition/blob/master/Report.ipynb or https://knowledge.udacity.com/questions/315134
* Insert hyper parameters IN MADDPG
* Cleanup jup.nb 
* jup. nb move code to .py files
* ~~insert saving agents~~
* insert loading agents
* ~~Fix score to use max~~
* Fix save gif images







