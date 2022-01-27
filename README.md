# Two-Layer-Reinforcement-Learning


Two Layer Structure : Reinforcement Mechanism Design for Multi Agent System 


![Two Layer Model](https://i.ibb.co/hL8bGNp/two-layer.png)




## Outer Layer - Mechanism Designer :

- Implemented based on [open-ai gym environment](https://gym.openai.com/)
- Mechanism Designer chooses a mechanism parameter as action
- Step Function is a full execution of the multi agent system (aka, inner layer)
- Mechanism Designer evaluates its actions based on a predefined reward function

### Training Outer Reinforcement Learning Problem 
- Single Agent Reinforcement Learning Problem
- Recommendation : Train using [TD3 Algorithm](https://stable-baselines.readthedocs.io/en/master/modules/td3.html) from [Stable Baseline Library](https://stable-baselines.readthedocs.io/en/master/)


## Inner Layer - Multi Agent System :

- Implemented based on [petting-zoo MPE](https://www.pettingzoo.ml/mpe)
- Simple Multi Agent Scenario
   - each agent collect food & gather score during each episode
   - can become more complex by adding other elements to the game (barrier, bomb, ... )


### Training Inner Reinforcement Learning Problem 
- Multi Agent Reinforcement Learning Problem
- Two approaches for training :
   - [MADDPG](https://github.com/openai/maddpg) : Multi Agent DDPG
   - DDDPG : Distributed DDPG (or other single agent environments)
   - for comparison of these two approaches check [maddpg paper](https://arxiv.org/pdf/1706.02275.pdf)
