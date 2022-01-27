import math
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from statistics import mean


class MechanismDesigner(gym.Env):
  
  def __init__(self, n, m):
    super(MechanismDesigner, self).__init__()
    self.action_space = spaces.Box(low=np.array([-10]),
                                   high=np.array([10]), 
                                   shape=(1,), 
                                   dtype=np.float32)
    self.observation_space = spaces.Box(low=np.array([0,0]),
                                       high=np.array([1,1]),
                                       shape=(2,),
                                       dtype=np.float32)     
    self.n = n
    self.m = m
    self.episode_threshold = 10
    self.step_count = 0
    self.state = 0
    self.inner_layer_total_time_steps = 10000


  def step(self, action):
    self.step_count = self.step_count + 1
    self.state = 0
    reward = self.reward(action)
    done = bool(self.step_count > self.episode_threshold)    

    return np.array([self.state]), reward, done, {}



  def reward(self,action):
    reward = 0
    if action[0] > 0 :
      welfare,variance = self.simulate_inner_layer(action[0])
      if (welfare > 0.5) & (variance < 0.5) :
        reward = 50
      elif (welfare > 0.5) & (variance > 0.5) :
        reward = 15
      elif (welfare < 0.5) & (variance < 0.5) :
        reward = 15
      else 
        reward = 0
    return reward


  
  def simulate_inner_layer(self,mechanism_parameter):
    inner_layer_env = MultiAgentEnv.create_environment(self.n,self.m,mechanism_parameter)
    welfare = maddpg.train(inner_layer_env, time_step = self.inner_layer_total_time_steps)
    return welfare, variance

  

  def reset(self, seed: Optional[int] = None):
      self.step_count=0
      if seed is not None or self.np_random is None:
          self.np_random, seed = seeding.np_random(seed)
      self.state = 0
      return np.array([self.state], dtype=np.float32)


  def render(self, mode='human', close=False):
      # TODO : PettingZOO MPE Render
      pass


  def close(self):
      pass

    
def simulate_inner_layer(mechanism_parameter,n,m):
  from multiagent.scenrios import inner_layer
  inner_layer_env = inner_layer.create_environment(n,m,mechanism_parameter)
  welfare,variance = maddpg.train(inner_layer_env, time_step = 100000)
  return welfare ,variance
