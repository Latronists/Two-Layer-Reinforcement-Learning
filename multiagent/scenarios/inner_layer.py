import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):


    def __init__(self):
      self.x = 0.5 # mechanism parameter
      self.alpha = self.x*(1-self.x) + 0.75  
      self.num_agents = 5
      self.num_food = 20
      self.food_score = np.ones(self.num_food)
      self.reward_threshold = np.sum(self.food_score) / self.num_agents 


    def make_world(self):
        world = World()
        # set any world properties first
        # world.dim_c = 4 #communication channel dimensionality
        num_agents = self.num_agents
        num_food = self.num_food
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = False
            agent.silent = True
            agent.adversary = False
            agent.size = 0.045
            agent.accel = 4.0
            agent.max_speed = 1.3
        # add landmarks
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False
        world.landmarks += world.food
        self.reset_world(world)
        return world

    #not called in this class    
    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color =  np.array([0.95, 0.45, 0.45])
        # random properties for landmarks
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            self.food_score[i] = 1
        

    #not called in this class    
    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    # return all agents that are not adversaries
    def agents(self, world):
        return world.agents


    def reward(self, agent, world):
        # boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.agent_reward(agent, world)
        return main_reward


    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False


    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0   
        for i,food in enumerate(world.food):
            if self.is_collision(agent, food):
                rew += self.food_score[i]
                self.food_score[i] = 0
        if rew > self.reward_threshold :
          rew *= self.alpha
        return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        '''
        comm = []
        other_pos = []
        other_vel = []
        
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_vel.append(other.state.p_vel)
            
        comm = [world.agents[0].state.c]
        '''
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos])
