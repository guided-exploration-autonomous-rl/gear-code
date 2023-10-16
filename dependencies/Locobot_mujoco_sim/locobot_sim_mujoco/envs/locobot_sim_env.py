import gym
import numpy as np
import pybullet as p
from locobot_sim_mujoco.resources.env import Environment
import time
from time import sleep
import yaml
import math
from locobot_sim_mujoco.resources.controllers import LoCoBot
from pathlib import Path

from gym.spaces import Discrete, Dict, box

class LocobotSimMujocoEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):
        self.action_space = Discrete(4)

        self.obs_space = box.Box(
            low=np.array([-3, -2, -1], dtype=np.float32),
            high=np.array([3, 2, 1], dtype=np.float32))
        
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.obs_space),
            ('achieved_goal', self.obs_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.obs_space),
            ('state_achieved_goal', self.obs_space),
        ])

        self.threshold = 0.25
        self.goals = np.array([[-2.25, 1.25, 0.5], [2, -1, 0]])
        self.id = 0
        self.goal = self.goals[self.id]
        self.goal_images = [None for _ in range(len(self.goals))]
        self.print_stuff = False

    def initialize_env(self, env_name, print_stuff = False):
        self.env = Environment(env_name)
        self.locobot = LoCoBot(self.env.sim, self.env.viewer, self.env.model)
        self.print_stuff = print_stuff
        self.start = self.get_state()

    def step(self, action):
        self.locobot.do_action(action)

        state = self._get_obs()
        done = False
        reward = 0
        if self.print_stuff:
            print("after action", action, "state is", self.get_state())   

        return state, reward, done, dict()

    def reset(self):
        self.sample_goal()
        self.env.reset()
        if self.print_stuff:
            print("current state after reset", self.get_state())    
        return self._get_obs()

    def _get_obs(self):
        state_obs = self.get_state()
        achieved_state_goal = state_obs.copy()
        intended_state_goal = self.goal.copy()

        obs = state_obs.copy()
    
        achieved_goal = achieved_state_goal.copy()
        intended_goal = intended_state_goal.copy()
            
        return dict(
            observation = obs,
            desired_goal = intended_goal,
            achieved_goal = achieved_goal,
            state_observation = state_obs,
            state_desired_goal = intended_state_goal,
            state_achieved_goal = achieved_state_goal,
        )

    def get_state(self):
        return self.locobot.get_base_pos_and_yaw()

    def render(self, mode = "rgb_array", width = 640, height = 480, camera_id = 0):
        return self.env.get_image(w = width, h = height)
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def render_image(self):
        return self.render()
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def observation(self, state):
        return state
    
    def extract_goal(self, state):
        return state
    
    def get_shaped_distance(self, state1, state2):
        return self.env.shaped_reward(state1[:2], state2[:2])
    
    def compute_shaped_distance(self, state1, state2):
        return self.get_shaped_distance(state1, state2)
    
    def sample_goal(self):
        """
        Returns a goal as [x, y, theta]
        """

        goal = self.goals[self.id]

        if self.print_stuff:
            print("sampling goal", self.id, goal, self.get_state())

        if self.compute_success(goal, self.get_state()):
            if self.print_stuff:
                print("success!")
            self.id = (self.id + 1) % 2
            goal = self.goals[self.id]

        goal_state = np.r_[goal, goal, goal]
        return goal_state

        
    def plot_trajectories(self):
        return

    def compute_success(self, state, goal):
        return self.get_shaped_distance(state, goal) < self.threshold

    def generate_goal_image(self):
        if self.goal_images[self.id] is None:
            saved_state = self.env.sim.get_state()

            real_pos = self.goal - self.start
            self.env.sim.data.qpos[:3] = [real_pos[1], -real_pos[0], real_pos[2]]
            self.env.sim.step()

            self.goal_images[self.id] = self.render_image()

            self.env.sim.set_state(saved_state)
            self.env.sim.step()

        return self.goal_images[self.id]
    
    def generate_goal_image_anywhere(self, goal):
        saved_state = self.env.sim.get_state()

        real_pos = goal - self.start
        self.env.sim.data.qpos[:3] = [real_pos[1], -real_pos[0], real_pos[2]]
        self.env.sim.step()
        _ = self.render_image()
        final_image = self.render_image()

        self.env.sim.set_state(saved_state)
        self.env.sim.step()
        _ = self.render_image()

        return final_image