"""
A GoalEnv which wraps my room_world environments

Observation Space (2 dim): Position 
Goal Space (2 dim): Position
Action Space (2 dim): Position Control
"""

import gym
import locobot_sim 
import numpy as np
from gear.envs.gymenv_wrapper import GymGoalEnvWrapper

from collections import OrderedDict
from multiworld.envs.env_util import create_stats_ordered_dict

class LoCoBotEnv(GymGoalEnvWrapper):
    def __init__(self):
        env = gym.make("LocobotSim-v0")
        self.base_env = env
        super(LoCoBotEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )

    def render_image(self):
        return self.base_env.render()

    def shaped_distance(self, states, goal_states):
        achieved_goals = states
        desired_goals = goal_states

        return np.array([
            self.base_env.compute_shaped_distance(achieved_goals[i], desired_goals[i])
            for i in range(achieved_goals.shape[0])
        ])
    
    def compute_shaped_distance(self, states, goal_states):
        return self.shaped_distance(np.array([states]), np.array([goal_states]))
    
    def get_shaped_distance(self, states, goal_states):
        return self.compute_shaped_distance(states, goal_states)
    
    def get_diagnostics(self, trajectories, desired_goal_states):
        return

