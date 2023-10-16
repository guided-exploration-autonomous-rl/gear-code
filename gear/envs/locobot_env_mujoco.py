import gym
import numpy as np
from gear.envs.gymenv_wrapper import GymGoalEnvWrapper
import locobot_sim_mujoco
from collections import OrderedDict
from multiworld.envs.env_util import create_stats_ordered_dict

class LoCoBotEnvMujoco(GymGoalEnvWrapper):
    def __init__(self, env_name, print_stuff=False):
        env = gym.make("LocobotSimMujoco-v0")
        self.base_env = env
        super(LoCoBotEnvMujoco, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )
        self.base_env.initialize_env(env_name, print_stuff=print_stuff)

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
    
    def compute_success(self, achieved_state, goal):        
        return self.base_env.compute_success(achieved_state, goal)

    def sample_goal(self, current_state = None, id = 0):
        return self.base_env.sample_goal()

    def generate_goal_image(self):
        return self.base_env.generate_goal_image()
    
    def get_obs(self):
        obs = self.base_env.get_state()
        return np.concatenate([obs, obs, obs])