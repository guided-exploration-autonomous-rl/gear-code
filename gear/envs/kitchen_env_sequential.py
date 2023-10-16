from lexa_benchmark.envs.kitchen import KitchenEnv
from collections import OrderedDict
import numpy as np
from gym.spaces import Discrete
from gym.spaces import Box, Dict
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)

from multiworld.envs.mujoco.mujoco_env import MujocoEnv
import copy

from multiworld.core.multitask_env import MultitaskEnv
import matplotlib.pyplot as plt
import os.path as osp
from gear.envs.gymenv_wrapper import GymGoalEnvWrapper
import numpy as np
import gym
import random
import itertools
from itertools import combinations
from envs.base_envs import BenchEnv
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightTopLeftBurnerV0
from gym import spaces
import torch

# Microwave joint range -2.094 0
# each finger joint range 0 0.04

MICROWAVE_OPEN = -0.35
MICROWAVE_CLOSED = 0

MICROWAVE_INITIAL_POS = [-0.63, 0.48, 1.8]
MICROWAVE_FINAL_POS = [-0.7, 0.38, 1.8]
MICROWAVE_ID = 0

INITIAL_STATE = np.array([
        7.11202298e-28, # microwave joint
        -3.14732322e-01,  3.09924310e-01,  2.04698437e+00]) # end-effector position x,y,z

# STARTING_JOINTS = [-0.030597951791111667, -1.4137967314454751, 2.0093666799307313, -2.2544317666930995, 0.05576320369454787, -0.09244765624259187, 2.010286706783047, 0.0002909448464700349, 0.00031835185315913625, -7.728535947456626e-05, -2.897208580979492e-07, 2.4595483039113212e-05, 2.957091012695921e-07, 2.4515928145606675e-05, 2.951715796076804e-07, 2.4682903588541757e-05, 2.962997773996938e-07, 2.1698423191214508e-05, 5.086309841613842e-06, -5.7000984837788824e-08, -0.00644129196, -1.6033586629436786e-07, 1.1384165547281068e-07, -0.2694578556192581, 0.3503701024075295, 1.6193838355559136, 0.9999999573641195, -6.103450547780426e-06, -5.340265392646867e-06, -0.00029190064878929897]
STARTING_JOINTS = [-1.35399023081732, -1.4370638785513594, 1.882077131797929, -2.682838254616274, -3.1288633556278755, -1.2330570852261866, 4.260094106186138, 0.0001288965016893593, 0.00015007870161729337, 2.4577815481891194e-05, 2.9559011722465936e-07, 2.4577741534602238e-05, 2.9558922801529394e-07, 2.4577741534601957e-05, 2.955892280151464e-07, 2.457774153460208e-05, 2.9558922801514433e-07, 2.1619625424384463e-05, 5.0807366986406985e-06, -1.7670858926922135e-28, -0.00644129196, -1.2077179144434306e-27, 7.1120229821461725e-28, -0.2695202013343078, 0.3504755104240286, 1.6193821185474209, 0.99999995322274, 0.00011206136879918574, -1.5645585061705534e-05, -0.0002841689341619422]
STARTING_QUAT = np.array([-6.81157955e-01,  9.02937511e-02,  6.37162830e-01,  -3.49133795e-01])


class KitchenIntermediateEnv(BenchEnv):
  def __init__(self, task_config=['microwave'], action_repeat=1, use_goal_idx=False, log_per_goal=False,  control_mode='end_effector', width=64):

    super().__init__(action_repeat, width)
    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    self.goal_index = 0
    self.last_good_checkpoint = None

    self.init_goals()
    self.num_fails = 0
    self.total_resets = 0

    with self.LOCK:
      self._env =  KitchenMicrowaveKettleLightTopLeftBurnerV0(frame_skip=16, control_mode = control_mode, imwidth=width, imheight=width)

      self._env.sim_robot.renderer._camera_settings = dict(
        distance=3, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)

      obs_upper = np.array([0.01])
      obs_lower = np.array([-2.1])

      # quaterion_low = -np.ones(4)
      # quaterion_high = np.ones(4)
  
      # TODO: could be better
      obs_upper_pose = 3 * np.ones(3)
      obs_lower_pose = -obs_upper_pose
      self._observation_space = spaces.Box(np.concatenate([obs_lower, obs_lower_pose]),np.concatenate([obs_upper, obs_upper_pose]), dtype=np.float32)
      self._goal_space = spaces.Box(np.concatenate([obs_lower, obs_lower_pose]),np.concatenate([obs_upper, obs_upper_pose]), dtype=np.float32)
      
      self.base_movement_actions = [[1,0,0,0,0,0,0],
                                    [-1,0,0,0,0,0,0],
                                    [0,1,0,0,0,0,0],
                                    [0,-1,0,0,0,0,0],
                                    [0,0,1,0,0,0,0],
                                    [0,0,-1,0,0,0,0]
                                    ]
    
      self.base_rotation_actions = [[0,0,0,1,0,0,0],
                                    [0,0,0,-1,0,0,0],
                                    [0,0,0,0,1,0,0],
                                    [0,0,0,0,-1,0,0],
                                    [0,0,0,0,0,1,0],
                                    [0,0,0,0,0,-1,0]
                                    ]
      self.gripper_actions = [[0,0,0,0,0,0,1],[0,0,0,0,0,0,-1]]
      
      print("observation space in kitchen", self._observation_space)
   
    initial_obs = self.reset()

    print("initial obs", initial_obs)

  def compute_success(self, achieved_state, goal):  
      # print("compute success!!", achieved_state, goal)
      return abs(achieved_state[MICROWAVE_ID] - goal[MICROWAVE_ID]) < 0.1 and np.linalg.norm(achieved_state[-3:] - goal[-3:]) < 0.25

  def init_goals(self):
    goal_0 = INITIAL_STATE.copy() 
    goal_1 = INITIAL_STATE.copy() 

    goal_0[MICROWAVE_ID] = MICROWAVE_OPEN
    goal_0[-3:] = MICROWAVE_FINAL_POS
    
    self.goals = np.array([goal_0, goal_1])
    self.goal = self.goals[0]

  def generate_goal(self):
    if self.compute_success(self._get_obs()["observation"], self.goals[self.goal_index]):
       self.goal_index = (self.goal_index + 1) % 2

    self.goal = self.goals[self.goal_index]
    return self.goal

  def internal_extract_state(self, obs):
      #gripper_pos = obs[7:9]
      microwave_joint = [obs[22]]
      return np.concatenate([microwave_joint])

  def render_image(self):
    return self._env.render(mode="rgb_array")

  def render(self, mode='rgb_array', width=480, height=64, camera_id=0):
      return self._env.render(mode=mode)
   
  @property
  def state_space(self):
    #shape = self._size + (p.linalg.norm(state - goal) < self.goal_threshold
    #shape = self._size + (3,)
    #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    #return gym.spaces.Dict({'image': space})
    return self._goal_space
  @property
  def action_space(self):
     return Discrete(7)
    # return self._env.action_space

  @property
  def goal_space(self):
    return self._env.goal_space
  
  @property
  def observation_space(self):
    #shape = self._size + (3,)
    #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    #return gym.spaces.Dict({'image': space})

    observation_space = Dict([
            ('observation', self.state_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.state_space),
            ('state_observation', self.state_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.state_space),
        ])
    return observation_space

  def _get_obs(self, ):
    #image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    #obs = {'image': image, 'state': state, 'image_goal': self.render_goal(), 'goal': self.goal}'
    world_obs = self.internal_extract_state(self._env._get_obs())
    ee_obs = self._env.get_ee_pose()
    obs = np.concatenate([world_obs, ee_obs])
    goal = self.goal #self._env.goal

    return dict(
            observation=obs,
            desired_goal=goal,
            achieved_goal=obs,
            state_observation=obs,
            state_desired_goal=goal,
            state_achieved_goal=obs
    )
    
  def orientation_diff(self):
    ee_quat = self._env.get_ee_quat()
    dist = np.linalg.norm(STARTING_QUAT - np.array(ee_quat))
    return dist

  def step(self, action):
    print("#RESETS TO LAST DECENT POSITION: ", self.total_resets)
    action = int(action)
    total_reward = 0.0
    
    current_state = self._env.sim.get_state()
    current_dist = self.orientation_diff()
    
    if current_dist < 0.1:
      self.last_good_checkpoint = current_state

    if action < 6:
       cont_action = self.base_movement_actions[action]
    else:
       cont_action = np.zeros(7)

    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(cont_action)
      reward = 0 #self.compute_reward()
      total_reward += reward
      if done:
        break
      
    new_dist = self.orientation_diff()
    
    if new_dist - current_dist > 0.025:
      self._env.sim.set_state(current_state)
    elif new_dist > 0.25:
      self._env.sim.set_state(current_state)
      
    if new_dist > 0.2:
      self.num_fails += 1
    else:
      self.num_fails = 0
      
    if self.num_fails >= 100:
      self._env.sim.set_state(self.last_good_checkpoint)
      self.num_fails = 0
      self.total_resets += 1
      
    obs = self._get_obs()
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v
        
    return obs, total_reward, done, info

  def reset(self):
    with self.LOCK:
      state = self._env.reset()
    self.goal = self.generate_goal()#self.goals[self.goal_idx]
    
    self._env.sim.data.qpos[:] = STARTING_JOINTS 
    self._env.sim.forward()
    
    return self._get_obs()

class KitchenSequentialGoalEnv(GymGoalEnvWrapper):
    def __init__(self, task_config='slide_cabinet,microwave', fixed_start=True, fixed_goal=False, images=False, image_kwargs=None):
        self.task_config = task_config.split(",")
        self.env = KitchenIntermediateEnv(task_config=self.task_config)
        self.fake_env = KitchenIntermediateEnv(task_config=self.task_config)
        self.microwave_handle_id = self.env._env.sim.model.geom_name2id("microwave_handle")
        self.microwave_joint = 22

        super(KitchenSequentialGoalEnv, self).__init__(
            self.env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )

        self.action_low = np.array([0.25, -0.5])
        self.action_high = np.array([0.75, 0.5])

    def compute_success(self, achieved_state, goal):
      return self.env.compute_success(achieved_state, goal)

    def distance_to_handle(self, state):
      pos_state = state[-3:]
      microwave_joint = state[MICROWAVE_ID]
      
      self.fake_env._env.sim.data.qpos[22] = microwave_joint

      self.fake_env._env.sim.forward()
      handle_pos = self.fake_env._env.sim.data.geom_xpos[self.microwave_handle_id]
      
      return np.linalg.norm(pos_state - handle_pos)
      
    def nearby_goal(self, goal_idx=0, eps=0.05):
        goal = self.base_env.goals[goal_idx]
        
        offset = np.random.random(goal.shape)#np.array([np.random.random(), np.random.random()])
        new_goal = goal + offset*eps*2 - eps

        return new_goal
    
    # The task is to open/close the microwave
    def compute_shaped_distance(self, state, goal):
        success = self.compute_success(state, goal)
        bonus = 10
        
        microwave_state = state[MICROWAVE_ID]
        microwave_goal = goal[MICROWAVE_ID]
        joint_difference = np.abs(microwave_state - microwave_goal)
        
        pos_state = state[-3:]
        pos_goal = goal[-3:]
        distance_to_key_pos = np.linalg.norm(pos_state - pos_goal)
        
        if success:
          final_distance = distance_to_key_pos + joint_difference
        else:
          final_distance = bonus + self.distance_to_handle(state)
        
        return final_distance
        
    def get_shaped_distance(self, states, goal_states):
        return self.compute_shaped_distance(states, goal_states)

    def render_image(self):
      return self.base_env.render_image()

    def get_obs(self):
      obs = self.env._get_obs()["observation"]

      return np.concatenate([obs, obs, obs])
    
    def get_diagnostics(self, trajectories, desired_goal_states):
        return OrderedDict()
    
    def sample_goal(self):
      goal = self.env.generate_goal()
      return np.concatenate([goal, goal, goal])