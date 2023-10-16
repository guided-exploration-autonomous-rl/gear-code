from fileinput import filename
from signal import default_int_handler
from re import I
from telnetlib import IP, PRAGMA_HEARTBEAT
from click import command
import numpy as np

from rlutil.logging import logger
import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu
import torch
import time
import tqdm
import os.path as osp
import copy
import pickle
import seaborn as sns
from gear.algo import buffer, networks
import matplotlib.cm as cm
import os
from datetime import datetime
import shutil
import copy 

import wandb
import skvideo.io
import random 
import torch.nn.functional as F

from math import floor

class FBRL_buffer():
    def __init__(self, env, buffer_size = 2e6):
        self.buffer_size = buffer_size
        
        self._state = np.zeros(
            (buffer_size, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        
        self._action = np.zeros(
            (buffer_size, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        
        self._reward = np.zeros(
            (buffer_size),
            dtype=np.float32
        )
        
        self._next_state = np.zeros(
            (buffer_size, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        
        self._goal = np.zeros(
            (buffer_size, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        
        self._done = np.zeros(
            (buffer_size),
            dtype=bool
        )
        
        self.pointer = 0
        self.current_buffer_size = 0
        
    def add_sample(self, state, action, reward, next_state, goal, done):
        self._state[self.pointer] = state
        self._action[self.pointer] = action
        self._reward[self.pointer] = reward
        self._next_state[self.pointer] = next_state
        self._goal[self.pointer] = goal
        self._done[self.pointer] = done

        self.current_buffer_size = max(self.current_buffer_size + 1, self.buffer_size)
        self.pointer += 1
        if self.pointer == self.buffer_size:
            self.pointer = 0
            
    def sample_batch(self, batch_size):
        indexes = np.random.choice(self.current_buffer_size, batch_size)
        
        return self._state[indexes], self._action[indexes], self._reward[indexes], self._next_state[indexes], self._goal[indexes], self._done[indexes]
        

class FBRL:
    def __init__(self,
        env,
        env_name,
        env_eval,
        policy,
        max_timesteps=1e6,
        max_path_length=50,
        explore_episodes=1e4,
        goal_threshold=0.05,
        eval_freq=5e3,
        batch_size=100,
        n_accumulations=1,
        policy_updates_per_step=1,
        lr=5e-4,
        render=False,
        logger_dump=False,
        epsilon_greedy_exploration=0.2,
        gamma = 0.99,
        buffer_size = 5000,
        train_frequency = 10,
        copy_target_frequency = 5,
        training_batch = 256,
    ):
        self.eval_env = env_eval
        self.env = env
        self.device = "cuda:0"
        print("cuda device", self.device)
        
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)
        
        self.policy.to(self.device)
        self.target_policy.to(self.device)
        
        self.total_timesteps = 0

        self.epsilon_greedy_exploration = epsilon_greedy_exploration

        self.replay_buffer = FBRL_buffer(env, buffer_size)
        self.validation_buffer = FBRL_buffer(env, buffer_size)

        self.max_timesteps = max_timesteps
        self.total_timesteps = 0
        self.max_path_length = max_path_length

        self.explore_episodes = explore_episodes
        self.render = render
        self.goal_threshold = goal_threshold
        self.eval_freq = eval_freq
        
        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.logger_dump = logger_dump

        self.env_name = env_name
        self.gamma = gamma

        self.train_frequency = train_frequency
        self.copy_target_frequency = copy_target_frequency
        self.training_batch = training_batch

    def compute_batch_loss(self, batch_size):     
        states, actions, rewards, next_states, goals, dones = self.buffer.sample_batch(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        goals = torch.FloatTensor(goals).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0))
        dones = dones.view(dones.size(0))

        # compute loss
        curr_Q = self.policy.forward(states, goals).gather(1, actions.view(actions.size(0), 1))
        next_Q = self.policy.forward(next_states)
        argmax_action = torch.argmax(next_Q)
        
        expected_next_Q = self.target_policy.foward(states, goals).gather(1, argmax_action.view(argmax_action.size(0), 1))
        expected_Q = rewards + (1 - dones) * self.gamma * expected_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())

        return loss
    
    def update_target(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
    
    def update(self):
        self.optimizer.zero_grad()
    
        loss = self.compute_batch_loss()

        loss.backward()
        self.optimizer.step()

        
        
    def get_action(self, state, greedy = True):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.policy.forward(state) # TODO fix
        
        if not greedy and np.random.randn() < self.epsilon_greedy_exploration:
            return self.env.action_space.sample()

        return np.argmax(qvals.cpu().detach().numpy())

    def sample_trajectory():
        return
    
    
    def train(self):
        total_iterations = 0
        total_training_iterations = 0
        
        while self.total_timesteps < self.max_timesteps:
            self.run_episode()
            
            if total_iterations % self.train_frequency == 0:
                self.update_policy()
                
                if total_training_iterations % self.copy_target_frequency == 0:
                    self.update_target()
    
    
                total_training_iterations += 1
    
    
    
            total_iterations += 1
    