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
from gear.algo import buffer, gear, networks
import matplotlib.cm as cm
import os
from datetime import datetime
import shutil

from gear import envs
from gear.algo import buffer, variants, networks
from gear.algo.ppo_new import PPO


from gear.envs.kitchen_env_sequential import KitchenSequentialGoalEnv
from gear.envs.kitchen_env_3d import Kitchen3DGoalEnv

import wandb
import skvideo.io
import random 
from PIL import Image

from math import floor

#from gcsl.envs.kitchen_env import KitchenGoalEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False

import tkinter
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from gym.spaces import Dict, Box, Discrete


class HumanPreferences:
    """Goal-conditioned Supervised Learning (GCSL).

    Parameters:
        env: A gcsl.envs.goal_env.GoalEnv
        policy: The policy to be trained (likely from gcsl.algo.networks)
        replay_buffer: The replay buffer where data will be stored
        validation_buffer: If provided, then 20% of sampled trajectories will
            be stored in this buffer, and used to compute a validation loss
        max_timesteps: int, The number of timesteps to run GCSL for.
        max_path_length: int, The length of each trajectory in timesteps

        # Exploration strategy
        
        explore_episodes: int, The number of timesteps to explore randomly
        expl_noise: float, The noise to use for standard exploration (eps-greedy)

        # Evaluation / Logging Parameters

        goal_threshold: float, The distance at which a trajectory is considered
            a success. Only used for logging, and not the algorithm.
        eval_freq: int, The policy will be evaluated every k timesteps
        eval_episodes: int, The number of episodes to collect for evaluation.
        save_every_iteration: bool, If True, policy and buffer will be saved
            for every iteration. Use only if you have a lot of space.
        log_tensorboard: bool, If True, log Tensorboard results as well

        # Policy Optimization Parameters
        
        start_policy_timesteps: int, The number of timesteps after which
            GCSL will begin updating the policy
        batch_size: int, Batch size for GCSL updates
        n_accumulations: int, If desired batch size doesn't fit, use
            this many passes. Effective batch_size is n_acc * batch_size
        policy_updates_per_step: float, Perform this many gradient updates for
            every environment step. Can be fractional.
        train_policy_freq: int, How frequently to actually do the gradient updates.
            Number of gradient updates is dictated by `policy_updates_per_step`
            but when these updates are done is controlled by train_policy_freq
        lr: float, Learning rate for Adam.
        demonstration_kwargs: Arguments specifying pretraining with demos.
            See GCSL.pretrain_demos for exact details of parameters        
    """
    def __init__(self,
        env,
        env_eval,
        model,
        reward_model,
        replay_buffer,
        reward_model_buffer,
        reward_model_buffer_validation,
        env_name,
        validation_buffer=None,
        max_timesteps=1e6,
        max_path_length=50,
        # Exploration Strategy
        explore_episodes=1e4,
        # Evaluation / Logging
        goal_threshold=0.05,
        eval_episodes=200,
        lr=5e-4,
        reward_model_epochs = 300,
        train_reward_model_freq = 10,#5000,
        display_trajectories_freq = 20,
        use_oracle=False,
        reward_model_num_samples=100,
        reward_model_batch_size = 128,
        display_plots=False,
        clip=5,
        goal=None,
        num_envs=4,
        num_steps_per_policy_step=1000,
        entropy_coefficient=0.01,
        fake_env = None,
        wandb_run=0,
        label_from_last_k_trajectories=-1,
        label_from_last_k_steps=-1,
        start_from_scratch_every_epoch=False,
        normalize_rewards=False,
        no_training=False,
        use_wrong_oracle=False,
        display_trajectories = False,
        device="cuda",
    ):
        # DDL specific
        # No frontier expansio
        self.display_trajectories = display_trajectories
        self.no_training = no_training
        self.start_from_scratch_every_epoch = start_from_scratch_every_epoch
        self.wandb_run = wandb_run
        self.entropy_coefficient = entropy_coefficient
        self.fake_env = fake_env
        self.max_path_length = max_path_length

        self.use_wrong_oracle = use_wrong_oracle

        if self.use_wrong_oracle:
            self.wrong_goal = [-0.2,0.2]

        self.normalize_rewards = normalize_rewards

        self.use_oracle = use_oracle
        self.start_frontier = max_path_length

        self.num_envs = num_envs
        self.reward_model = reward_model

        self.env_name = env_name
        self.num_steps_per_policy_step=num_steps_per_policy_step
        
        if label_from_last_k_trajectories == -1:
            self.label_from_last_k_trajectories = floor(self.num_steps_per_policy_step/self.max_path_length)+1
        else:
            self.label_from_last_k_trajectories = label_from_last_k_trajectories

        if label_from_last_k_steps == -1:
            self.label_from_last_k_steps = self.max_path_length
        else:
            self.label_from_last_k_steps = label_from_last_k_steps

        self.reward_model_backup = copy.deepcopy(reward_model)


        self. reward_model_buffer_validation = reward_model_buffer_validation

        self.env = env
        self.env_eval = env_eval
        self.model = model

        self.reward_model_batch_size = reward_model_batch_size
     
        #with open(f'human_dataset_06_10_2022_20:15:53.pickle', 'rb') as handle:
        #    self.human_data = pickle.load(handle)
        #    print(len(self.human_data))
        
        self.total_timesteps = 0

        self.previous_goal = None

        self.buffer_filename = "buffer_saved.csv"
        self.val_buffer_filename = "val_buffer_saved.csv"

        self.display_plots = display_plots
        self.lr = lr
        self.clip = clip
        self.evaluate_reward_model = True

        self.reward_model_buffer = reward_model_buffer

        self.store_model = False

        self.num_labels_queried = 0

        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer

        self.max_timesteps = max_timesteps

        self.explore_episodes = explore_episodes
        self.goal_threshold = goal_threshold
        self.eval_episodes = eval_episodes

        self.reward_model_num_samples = reward_model_num_samples


        self.train_reward_model_freq = train_reward_model_freq
        self.display_trajectories_freq = display_trajectories_freq

        self.human_exp_idx = 0

        self.goal = self.fake_env.extract_goal(self.fake_env.sample_goal())
        
        #print("action space low and high", self.env.action_space.low, self.env.action_space.high)

        #if train_policy_freq is None:
        #    self.train_policy_freq = 1#self.max_path_length
        #else:
        #    self.train_policy_freq = train_policy_freq
        self.start_policy_timesteps = explore_episodes#start_policy_timesteps

        self.train_policy_freq = 1
        self.summary_writer = None

        self.dict_labels = {
            'state_1': [],
            'state_2': [],
            'label': [],
            'goal':[],
        }
        now = datetime.now()
        self.dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

        self.save_trajectories_filename = f"traj_{self.dt_string}.pkl"
        self.save_trajectories_arr = []
        
        self.device = device


        if self.use_oracle:
            self.reward_model = self.oracle_model
        else:
            self.reward_model = reward_model
            self.reward_optimizer = torch.optim.Adam(list(self.reward_model.parameters()))
            self.reward_model.to(self.device)
        
        self.reward_model_epochs = reward_model_epochs


        self.traj_num_file = 0
        self.collected_trajs_dump = []
        self.success_ratio_eval_arr = []
        self.train_loss_arr = []
        self.distance_to_goal_eval_arr = []
        self.success_ratio_relabelled_arr = []
        self.eval_trajectories_arr = []
        self.train_loss_reward_model_arr = []
        self.eval_loss_arr = []
        self.distance_to_goal_eval_relabelled = []

    
    def train(self):
        self.test_rewardmodel() # TODO: plot rewardmodel
        self.total_timesteps = 0
        explore_trajs = []
        # Rollout some random trajectories to learn the reward model
        for i in range(self.explore_episodes):
            states, actions = self.collect_random_trajectories()
            self.replay_buffer.add_multiple_trajectory(states, actions)
            self.total_timesteps += self.max_path_length
            for traj in states:
                explore_trajs.append(traj)
        # goal_arr = np.array([self.goal for _ in range(len(explore_trajs))])
        self.plot_trajectories(np.array(explore_trajs), filename=f"train_trajectories_{self.total_timesteps}.png")

        # Collect and train reward model
        if not self.use_oracle:
            self.collect_and_train_reward_model()


        while self.total_timesteps < self.max_timesteps:
            # Create environment with this new function

            # Policy steps using PPO for number of timesteps
            # goal_tensor = torch.Tensor(np.array([self.goal for _ in range(self.num_envs)]))

            # self.model = PPO("MlpPolicy", self.env, verbose=2, tensorboard_log=f'runs/1', policy_kwargs=policy_kwargs, ent_coef=ent)
            # if self.start_from_scratch_every_epoch:
            #     policy_kwargs = dict()
            #     policy_kwargs['net_arch'] = [400,600,600,300]
            #     self.model = PPO("MlpPolicy", self.env, verbose=2, tensorboard_log=f'runs/{self.wandb_run.id}', ent_coef=self.entropy_coefficient, policy_kwargs=policy_kwargs)
            # import IPython
            # IPython.embed()

            # print("goal", self.goal)
            states, actions, timesteps = self.model.learn(total_timesteps=self.num_steps_per_policy_step, normalize_rewards=self.normalize_rewards, reward_model=self.reward_model, fake_env=self.fake_env) ## TODO Pass callback that will collect the rollouts
            self.goal = self.fake_env.extract_goal(states[-1][-1])

            print("total timesteps", self.total_timesteps, self.num_steps_per_policy_step, timesteps)
            print("goal", self.goal)
            self.total_timesteps += 8192#timesteps

            # eval rollout
            success_rate, distance = self.eval_policy()
            # wandb.log({'Eval/success ratio':success_rate, 'Eval/distance':distance, 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})
            # TODO
            
            self.replay_buffer.add_multiple_trajectory(states, actions)
            # arr_goal = np.array([self.goal for i in range(len(states))])
            self.plot_trajectories(states , filename=f"train_trajectories_{self.total_timesteps}.png")

            # Collect and train reward model
            if not self.use_oracle:
                self.collect_and_train_reward_model()

    def get_target_goal(self, obs):
        if  "pointmass" in self.env_name:
            return obs[2:4]
        if "pusher" in self.env_name:
            return obs[4:8]
        
        return self.fake_env.extract_goal(obs)
    def eval_policy(self):
        success_rate = 0 
        distance = 0
        eval_trajs = []
        for i in range(self.eval_episodes):
            obs = self.env_eval.reset()
            t = 0 
            this_traj = []
            while t < self.max_path_length:
                this_traj.append(obs)
                action, _states = self.model.predict(obs)
                obs, rewards, dones, info = self.env_eval.step(action)
                t += 1
                # plot some results
                    
                

            states = np.array(this_traj).transpose(1,0,2)
            for traj in states:
                eval_trajs.append(traj)

        eval_trajs = np.array(eval_trajs)

        self.plot_trajectories(eval_trajs, filename=f"eval_trajectories_{self.total_timesteps}.png")

        distances = []
        successes = []
        for traj in eval_trajs:
            dist = self.fake_env.compute_shaped_distance(self.fake_env.observation(traj[-1]), self.get_target_goal(traj[-1]))
            success = self.fake_env.compute_success(self.fake_env.observation(traj[-1]), self.get_target_goal(traj[-1]))
            distances.append(dist)
            successes.append(success)

        print("Eval/Distance:", np.mean(distances), "Eval/success rate", np.mean(successes))
        wandb.log({"Eval2/Distance":np.mean(distances), "Eval2/success rate": np.mean(successes), "timesteps":self.total_timesteps})
        
        return np.mean(distances), np.mean(successes)

    def collect_random_trajectories(self,):
        states = []
        actions = []
        state = self.env.reset()
        t = 0

        while t < self.max_path_length:

            states.append(state)
                
            if isinstance(self.env.action_space, Discrete):
                action = np.random.randint(self.env.action_space.n, size=self.num_envs)
            else:
                action_low = self.env.action_space.low #np.array([0.25, -0.5])
                action_high = self.env.action_space.high #np.array([0.75, 0.5])
                print("Action space low", action_low, "action space high", action_high)

                action_space_mean = (action_low + action_high)/2
                action_space_range = (action_high - action_low)/2
                assert self.env.action_space.shape[0] == 2
                action = np.random.normal(0, 1, size=(self.num_envs, self.env.action_space.shape[0]))
                print("Action norm is", action)
                action = action*action_space_range+action_space_mean
               
            actions.append(action)
            
            state, _, _, _ = self.env.step(action)
            t+=1
            
        states = np.array(states).transpose(1,0,2)
        if isinstance(self.env.action_space, Discrete):
            actions = np.array(actions).transpose(1,0)
        else:
            actions = np.array(actions).transpose(1,0,2)
        return states, actions

    def generate_pref_labels(self, goal_states):
        print("label from last k steps", self.label_from_last_k_steps)
        observations_1, _ = self.replay_buffer.sample_obs_last_steps(self.reward_model_num_samples, k=self.label_from_last_k_steps, last_k_trajectories=self.label_from_last_k_trajectories) # TODO: add
        observations_2, _ = self.replay_buffer.sample_obs_last_steps(self.reward_model_num_samples, k=self.label_from_last_k_steps, last_k_trajectories=self.label_from_last_k_trajectories) # TODO: add
   
        goals = [] 
        labels = []
        achieved_state_1 = []
        achieved_state_2 = []

        num_goals = len(goal_states)
        for state_1, state_2 in zip(observations_1, observations_2):
            goal_idx = np.random.randint(0, len(goal_states)) 
            goal = goal_states[goal_idx]
            labels.append(self.oracle(state_1, state_2, goal)) 

            self.num_labels_queried += 1 

            achieved_state_1.append(state_1) 
            achieved_state_2.append(state_2) 
            goals.append(goal)

        achieved_state_1 = np.array(achieved_state_1)
        achieved_state_2 = np.array(achieved_state_2)
        goals = np.array(goals)
        labels = np.array(labels)
        
        return achieved_state_1, achieved_state_2, goals, labels # TODO: check ordering
        
    def test_rewardmodel(self, goal=None):
        if not self.display_trajectories:
            return
        if goal is None:
            goal = self.goal#self.fake_env.extract_goal(self.fake_env.sample_goal())
        size=50
        if "ravens" in self.env_name or "pusher" in self.env_name or "kitchen" in self.env_name or "locobot_mujoco" in self.env_name:
            return
        goal_pos =  goal

        if "maze" in self.env_name:
            #states = np.concatenate([np.random.uniform( size=(10000, 2)), np.random.uniform(-1,1, size=(10000,2))], axis=1)
            pos = np.meshgrid(np.linspace(0, 11.5,size), np.linspace(0, 12.5,size))
            vels = np.meshgrid(np.random.uniform(-1,1, size=(size)),np.zeros((size)))
            
            pos = np.array(pos).reshape(2,-1).T
            vels = np.array(vels).reshape(2,-1).T
            states = np.concatenate([pos, vels], axis=-1)
            goals = np.repeat(goal_pos[None], size*size, axis=0)


        else:
            goal_pos = goal
            states = np.meshgrid(np.linspace(-.6,.6,200), np.linspace(-.6,.6,200))
            states = np.array(states).reshape(2,-1).T
            goals = np.repeat(goal_pos[None], 200*200, axis=0)

        states_t = torch.Tensor(self.fake_env.observation(states)).to(self.device)
        goals_t = torch.Tensor(goals).to(self.device)
        r_val = self.reward_model(states_t, goals_t)
        #print("goal pos", goal_pos.shape)
        #r_val = self.oracle_model(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        #self.display_wall(plt)
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)

        if self.env_name == "pusher":
            self.display_wall_pusher()

            plt.scatter(goal_pos[2], goal_pos[3], marker='o', s=100, color='black')
        else:
            if self.env_name == "complex_maze":
                self.display_wall_maze()
            else:
                self.display_wall()
            plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')
            # plt.scatter(self.goal[0], self.goal[1], marker='+', s=100, color='black')

        wandb.log({"reward model": wandb.Image(plt)})
        
    def collect_and_train_reward_model(self):

        print("Collecting and training reward_model")
        achieved_state_1, achieved_state_2, goals, labels = self.generate_pref_labels(np.array([self.goal]))
        # TODO: add validation buffer
        if ("maze" in self.env_name or "ravens" in self.env_name or "pointmass" in self.env_name):
            self.display_collected_labels(achieved_state_1, achieved_state_2, goals)
        
        validation_set = random.sample(range(len(achieved_state_1)), floor(len(achieved_state_1)*0.2))
         
        train_set_mask = np.ones(len(achieved_state_1), bool)
        train_set_mask[validation_set] = False

        self.reward_model_buffer.add_multiple_data_points(achieved_state_1[train_set_mask], achieved_state_2[train_set_mask], goals[train_set_mask], labels[train_set_mask])
        self.reward_model_buffer_validation.add_multiple_data_points(achieved_state_1[validation_set], achieved_state_2[validation_set], goals[validation_set], labels[validation_set])
       
        if not self.no_training:
            print("Training reward model")
            losses_reward_model, eval_loss_reward_model = self.train_reward_model(self.device)
            torch.save(self.reward_model.state_dict(), f"reward_model_{self.total_timesteps}.pth")

        else:
            losses_reward_model, eval_loss_reward_model = 0.0,0.0


        self.test_rewardmodel() # TODO: plot rewardmodel

        print("Computing reward model loss ", np.mean(losses_reward_model), "eval loss is: ", eval_loss_reward_model)
        if self.summary_writer:
            self.summary_writer.add_scalar('Lossesreward_model/Train', np.mean(losses_reward_model), self.env_nametotal_timesteps)
        wandb.log({'Lossesreward_model/Train':np.mean(losses_reward_model), 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})
        wandb.log({'Lossesreward_model/Eval':eval_loss_reward_model, 'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})

        self.train_loss_reward_model_arr.append((np.mean(losses_reward_model), self.total_timesteps))
        
        return losses_reward_model, eval_loss_reward_model

    def train_reward_model(self,device='cuda'):
        if self.no_training:
            return
        # Train standard goal conditioned policy

        loss_fn = torch.nn.CrossEntropyLoss() 
        #loss_fn = torch.nn.CrossEntropyLoss() 
        losses_eval = []

        self.reward_model.train()
        running_loss = 0.0
        prev_losses = []

        # Train the model with regular SGD
        for epoch in range(self.reward_model_epochs):  # loop over the dataset multiple times
            start = time.time()
            
            achieved_states_1, achieved_states_2, goals ,labels = self.reward_model_buffer.sample_batch(self.reward_model_batch_size)
            
            self.reward_optimizer.zero_grad()

            #t_idx = np.random.randint(len(goals), size=(batch_size,)) # Indices of first trajectory
            
            state1 = torch.Tensor(self.fake_env.observation(achieved_states_1)).to(self.device)
            state2 = torch.Tensor(self.fake_env.observation(achieved_states_2)).to(self.device)
            goal = torch.Tensor(goals).to(self.device)

            label_t = torch.Tensor(labels).long().to(self.device)
            g1 = self.reward_model(state1, goal)
            g2 = self.reward_model(state2, goal)
            g1g2 = torch.cat([g1,g2 ], axis=-1)

  
            loss = loss_fn(g1g2, label_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.reward_model.parameters(), self.clip)

            self.reward_optimizer.step()

            # print statistics
            running_loss += float(loss.item())
            prev_losses.append(float(loss.item()))
        if prev_losses[0]==prev_losses[-1]:
            print("Attention: Model degenerated!")
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
            torch.save(self.reward_model.state_dict(), f"checkpoint/reward_model_model_{dt_string}.h5")
            # Save a model file manually from the current directory:
            wandb.save(f"checkpoint/reward_model_{dt_string}.h5")
            wandb.log({"Control/Model_degenerated":1, "timesteps":self.total_timesteps})

            self.reward_model = copy.deepcopy(self.reward_model_backup)
            self.reward_optimizer = torch.optim.Adam(list(self.reward_model.parameters()))
            self.reward_model.to(self.device)
            return self.train_reward_model(self.device)
            
        self.reward_model.eval()
        eval_loss = 0.0
        achieved_states_1, achieved_states_2, goals ,labels = self.reward_model_buffer_validation.sample_batch(1000)

        state1 = torch.Tensor(self.fake_env.observation(achieved_states_1)).to(self.device)
        state2 = torch.Tensor(self.fake_env.observation(achieved_states_2)).to(self.device)
        goal = torch.Tensor(goals).to(self.device)

        label_t = torch.Tensor(labels).long().to(self.device)
        g1 = self.reward_model(state1, goal)
        g2 = self.reward_model(state2, goal)
        g1g2 = torch.cat([g1,g2 ], axis=-1)

        loss = loss_fn(g1g2, label_t)
        eval_loss = float(loss.item())

        return running_loss/self.reward_model_epochs, eval_loss

    def env_distance(self, state, goal):
        obs = self.fake_env.observation(state)
        
        if "pointmass" in self.env_name:
            return self.fake_env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return self.fake_env.get_shaped_distance(obs, goal)
            

    def oracle_model(self, state, goal):
        state = state.detach().cpu().numpy()

        goal = goal.detach().cpu().numpy()

        if self.use_wrong_oracle:
            goal = np.array([self.wrong_goal for i in range(state.shape[0])])

        dist = [
            self.env_distance(state[i], goal[i]) #+ np.random.normal(scale=self.distance_noise_std)
            for i in range(goal.shape[0])
        ] #- np.linalg.norm(state - goal, axis=1)

        scores = - torch.tensor(np.array([dist])).T
        return scores
        
    # TODO: generalise this
    def oracle(self, state1, state2, goal):
        d1_dist = self.env_distance(state1, goal) #+ np.random.normal(scale=self.distance_noise_std) #self.env.shaped_distance(state1, goal) # np.linalg.norm(state1 - goal, axis=-1)
        d2_dist = self.env_distance(state2, goal) #+ np.random.normal(scale=self.distance_noise_std) #self.env.shaped_distance(state2, goal) # np.linalg.norm(state2 - goal, axis=-1)

        if d1_dist < d2_dist:
            return 0
        else:
            return 1


    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if not self.continuous_action_space else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype).to(self.device)
        goals_torch = torch.tensor(goals, dtype=obs_dtype).to(self.device)
        actions_torch = torch.tensor(actions, dtype=action_dtype).to(self.device)

        if horizons is not None:
            horizons_torch = torch.tensor(horizons, dtype=obs_dtype).to(self.device)
        else:
            horizons_torch = None
        weights_torch = torch.tensor(weights, dtype=torch.float32).to(self.device)
        if self.continuous_action_space:
            conditional_nll = self.policy.loss_regression(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
        else:
            conditional_nll = self.policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
        nll = conditional_nll
        if self.weighted_sl:
            return torch.mean(nll * weights_torch)
        else:
            return torch.mean(nll)


    def create_video(self, images, video_filename):
        images = np.array(images).astype(np.uint8)

        if self.save_videos:
            skvideo.io.vwrite(f"{self.trajectories_videos_folder}/{video_filename}.mp4", images)
        images = images.transpose(0,3,1,2)
        
        if 'eval' in video_filename:
            wandb.log({"eval_video_trajectories":wandb.Video(images, fps=10)})
        else:
            wandb.log({"video_trajectories":wandb.Video(images, fps=10)})
    
    

    def get_distances(self, state, goal):
        obs = self.fake_env.observation(state)

        if not "kitchen" in self.env_name:
            return None, None, None, None, None, None

        per_pos_distance, per_obj_distance = self.fake_env.success_distance(obs)
        distance_to_slide = per_pos_distance['slide_cabinet']
        distance_to_hinge = per_pos_distance['hinge_cabinet']
        distance_to_microwave = per_pos_distance['microwave']
        distance_joint_slide = per_obj_distance['slide_cabinet']
        distance_joint_hinge = per_obj_distance['hinge_cabinet']
        distance_microwave = per_obj_distance['microwave']

        return distance_to_slide, distance_to_hinge, distance_to_microwave, distance_joint_slide, distance_joint_hinge, distance_microwave

    def plot_trajectories(self,traj_accumulated_states, extract=True, filename=""):
        if not self.display_trajectories:
            return
        if "pointmass" in self.env_name:
            return self.plot_trajectories_rooms(traj_accumulated_states.copy(), extract, "pointmass/" + filename)
        if self.env_name == "pusher":
            return self.plot_trajectories_pusher(traj_accumulated_states.copy(), extract, "pusher/" + filename)
        if self.env_name == "pusher_hard":
            return self.plot_trajectories_pusher_hard(traj_accumulated_states.copy(), extract, "pusher_hard/" + filename)
        if self.env_name == "complex_maze":
            #if 'train' in filename:
            #    self.plot_trajectories_complex_maze(self.replay_buffer._states.copy(), traj_accumulated_goal_states, extract, "complex_maze/"+f"train_states_preferences/replay_buffer{self.total_timesteps}.png")

            return self.plot_trajectories_complex_maze(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "complex_maze/"+filename)
        if "ravens" in self.env_name:
            return self.plot_trajectories_ravens(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "complex_maze/"+filename)

    def display_wall_maze(self):
        from matplotlib.patches import Rectangle

        maze_arr = self.fake_env.wrapped_env.base_env.maze_arr
        width, height = maze_arr.shape
        for w in range(width):
            for h in range(height):
                if maze_arr[w, h] == 10:

                    plt.gca().add_patch(Rectangle((w-0.7,h-0.7),1,1,
                    edgecolor='black',
                    facecolor='black',
                    lw=0))
                    #plt.scatter([w], [h], color="black")

    def plot_trajectories_complex_maze(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall_maze()
        
        states_plot =  traj_accumulated_states
        colors = sns.color_palette('hls', (traj_accumulated_states.shape[0]))
        for j in range(traj_accumulated_states.shape[0]):
            color = colors[j]
            plt.plot(self.fake_env.observation(states_plot[j ])[:,0], self.fake_env.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
        plt.savefig(filename)
        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def plot_trajectories_ravens(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        #self.display_wall_maze()
        
        states_plot =  traj_accumulated_states
        colors = sns.color_palette('hls', (traj_accumulated_states.shape[0]))
        for j in range(traj_accumulated_states.shape[0]):
            color = colors[j]
            plt.plot(self.fake_env.observation(states_plot[j ])[:,0], self.fake_env.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
            box_position_end = self.fake_env.observation(states_plot[j])[-1,3:]
            plt.scatter(box_position_end[0],
                        box_position_end[1], marker='s', s=20, color=color)
            if len(box_position_end) > 2:
                plt.scatter(box_position_end[2],
                    box_position_end[3], marker='^', s=20, color=color)
            if len(box_position_end) > 4:
                plt.scatter(box_position_end[4],
                    box_position_end[5], marker='D', s=20, color=color)
                    
        box_position = self.fake_env.observation(states_plot[j])[0,3:]
        
        goal_position = self.fake_env.sample_goal()
        plt.scatter(box_position[0],
                    box_position[1], marker='+', s=20, color="black")
        plt.scatter(goal_position[-2],
                    goal_position[-1], marker='x', s=20, color="yellow")
        if len(box_position) > 2:
            plt.scatter(box_position[2],
                box_position[3], marker='+', s=20, color="red")
        if len(box_position) > 4:
            plt.scatter(box_position[4],
                box_position[5], marker='+', s=20, color="blue")
        plt.xlim([0.25, 0.75])
        plt.ylim([-0.5, 0.5])
        from PIL import Image
        plt.savefig(filename)
        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})


    def plot_trajectories_rooms(self,traj_accumulated_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        traj_accumulated_states = np.array(traj_accumulated_states)        
        colors = sns.color_palette('hls', (len(traj_accumulated_states)))
        for j in range(len(traj_accumulated_states)):
            color = colors[j]
            plt.plot(self.fake_env.observation(traj_accumulated_states[j ])[:,0], self.fake_env.observation(traj_accumulated_states[j])[:, 1], color=color, zorder = -1)
            #if 'train_states_preferences' in filename:
            #    color = 'black'
            
            plt.scatter(traj_accumulated_states[j][-1,-2],
                    traj_accumulated_states[j][-1, -1], marker='o', s=20, color=color, zorder=1)
        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

        print("Plotting traj rooms")

    def plot_trajectories_pusher(self,traj_accumulated_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        plt.cla()
        self.display_wall_pusher()
        #if extract:

        states_plot =  self.fake_env._extract_sgoal(traj_accumulated_states)

        #else:
        #    states_plot = traj_accumulated_states
        #shutil.rmtree("train_states_preferences")
        colors = sns.color_palette('hls', (states_plot.shape[0]))
        for j in range(states_plot.shape[0]):
            color = colors[j]
            plt.plot(states_plot[j ][:,2], states_plot[j][:, 3], color=color)
            plt.scatter(states_plot[j][:,-2],
                    states_plot[j][:,-1], marker='o', s=20, color=color)
        
        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def plot_trajectories_pusher_hard(self,traj_accumulated_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        plt.cla()
        self.display_wall_pusher_hard()
        #if extract:
        traj_accumulated_states = np.array(traj_accumulated_states)

        states_plot =  traj_accumulated_states

        #else:
        #    states_plot = traj_accumulated_states
        #shutil.rmtree("train_states_preferences")
        colors = sns.color_palette('hls', (len(states_plot)))
        for j in range(len(states_plot)):
            color = colors[j]
            plt.plot(self.fake_env.observation(states_plot[j ])[:,0], self.fake_env.observation(states_plot[j])[:, 1], color=color)

            plt.scatter(states_plot[j][-1,-2],
                    states_plot[j][-1,-1], marker='+', s=20, color=color)
            plt.scatter(states_plot[j][-1,-4],
                    states_plot[j][-1,-3], marker='o', s=20, color=color)
            plt.scatter(self.fake_env.observation(states_plot[j ])[:,2], self.fake_env.observation(states_plot[j])[:, 3], marker='x', s=20, color=color)
                    
        plt.savefig(filename)

        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def display_collected_labels(self, state_1, state_2, goals, is_oracle=False):
        if  "pointmass" in self.env_name :
            self.display_collected_labels_pointmass(state_1, state_2, goals)
        elif self.env_name == "complex_maze" and not is_oracle:
            self.display_collected_labels_complex_maze(state_1, state_2, goals)
        elif "ravens" in self.env_name :
            self.display_collected_labels_ravens(state_1, state_2, goals, is_oracle)

    def display_collected_labels_ravens(self, state_1, state_2, goals, is_oracle=False):
            # plot added trajectories to fake replay buffer
            print("display collected labels ravens")
            plt.clf()
            #self.display_wall_maze()
            
            colors = sns.color_palette('hls', (state_1.shape[0]))
            plt.xlim([0.25, 0.75])
            plt.ylim([-0.5, 0.5])
            for j in range(state_1.shape[0]):
                color = colors[j]
                if is_oracle:
                    plt.scatter(self.fake_env.observation(state_1[j])[0], self.fake_env.observation(state_1[j])[1], color=color, zorder = -1)
                else:
                    plt.scatter(self.fake_env.observation(state_1[j])[0], self.fake_env.observation(state_1[j])[1], color=color, zorder = -1)
                    plt.scatter(self.fake_env.observation(state_2[j])[0], self.fake_env.observation(state_2[j])[1], color=color, zorder = -1)
                
                if not is_oracle:
                    plt.scatter(goals[j][0],
                        goals[j][1], marker='+', s=20, color=color, zorder=1)
            if is_oracle:
                plt.scatter(goals[0],
                        goals[1], marker='+', s=20, color=color, zorder=1)
            filename = self.env_name+f"/reward_model_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            plt.savefig(filename)
            
            if is_oracle:
                wandb.log({"reward_model_candidates": wandb.Image(plt)})
            else:
                wandb.log({"reward_model_labels": wandb.Image(plt)})

    def display_collected_labels_complex_maze(self, state_1, state_2, goals):
            # plot added trajectories to fake replay buffer
            plt.clf()
            self.display_wall_maze()
            
            colors = sns.color_palette('hls', (state_1.shape[0]))
            for j in range(state_1.shape[0]):
                color = colors[j]
                plt.scatter(self.fake_env.observation(state_1[j])[0], self.fake_env.observation(state_1[j])[1], color=color, zorder = -1)
                plt.scatter(self.fake_env.observation(state_2[j])[0], self.fake_env.observation(state_2[j])[1], color=color, zorder = -1)
                
                plt.scatter(goals[j][0],
                        goals[j][1], marker='o', s=20, color=color, zorder=1)
            
            filename = "complex_maze/"+f"reward_model_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            plt.savefig(filename)
            
            wandb.log({"reward_model_labels": wandb.Image(plt)})

    def display_collected_labels_pointmass(self, state_1, state_2, goals):
            # plot added trajectories to fake replay buffer
            plt.clf()
            self.display_wall()
            
            colors = sns.color_palette('hls', (state_1.shape[0]))
            for j in range(state_1.shape[0]):
                color = colors[j]
                plt.scatter(self.fake_env.observation(state_1[j])[0], self.fake_env.observation(state_1[j])[1], color=color, zorder = -1)
                plt.scatter(self.fake_env.observation(state_2[j])[0], self.fake_env.observation(state_2[j])[1], color=color, zorder = -1)
                
                plt.scatter(goals[j][0],
                        goals[j][1], marker='o', s=20, color=color, zorder=1)
            
            filename = f"{self.env_name}/"+f"reward_model_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            plt.savefig(filename)
            
            wandb.log({"reward_model_labels": wandb.Image(plt)})

    def display_goal_selection(self, states, goal, commanded_goal):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.test_rewardmodel()

        self.display_wall_maze()
        
        for j in range(states.shape[0]):
            plt.scatter(self.fake_env.observation(states[j])[0], self.fake_env.observation(states[j])[1], color="black")
            
        plt.scatter(goal[0],
                goal[1], marker='o', s=20, color="yellow", zorder=1)

        plt.scatter(commanded_goal[0],
                commanded_goal[1], marker='o', s=20, color="green", zorder=1)
        
        filename = "complex_maze/"+f"goal_selection_candidates_{self.total_timesteps}_{np.random.randint(10)}.png"
        wandb.log({"reward_model_labels_and_state": wandb.Image(plt)})
    

    def display_wall(self):
        walls = self.fake_env.base_env.room.get_walls()

        if self.env_name == "pointmass":
            walls.append([[0.6,-0.6], [0.6,0.6]])
            walls.append([[0.6,0.6], [-0.6,0.6]])
            walls.append([[-0.6,0.6], [-0.6,-0.6]])
            walls.append([[-0.6,-0.6], [0.6,-0.6]])
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='',  color = 'black', linewidth=4)
    def display_wall_pusher_hard(self):
        walls = [
            [(-0.025, 0.625), (0.025, 0.625)],
            [(0.025, 0.625), (0.025, 0.575)],
            [(0.025, 0.575), (-0.025, 0.575) ],
            [(-0.025, 0.575), (-0.025, 0.625)]
        ]

        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='o',  color = 'b')
    def display_wall_pusher(self):
        walls = [
            [(-0.025, 0.625), (0.025, 0.625)],
            [(0.025, 0.625), (0.025, 0.575)],
            [(0.025, 0.575), (-0.025, 0.575) ],
            [(-0.025, 0.575), (-0.025, 0.625)]
        ]

        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='o',  color = 'b')
