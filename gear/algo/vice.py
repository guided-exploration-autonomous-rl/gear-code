from fileinput import filename
from signal import default_int_handler
from re import I
from telnetlib import IP, PRAGMA_HEARTBEAT
from click import command
import numpy as np
from gear.envs.ravens_block_stacking import RavensGoalEnvStackBlock
from gear.envs.complex_maze_env import ComplexMazeGoalEnv
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
from gear.algo import buffer, networks,reachable_set
import matplotlib.cm as cm
import os
from datetime import datetime
import shutil
from gear.envs.room_env import PointmassGoalEnv
from gear.envs.sawyer_push import SawyerPushGoalEnv
from gear.envs.sawyer_push_hard import SawyerHardPushGoalEnv
from gear.envs.kitchen_simplified_state_space import KitchenGoalEnv
from gear.envs.ravens_env_continuous import RavensGoalEnvContinuous
from gear.envs.ravens_env_simple import RavensGoalEnvSimple
from gear.envs.ravens_env_reaching import RavensGoalEnvReaching
from gear.envs.ravens_env_pick_place import RavensGoalEnvPickAndPlace
from gear.envs.ravens_env_pick_or_place import RavensGoalEnvPickOrPlace
from gear.envs.ravens_block_stack_continuous import RavensGoalEnvStackBlockContinuous
from gear.envs.locobot_env_mujoco import LoCoBotEnvMujoco


from gear.envs.kitchen_env_sequential import KitchenSequentialGoalEnv
from gear.envs.kitchen_env_3d import Kitchen3DGoalEnv

import wandb
import skvideo.io
import random 
import cv2

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

import matplotlib.pyplot as plt


NOT_ANSWERED = -2
DONT_KNOW = -1

curr_label = 0


class Index:
    def first(self, event):
        global curr_label
        curr_label = 0
        #plt.close()
    def second(self, event):
        global curr_label
        curr_label = 1
        #plt.close()
    def dontknow(self, event):
        global curr_label
        curr_label = DONT_KNOW
        #plt.close()

#TODO: missing to dump trajectories

# New version GCSL with preferences
# Sample random goals
# Search on the buffer the set of achieved goals and pick up the closest achieved goal
# Launch batch of trajectories with all new achieved goals 
# we can launch one batch without exploration, just to reinforce stopping at the point and then another one with exploration
# add all trajectories to the buffer
# train standard GCSL
# THIS SHOULD WORK BY 11am, 12pm we have positive results on the 2d point environment

class GCSL:
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
        policy,
        policy_list,
        goal_selector,
        replay_buffer,
        goal_selector_buffer,
        goal_selector_buffer_validation,
        validation_buffer=None,
        max_timesteps=1e6,
        max_path_length=50,
        # Exploration Strategy
        explore_episodes=1e4,
        expl_noise=0.1,
        # Evaluation / Logging
        goal_threshold=0.05,
        eval_freq=5e3,
        eval_episodes=200,
        save_every_iteration=False,
        log_tensorboard=False,
        # Policy Optimization Parameters
        start_policy_timesteps=0,
        batch_size=100,
        n_accumulations=1,
        policy_updates_per_step=1,
        train_policy_freq=None,
        demonstrations_kwargs=dict(),
        train_with_preferences=True,
        lr=5e-4,
        goal_selector_epochs = 300,
        train_goal_selector_freq = 10,#5000,
        display_trajectories_freq = 20,
        use_oracle=False,
        exploration_horizon=30,
        goal_selector_num_samples=100,
        comment="",
        select_best_sample_size = 1000,
        load_buffer=False,
        save_buffer=-1,
        goal_selector_batch_size = 128,
        train_regression = False,
        load_goal_selector=False, 
        render=False,
        not_save_videos=False,
        sample_softmax = False,
        display_plots=False,
        data_folder="data",
        clip=5,
        remove_last_steps_when_stopped = True,
        exploration_when_stopped = True,
        distance_noise_std = 0.0,
        save_videos=True,
        logger_dump=False,
        human_input=False,
        epsilon_greedy_exploration=0.2,
        set_desired_when_stopped=True,
        remove_last_k_steps=10, # steps to look into for checking whether it stopped
        select_last_k_steps=10,
        explore_length=10,
        greedy_before_stopping=True,
        stopped_thresh = 0.05,
        weighted_sl = False,
        sample_new_goal_freq =1,
        k_goal=1,
        start_frontier = -1,
        frontier_expansion_rate=-1,
        frontier_expansion_freq=-1,
        select_goal_from_last_k_trajectories=-1,
        throw_trajectories_not_reaching_goal=False,
        command_goal_if_too_close=False,
        epsilon_greedy_rollout=0,
        label_from_last_k_steps=-1,
        label_from_last_k_trajectories=-1,
        contrastive = False,
        deterministic_rollout = False,
        repeat_previous_action_prob=0.9,
        continuous_action_space=False,
        expl_noise_mean = 0,
        expl_noise_std = 1,
        desired_goal_sampling_freq=0.0,
        check_if_stopped=False,
        check_if_close=False,
        human_data_file=None,
        wait_time=30,
        use_wrong_oracle = False,
        no_training_goal_selector=False,
        stop_training_goal_selector_after=-1,
        reset_free=False,
        use_reachable_set=True,
        reachable_sample_rate=300,
        reachable_thres=0.5,
        input_image_size=None,
        use_images_in_policy=False,
        use_images_in_reward_model=False,
        use_images_in_stopping_criteria=False,
        classifier_model=None,
        num_demos=0,
        pretrain_policy=False,
        pretrain_goal_selector=False,
        demo_pretrain_epochs=5000,
        offset=500,
        buffer_random_init=300,
        use_reachable_set_densities=False,
        demos_folder_name=None,
        use_prop = False,
        vice_num_positives=1000,
    ):
        self.use_prop = use_prop
        self.vice_num_positives = vice_num_positives
        # self.eval_env = copy.deepcopy(env)
        self.num_demos_goal_selector = goal_selector_buffer.max_buffer_size
        self.demos_folder_name=demos_folder_name
        self.offset = offset
        self.num_demos = num_demos
        self.pretrain_policy = pretrain_policy
        self.pretrain_goal_selector = pretrain_goal_selector
        self.demo_pretrain_epochs = demo_pretrain_epochs
        self.eval_env = env_eval
        self.curr_goal = None
        self.reset_free = reset_free
        self.use_reachable_set = use_reachable_set
        self.reachable_sample_rate = reachable_sample_rate
        self.reachable_thres = reachable_thres
        self.input_image_size=input_image_size
        self.use_images_in_policy=use_images_in_policy
        self.use_images_in_reward_model=use_images_in_reward_model
        self.use_images_in_stopping_criteria=use_images_in_stopping_criteria
        self.classifier_model = classifier_model
        if self.use_images_in_stopping_criteria:
            self.classifier_model.to(self.device)
            self.classifier_optimizer = torch.optim.Adam(self.classifier_model.parameters(), lr=lr)
        if stop_training_goal_selector_after <= 0:
            self.stop_training_goal_selector_after = max_timesteps
        else:
            self.stop_training_goal_selector_after = stop_training_goal_selector_after
        self.no_training_goal_selector = no_training_goal_selector
        self.wait_time=wait_time
        if human_input:
            matplotlib.use('TkAgg')
        # self.fake_env = copy.deepcopy(env)
        self.not_save_videos = not_save_videos
        self.expl_noise_mean = expl_noise_mean
        self.expl_noise_std = expl_noise_std
        self.continuous_action_space = continuous_action_space
        self.deterministic_rollout = deterministic_rollout
        self.contrastive = contrastive
        if label_from_last_k_trajectories == -1:
            self.label_from_last_k_trajectories = train_goal_selector_freq
        else:
            self.label_from_last_k_trajectories = label_from_last_k_trajectories
        self.repeat_previous_action_prob = repeat_previous_action_prob
        self.desired_goal_sampling_freq = desired_goal_sampling_freq
        self.goal_selector_backup = copy.deepcopy(goal_selector)
        self.check_if_stopped = check_if_stopped
        self.check_if_close = check_if_close

        if human_data_file is not None and len(human_data_file)!=0:
            print("human data file")
            self.human_data_info = pickle.load(open(human_data_file, "rb"))
            self.human_data_index = 0
        else:
            self.human_data_info = None

        self. goal_selector_buffer_validation = goal_selector_buffer_validation

        if label_from_last_k_steps==-1:
            self.label_from_last_k_steps = max_path_length
        else:
            self.label_from_last_k_steps = label_from_last_k_steps

        self.epsilon_greedy_rollout = epsilon_greedy_rollout
        self.command_goal_if_too_close = command_goal_if_too_close
        if select_goal_from_last_k_trajectories == -1:
            self.select_goal_from_last_k_trajectories = replay_buffer.max_buffer_size
        else:
            self.select_goal_from_last_k_trajectories = select_goal_from_last_k_trajectories

        print("Select goal from last k trajectories", self.select_goal_from_last_k_trajectories)
        self.curr_frontier = max_path_length
        # if start_frontier == -1 or self.pretrain_policy:
        #     self.curr_frontier = max_path_length
        # else:
        #     self.curr_frontier = min(max_path_length, start_frontier)

        print("Curr frontier beginning", self.curr_frontier)
        if frontier_expansion_freq == -1:
            self.frontier_expansion_freq = sample_new_goal_freq
        else:
            self.frontier_expansion_freq = frontier_expansion_freq

        self. throw_trajectories_not_reaching_goal = throw_trajectories_not_reaching_goal

        if frontier_expansion_rate == -1:
            self.frontier_expansion_rate = explore_length
        else:
            self.frontier_expansion_rate = frontier_expansion_rate

        self.sample_new_goal_freq = sample_new_goal_freq
        self.weighted_sl = weighted_sl
        self.env = env
        
        self.device = "cuda:0" #"cuda:0"
        print("cuda device", self.device)
        self.policy = policy
        self.policy.to(self.device)
        self.policy_list = policy_list
        for i in range(len(self.policy_list)):
            self.policy_list[i].to(self.device)
    
        self.random_policy = copy.deepcopy(policy)
        self.shortest = False

        self.explore_length = explore_length

        self.goal_selector_batch_size = goal_selector_batch_size
        self.train_regression = train_regression
        self.set_desired_when_stopped = set_desired_when_stopped
        self.stopped_thresh = stopped_thresh

        self.k_goal = k_goal
     
        #with open(f'human_dataset_06_10_2022_20:15:53.pickle', 'rb') as handle:
        #    self.human_data = pickle.load(handle)
        #    print(len(self.human_data))
        
        self.greedy_before_stopping = greedy_before_stopping
        self.remove_last_k_steps = remove_last_k_steps
        if select_last_k_steps == -1:
            self.select_last_k_steps = explore_length
        else:
            self.select_last_k_steps = select_last_k_steps        
        self.total_timesteps = 0

        self.previous_goal = None

        self.buffer_filename = "buffer_saved.csv"
        self.val_buffer_filename = "val_buffer_saved.csv"
        self.data_folder = data_folder

        self.train_with_preferences = train_with_preferences

        self.exploration_when_stopped = exploration_when_stopped

        if not self.train_with_preferences:
            self.exploration_when_stopped = False

        self.load_buffer = load_buffer
        self.save_buffer = save_buffer


        self.use_wrong_oracle = use_wrong_oracle
        if self.use_wrong_oracle:
            self.wrong_goal = [-0.2,0.2]

        self.comment = comment
        self.display_plots = display_plots
        self.lr = lr
        self.clip = clip
        self.evaluate_goal_selector = True

        self.goal_selector_buffer = goal_selector_buffer

        self.select_best_sample_size = select_best_sample_size

        self.store_model = False

        self.num_labels_queried = 0

        self.epsilon_greedy_exploration = epsilon_greedy_exploration

        self.load_goal_selector = load_goal_selector

        self.remove_last_steps_when_stopped = remove_last_steps_when_stopped

        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer

        self.max_timesteps = max_timesteps
        self.max_path_length = max_path_length

        self.explore_episodes = explore_episodes
        self.expl_noise = expl_noise
        self.render = render
        self.goal_threshold = goal_threshold
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_every_iteration = save_every_iteration
        self.buffer_random_init = buffer_random_init    #image_rooms: 100, locobot_four_loc: 1000

        self.goal_selector_num_samples = goal_selector_num_samples


        self.train_goal_selector_freq = train_goal_selector_freq
        self.display_trajectories_freq = display_trajectories_freq

        self.human_exp_idx = 0
        self.distance_noise_std = distance_noise_std
        
        #print("action space low and high", self.env.action_space.low, self.env.action_space.high)

        #if train_policy_freq is None:
        #    self.train_policy_freq = 1#self.max_path_length
        #else:
        #    self.train_policy_freq = train_policy_freq
        self.start_policy_timesteps = explore_episodes#start_policy_timesteps

        self.train_policy_freq = 1
        print("Train policy freq is, ", train_policy_freq)

        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_optimizer_list = []
        for i in range(len(self.policy_list)):
            self.policy_optimizer_list.append(torch.optim.Adam(self.policy_list[i].parameters(), lr=lr))
        
        self.log_tensorboard = log_tensorboard and tensorboard_enabled
        self.summary_writer = None

        self.exploration_horizon = exploration_horizon

        self.logger_dump = logger_dump

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
        
        self.use_oracle = use_oracle
        if self.use_oracle:
            self.goal_selector = self.oracle_model
            if load_goal_selector:
                self.goal_selector = goal_selector
                self.goal_selector.load_state_dict(torch.load("goal_selector.pth"))
        else:
            self.goal_selector = goal_selector
            if load_goal_selector:
                self.goal_selector.load_state_dict(torch.load("goal_selector.pth"))
            self.reward_optimizer = torch.optim.Adam(list(self.goal_selector.parameters()))
            self.goal_selector.to(self.device)
        
        

        self.goal_selector_epochs = goal_selector_epochs


        self.sample_softmax = sample_softmax

        self.human_input = human_input

        self.traj_num_file = 0
        self.collected_trajs_dump = []
        self.success_ratio_eval_arr = []
        self.train_loss_arr = []
        self.distance_to_goal_eval_arr = []
        self.success_ratio_relabelled_arr = []
        self.eval_trajectories_arr = []
        self.train_loss_goal_selector_arr = []
        self.eval_loss_arr = []
        self.distance_to_goal_eval_relabelled = []
        

        if isinstance(self.env.wrapped_env, PointmassGoalEnv):
            self.env_name = "pointmass"
        if isinstance(self.env.wrapped_env, SawyerPushGoalEnv):
            self.env_name ="pusher"        
        if isinstance(self.env.wrapped_env, SawyerHardPushGoalEnv):
            self.env_name ="pusher_hard"
        if isinstance(self.env.wrapped_env, KitchenGoalEnv):
            self.env_name ="kitchen"
        if isinstance(self.env.wrapped_env, Kitchen3DGoalEnv):
            self.env_name ="kitchen3D"
        if isinstance(self.env.wrapped_env, KitchenSequentialGoalEnv):
            self.env_name ="kitchenSeq"
        if isinstance(self.env.wrapped_env, RavensGoalEnvContinuous):
            self.env_name = "ravens_continous"        
        if isinstance(self.env.wrapped_env, RavensGoalEnvSimple):
            self.env_name = "ravens_simple"
        if isinstance(self.env.wrapped_env, RavensGoalEnvReaching):
            self.env_name = "ravens_reaching"
        if isinstance(self.env.wrapped_env, RavensGoalEnvPickAndPlace):
            self.env_name = "ravens_pick_place"
        if isinstance(self.env.wrapped_env, RavensGoalEnvPickOrPlace):
            self.env_name = "ravens_pick_or_place"
        if isinstance(self.env.wrapped_env, RavensGoalEnvStackBlock):
            self.env_name = "ravens_stack_blocks"
        if isinstance(self.env.wrapped_env, RavensGoalEnvStackBlockContinuous):
            self.env_name = "ravens_stack_block_continuous"       
        if isinstance(self.env.wrapped_env, ComplexMazeGoalEnv):
            self.env_name = "complex_maze"
        if isinstance(self.env.wrapped_env, LoCoBotEnvMujoco):
            self.env_name = "locobot_mujoco"
        os.makedirs(self.data_folder, exist_ok=True)
        print("Goal selector batch size", self.goal_selector_batch_size)
        os.makedirs(os.path.join(self.data_folder, 'eval_trajectories'), exist_ok=True)

        # Initialize reachable set
        self.use_reachable_set = use_reachable_set or use_reachable_set_densities
        self.use_reachable_set_densities = use_reachable_set_densities
        


        if self.no_training_goal_selector:
            self.test_goal_selector(0)

        goal_positives_0 = self.collect_goal_data(goal=0)
        goal_positives_1 = self.collect_goal_data(goal=1)

        self.goal_selector_buffer.add_multiple_data_points(goal_positives_0, goal=0)
        self.goal_selector_buffer.add_multiple_data_points(goal_positives_1, goal=1)

    def collect_goal_data(self, goal=0):
        all_states = []
        for i in range(self.vice_num_positives):
            state = self.eval_env.nearby_goal(goal)
            all_states.append(state)
        
        return all_states
    
    def contrastive_loss(self, pred, label):
        label = label.float()
        pos = label@torch.clamp(pred[:,0]-pred[:,1], min=0)
        neg = (1-label)@torch.clamp(pred[:,1]-pred[:,0], min=0)

        #print("pos shape", pos.shape)
        return  pos + neg
    
    def eval_goal_selector(self, eval_data, batch_size=32):
        achieved_states_1, achieved_states_2, goals ,labels = eval_data

        losses = []
        idxs = np.array(range(len(goals)))
        num_batches = len(idxs) // batch_size + 1
        losses = []
        loss_fn = torch.nn.CrossEntropyLoss()
        losses_eval = []

        # Eval the model
        mean_loss = 0.0
        start = time.time()
        total_samples = 0
        accuracy = 0
        for i in range(num_batches):

            t_idx = np.random.randint(len(goals), size=(batch_size,)) # Indices of first trajectory
                
            state1 = torch.Tensor(achieved_states_1[t_idx]).to(self.device)
            state2 = torch.Tensor(achieved_states_2[t_idx]).to(self.device)
            goal = torch.Tensor(goals[t_idx]).to(self.device)
            label_t = torch.Tensor(labels[t_idx]).long().to(self.device)

            g1g2 = torch.cat([self.goal_selector(state1, goal), self.goal_selector(state2, goal)], axis=-1)
            loss = loss_fn(g1g2, label_t)
            pred = torch.argmax(g1g2, dim=-1)
            accuracy += torch.sum(pred == label_t)
            total_samples+=len(label_t)
            # print statistics
            mean_loss += loss.item()

        mean_loss /=num_batches
        accuracy = accuracy.cpu().numpy() / total_samples

        return mean_loss, accuracy

    def train_goal_selector(self,epochs=-1):
        # Train standard goal conditioned policy
        if epochs == -1:
            epochs = self.goal_selector_epochs

        if self.goal_selector_buffer.current_buffer_size_neg == 0:
            return 0.0,0.0
        loss_fn = torch.nn.BCELoss()

        losses_eval = []

        self.goal_selector.train()
        running_loss = 0.0
        prev_losses = []
        # Train the model with regular SGD
        for epoch in range(self.goal_selector_epochs):  # loop over the dataset multiple times
            start = time.time()
            
            states ,goals, labels = self.goal_selector_buffer.sample_batch(self.goal_selector_batch_size, goal=np.random.randint(2))
            self.reward_optimizer.zero_grad()

            #t_idx = np.random.randint(len(goals), size=(batch_size,)) # Indices of first trajectory
            states = torch.Tensor(states).to(self.device)
            goals = torch.Tensor(goals).to(self.device)

            label_t = torch.Tensor(labels).float().to(self.device).reshape(-1)
            m = torch.nn.Sigmoid()
            pred = m(self.goal_selector(states, goals)).float().reshape(-1)
            loss = loss_fn(pred, label_t)

            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.goal_selector.parameters(), self.clip)

            self.reward_optimizer.step()

            # print statistics
            running_loss += float(loss.item())
            prev_losses.append(float(loss.item()))
        # if prev_losses[0]==prev_losses[-1]:
        #     print("Attention: Model degenerated!")
        #     now = datetime.now()
        #     dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        #     torch.save(self.goal_selector.state_dict(), f"checkpoint/goal_selector_model_{dt_string}.h5")
        #     # Save a model file manually from the current directory:
        #     wandb.save(f"checkpoint/goal_selector_model_{dt_string}.h5")
        #     wandb.log({"Control/Model_degenerated":1, "timesteps":self.total_timesteps})

        #     self.goal_selector = copy.deepcopy(self.goal_selector_backup)
        #     self.reward_optimizer = torch.optim.Adam(list(self.goal_selector.parameters()))
        #     self.goal_selector.to(self.device)
        #     return 0,0
            
        # self.goal_selector.eval()
        # eval_loss = 0.0
        # if self.goal_selector_buffer_validation.current_buffer_size == 0:
        #     return running_loss/self.goal_selector_epochs, eval_loss
        # achieved_states_1, achieved_states_2, goals ,labels = self.goal_selector_buffer_validation.sample_batch(1000)

        # state1 = torch.Tensor(achieved_states_1).to(self.device)
        # state2 = torch.Tensor(achieved_states_2).to(self.device)
        # goal = torch.Tensor(goals).to(self.device)

        # label_t = torch.Tensor(labels).long().to(self.device)
        # g1 = self.goal_selector(state1, goal)
        # g2 = self.goal_selector(state2, goal)
        # g1g2 = torch.cat([g1,g2 ], axis=-1)
        # if self.contrastive:
        #     loss = self.contrastive_loss(g1g2, label_t)
        # else:
        #     #mean = torch.mean(g1)
        #     #std = torch.std(g1)
        #     #g1_norm = torch.exp((g1-mean)/std)
        #     #g2_norm = torch.exp((g2-mean)/std)
        #     #g1g2 = g1_norm/(g2_norm+g1_norm) #torch.cat([g1,g2 ], axis=-1)
        #     #g1g2 = g1g2.squeeze()
        #     loss = loss_fn(g1g2, label_t)
        # eval_loss = float(loss.item())
        # #if eval_data is not None:
        # #    eval_loss, _ = self.eval_goal_selector(eval_data, batch_size)
        # #    losses_eval.append(eval_loss)
        return running_loss/self.goal_selector_epochs, 0#, (losses_eval, acc_eval)
    
    
    def get_closest_achieved_state(self, goal_candidates, observations=None, device="cuda"):
        reached_state_idxs = []
        if observations is None:
            observations, _ = self.replay_buffer.sample_obs_last_steps(self.select_best_sample_size, self.select_last_k_steps, last_k_trajectories=self.select_goal_from_last_k_trajectories)
        #print("observations 0", observations[0])
        achieved_states = self.env.observation(observations)
        #print("achieved states", achieved_states[0])
        if self.full_iters % self.display_trajectories_freq == 0:
            self.display_collected_labels(achieved_states, achieved_states, goal_candidates[0], is_oracle=True)
        request_goals = []

        for goal_candidate in goal_candidates:
            
            state_tensor = torch.Tensor(achieved_states).to(self.device)
            goal_tensor = torch.Tensor(np.repeat(goal_candidate[None], len(achieved_states), axis=0)).to(self.device)
            if self.use_oracle:
                reward_vals = self.oracle_model(state_tensor, goal_tensor).cpu().detach().numpy()
                self.num_labels_queried += len(state_tensor)
            else:
                reward_vals = self.goal_selector(state_tensor, goal_tensor).cpu().detach().numpy()
            
            if self.sample_softmax:
                best_idx = torch.distributions.Categorical(logits=torch.tensor(reward_vals.reshape(-1))).sample()
            else:
                best_idx = reward_vals.reshape(-1).argsort()[-self.k_goal]
                best_idx_max = reward_vals.argmax()

                #assert reward_vals[best_idx_max] == reward_vals[best_idx]

            request_goals.append(achieved_states[best_idx])

            if self.full_iters % self.display_trajectories_freq == 0 and ("maze" in self.env_name or "room" in self.env_name):
                self.display_goal_selection(observations, goal_candidate, achieved_states[best_idx])
        request_goals = np.array(request_goals)

        return request_goals

    def env_distance(self, state, goal):
        obs = self.env.observation(state)
        
        if isinstance(self.env.wrapped_env, PointmassGoalEnv):
            return self.env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return self.env.get_shaped_distance(obs, goal)
            
        #if isinstance(self.env.wrapped_env, KitchenGoalEnv):
        #    state = self.env.observation(state)
        #    if goal.shape[0]==90:
        #        goal = self.env.extract_goal(goal)
        #    return self.env.get_shaped_distance(state, goal)
        return None
    def oracle_model(self, state, goal):
        state = state.detach().cpu().numpy()

        goal = goal.detach().cpu().numpy()

        if self.use_wrong_oracle:
            goal = np.array([self.wrong_goal for i in range(state.shape[0])])

        dist = [
            self.env_distance(state[i], goal[i]) + np.random.normal(scale=self.distance_noise_std)
            for i in range(goal.shape[0])
        ] #- np.linalg.norm(state - goal, axis=1)

        scores = - torch.tensor(np.array([dist])).T
        return scores
        
    # TODO: generalise this
    def oracle(self, state1, state2, goal):
        if self.use_wrong_oracle:
            goal = self.wrong_goal

        d1_dist = self.env_distance(state1, goal) + np.random.normal(scale=self.distance_noise_std) #self.env.shaped_distance(state1, goal) # np.linalg.norm(state1 - goal, axis=-1)
        d2_dist = self.env_distance(state2, goal) + np.random.normal(scale=self.distance_noise_std) #self.env.shaped_distance(state2, goal) # np.linalg.norm(state2 - goal, axis=-1)

        if d1_dist < d2_dist:
            return 0
        else:
            return 1

        
    def display_wall_fig(self, fig, ax):
        walls = self.env.base_env.room.get_walls()
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            ax.plot([sx, ex], [sy, ey], marker='o',  color = 'b')

    def plot_pusher_hard(self, state_1, state_2, goal):

        from matplotlib.patches import Rectangle
        center1 = np.array([-0.1, 0.5])
        center2 = np.array([0.1, 0.7])
        dim = np.array([0.025,0.15])*2
        #plt.Rectangle(center1 - dim/2, dim[0], dim[1], edgecolor='black', facecolor="black", lw=0)
        plt.gca().add_patch(Rectangle((center1[0]-dim[0]/2, center1[1]-dim[1]/2) , dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
        plt.gca().add_patch(Rectangle((center2[0]-dim[0]/2, center2[1]-dim[1]/2), dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
        center_board = np.array([0,0.6])
        dim_board = np.array([0.4, 0.2])*2

        plt.xlim((center_board[0]-dim_board[0]/2, center_board[0]+dim_board[0]/2))
        plt.ylim((center_board[1]-dim_board[1]/2, center_board[1]+dim_board[1]/2))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(self.env.observation(state_1)[0], self.env.observation(state_1)[1], zorder=2, color="blue", marker="x")
        plt.scatter(self.env.observation(state_2)[0], self.env.observation(state_2)[1], zorder=2, color="red", marker="x")
        plt.scatter(self.env.observation(state_1)[2], self.env.observation(state_1)[3], zorder=1, color="blue", marker="o")
        plt.scatter(self.env.observation(state_2)[2], self.env.observation(state_2)[3], zorder=1, color="red", marker="o")



        plt.scatter(self.env.observation(goal)[2], self.env.observation(goal)[3], marker='o', s=20, color="black")
                
    def ask_human_labels(self, state1, state2, goal):
        #if self.human_exp_idx < len(self.human_data['label']):
        #    label = self.human_data['label'][self.human_exp_idx]
        #    self.human_exp_idx += 1
        #    return label
        #else:
            from matplotlib.widgets import Button
            global curr_label
            img1 = self.fake_env.image_render_state(state1)
            img2 = self.fake_env.image_render_state(state2)
            curr_label = NOT_ANSWERED
            callback = Index()
            fig, ax = plt.subplots(1)
            fig.set_size_inches(15,8)
            self.plot_pusher_hard(state1, state2, goal)

            #ax[0].imshow(img1)
            #ax[1].imshow(img2)
            #fig.subplots_adjust(bottom=0.2)
            axfirst = fig.add_axes([0.7,0.05, 0.1, 0.075])
            axsecond = fig.add_axes([0.81,0.05,0.1,0.075])
            axthird = fig.add_axes([0.9,0.05,0.1,0.075])
            #ax.scatter(state1[0], state1[1], color="blue")
            #ax.scatter(state2[0], state2[1], color="red")
            #ax.scatter(goal[0], goal[1], marker='o', s=100, color='seagreen')
            bfirst = Button(axfirst, 'Blue')
            bfirst.color = 'royalblue'
            bfirst.hovercolor = 'blue'
            bfirst.on_clicked(callback.first)
            bsecond = Button(axsecond, 'Red')
            bsecond.color = 'salmon'
            bsecond.hovercolor = 'red'
            bsecond.on_clicked(callback.second)
            bthird = Button(axthird, 'black')
            bthird.color = 'black'
            bthird.hovercolor = 'black'
            bthird.on_clicked(callback.dontknow)
            plt.show(block=False)
            t=0
            while curr_label == NOT_ANSWERED and t < self.wait_time:
                plt.pause(1)
                t+=1
            plt.close()

            return curr_label
    
    def generate_pref_from_human(self, goal_states):
        observations_1, _, _, _, _, _ = self.replay_buffer.sample_batch_last_steps(self.goal_selector_num_samples) # TODO: add
        observations_2, _, _, _, _, _ = self.replay_buffer.sample_batch_last_steps(self.goal_selector_num_samples) # TODO: add
   
        goals = []
        labels = []
        achieved_state_1 = []
        achieved_state_2 = []

        num_goals = len(goal_states)
        for state_1, state_2 in zip(observations_1, observations_2):
            goal_idx = np.random.randint(0, len(goal_states)) 
            goal = self.env.extract_goal(goal_states[goal_idx])
            if self.human_data_info is not None and self.human_data_index < len(self.human_data_info['state_1']):
                # TODO READ DATA
                state_1 = self.human_data_info['state_1'][self.human_data_index]
                state_2 = self.human_data_info['state_2'][self.human_data_index]
                label = self.human_data_info['label'][self.human_data_index]
                goal = self.human_data_info['goal'][self.human_data_index]

                self.human_data_index += 1
            else:
                label = self.ask_human_labels(state_1, state_2, goal)
                if label == NOT_ANSWERED:
                    return None, None, None, None
                if label == DONT_KNOW:
                    continue

            label_oracle = self.oracle(state_1, state_2, goal)

            print("Correct:", label==label_oracle, "label", label, "label_oracle", label_oracle)

            labels.append(label) 

            self.num_labels_queried += 1 

            achieved_state_1.append(state_1) 
            achieved_state_2.append(state_2) 
            goals.append(goal)

            # dump data
            self.dict_labels['state_1'].append(state_1)
            self.dict_labels['state_2'].append(state_2)
            self.dict_labels['label'].append(label)
            self.dict_labels['goal'].append(goal)
            with open(f'human_dataset_{self.dt_string}.pickle', 'wb') as handle:
                pickle.dump(self.dict_labels, handle)

        achieved_state_1 = np.array(achieved_state_1)
        achieved_state_2 = np.array(achieved_state_2)
        goals = np.array(goals)
        labels = np.array(labels)
        
        return achieved_state_1, achieved_state_2, goals, labels # TODO: check ordering

    # TODO: this is not working too well witht the shaped distances
    def generate_pref_labels(self, goal_states):
        print("label from last k steps", self.label_from_last_k_steps)
        observations_1, _ = self.replay_buffer.sample_obs_last_steps(self.goal_selector_num_samples, k=self.label_from_last_k_steps, last_k_trajectories=self.label_from_last_k_trajectories) # TODO: add
        observations_2, _ = self.replay_buffer.sample_obs_last_steps(self.goal_selector_num_samples, k=self.label_from_last_k_steps, last_k_trajectories=self.label_from_last_k_trajectories) # TODO: add
   
        goals = [] 
        labels = []
        achieved_state_1 = []
        achieved_state_2 = []

        num_goals = len(goal_states)
        for state_1, state_2 in zip(observations_1, observations_2):
            goal_idx = np.random.randint(0, len(goal_states)) 
            goal = self.env.extract_goal(goal_states[goal_idx])
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

    def generate_pref_labels_from_images(self, goal_states, goal_images):
        observations_1, img_obs1, _ = self.replay_buffer.sample_obs_last_steps(self.goal_selector_num_samples, k=self.label_from_last_k_steps, last_k_trajectories=self.label_from_last_k_trajectories)
        observations_2, img_obs2, _ = self.replay_buffer.sample_obs_last_steps(self.goal_selector_num_samples, k=self.label_from_last_k_steps, last_k_trajectories=self.label_from_last_k_trajectories)

        goals = []
        img_goals = []
        labels = []
        achieved_state_1 = []
        achieved_state_2 = []

        for state_1, state_2 in zip(observations_1, observations_2):
            goal_idx = np.random.randint(0, len(goal_states)) 
            goal = self.env.extract_goal(goal_states[goal_idx])
            labels.append(self.oracle(state_1, state_2, goal)) 

            self.num_labels_queried += 1 

            achieved_state_1.append(state_1) 
            achieved_state_2.append(state_2) 
            goals.append(goal)
            img_goals.append(goal_images[goal_idx])

        achieved_state_1 = np.array(achieved_state_1)
        achieved_state_2 = np.array(achieved_state_2)
        goals = np.array(goals)
        img_goals = np.array(img_goals)
        labels = np.array(labels)
        
        return achieved_state_1, achieved_state_2, goals, labels, img_obs1, img_obs2, img_goals
    
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
        conditional_nll_list = []
        if self.continuous_action_space:
            conditional_nll = self.policy.loss_regression(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
            for i in range(len(self.policy_list)):
                conditional_nll_list.append(self.policy_list[i].loss_regression(observations_torch, goals_torch, actions_torch, horizon=horizons_torch))
        else:
            conditional_nll = self.policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
            for i in range(len(self.policy_list)):
                conditional_nll_list.append(self.policy_list[i].nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch))
        nll = conditional_nll
        if self.weighted_sl:
            return torch.mean(nll * weights_torch), [torch.mean(nll * weights_torch) for nll in conditional_nll_list]
        else:
            return torch.mean(nll), [torch.mean(nll) for nll in conditional_nll_list]

    def states_close(self, state, goal):
        if self.env_name == "complex_maze":
            return np.linalg.norm(self.env.observation(state)[:2]-goal[:2]) < self.stopped_thresh
        if self.env_name == "ravens_pick_or_place":
            return self.env.states_close(state, goal)
            
        if self.env_name == "kitchenSeq":
            obs = self.env.observation(state)
            return np.linalg.norm(obs[-3:] - goal[-3:] ) < self.stopped_thresh and np.linalg.norm(obs[:3]-goal[:3])

        return  np.linalg.norm(self.env.observation(state) - goal) < self.stopped_thresh

    def traj_stopped(self, states):
        if len(states) < self.remove_last_k_steps:
            return False

        state1 = states[-self.remove_last_k_steps]
        final_state = states[-1]

        return np.linalg.norm(state1-final_state) < self.stopped_thresh

    def create_video(self, images, video_filename):
        images = np.array(images).astype(np.uint8)

        images = images.transpose(0,3,1,2)
        
        if 'eval' in video_filename:
            wandb.log({"eval_video_trajectories":wandb.Video(images, fps=10)})
        else:
            wandb.log({"video_trajectories":wandb.Video(images, fps=10)})
    
    def goals_too_close_new(self, goal1, goal2):
        """
        Returns whether the given goals are too close from eah other.
        If we are using images to compare states then both inputs should be images,
        otherwise, they should be numpy arrays.
        """

        if self.use_images_in_stopping_criteria:
            return self.classified_similar(goal1, goal2, True)

        return np.linalg.norm(goal1 - goal2) < self.goal_threshold
    
    def goals_too_close(self, goal1, goal2):
        return np.linalg.norm(goal1-goal2) < self.goal_threshold
    
    def get_goal_to_rollout_new(self, goal):
        goal_image = None
        desired_goal_image = None
        reached_goal_image = None
        actions_rollout = None

        if goal is None:
            goal_state = self.env.sample_goal()
            desired_goal_state = goal_state.copy()
            desired_goal = self.env.extract_goal(goal_state.copy())

            commanded_goal_state = goal_state.copy()
            commanded_goal = self.env.extract_goal(goal_state.copy())
            goal = commanded_goal

            if self.use_images_in_reward_model or self.use_images_in_policy or self.use_images_in_stopping_criteria:
                # get goal image
                desired_goal_image = self.generate_image(goal)
                goal_image = desired_goal_image 

            # Get closest achieved state
            # TODO: this might be too much human querying, except if we use the reward model
            if self.replay_buffer.current_buffer_size > 0 and self.train_with_preferences and np.random.random() > self.desired_goal_sampling_freq:
                if self.full_iters % self.sample_new_goal_freq == 0 or self.previous_goal is None:
                    if self.use_images_in_policy:
                        goal, _, reached_goal_image = self.get_closest_achieved_state_new([commanded_goal], device=self.device,)
                    else:
                        goal = self.get_closest_achieved_state([commanded_goal], device=self.device,)
                        
                    goal = goal[0]

                    if self.use_images_in_reward_model or self.use_images_in_policy or self.use_images_in_stopping_criteria:
                        reached_goal_image = reached_goal_image[0]

                    self.previous_goal = goal
                    self.previous_goal_image = reached_goal_image
                else:
                    goal = self.previous_goal
                    reached_goal_image = self.previous_goal_image

                goals_are_too_close = False # np.linalg.norm(commanded_goal - goal) < self.goal_threshold:
                if self.command_goal_if_too_close:
                    if self.use_images_in_stopping_criteria:
                        goals_are_too_close = self.goals_too_close(reached_goal_image, goal_image)
                    else:
                        goals_are_too_close = self.goals_too_close(commanded_goal, goal)

                if goals_are_too_close:
                    goal = commanded_goal
                    print("Goals too close, preferences disabled")
                else:
                    commanded_goal = goal.copy()
                    goal_image = reached_goal_image
                    # print("Using preferences")
        else:
            # TODO: URGENT should fix this
            commanded_goal = goal.copy()
            desired_goal = goal.copy()
            commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])
            desired_goal_state = commanded_goal_state.copy()

            # We assume that the goal is equal to the sampled goal for ravens
            if self.use_images_in_reward_model or self.use_images_in_policy or self.use_images_in_stopping_criteria:
                # get goal image
                goal_image = self.generate_image(goal)
                desired_goal_image = goal_image
            
        commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])
        return goal, desired_goal, commanded_goal, desired_goal_state, commanded_goal_state, None, goal_image, desired_goal_image
    
    def get_goal_to_rollout(self, goal, config='unimodal', is_eval=False):
        actions_rollout = None
        if goal is None:
            #print("i")
            
            if is_eval:
                goal_state = self.eval_env.sample_goal()
            else:
                goal_state = self.env.sample_goal()

            desired_goal_state = goal_state.copy()
            desired_goal = self.env.extract_goal(goal_state.copy())
            #print("goal state", goal_state)
            commanded_goal_state = goal_state.copy()
            commanded_goal = self.env.extract_goal(goal_state.copy())

            # Get closest achieved state
            # TODO: this might be too much human querying, except if we use the reward model
            if self.train_with_preferences and np.random.random() > self.desired_goal_sampling_freq:
                if self.full_iters % self.sample_new_goal_freq == 0 or self.previous_goal is None:
                    if self.replay_buffer.current_buffer_size > 0:
                        goal = self.get_closest_achieved_state([commanded_goal], device=self.device,)
                        goal = goal[0]
                        self.previous_goal = goal
                    else:
                        goal = commanded_goal.copy()
                else:
                    goal = self.previous_goal

                print("goal", goal)
                #print(f"goal {goal}, commanded_goal {commanded_goal}")
                if self.command_goal_if_too_close and self.goals_too_close(commanded_goal, goal): #np.linalg.norm(commanded_goal - goal) < self.goal_threshold:
                    goal = commanded_goal
                    print("Goals too close, prefrences disabled")
                else:
                    commanded_goal = goal.copy()
                    print("Using preferences")
            else:
                goal = commanded_goal

        else:
            # TODO: URGENT should fix this
            commanded_goal = goal.copy()
            desired_goal = goal.copy()
            commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])
            desired_goal_state = commanded_goal_state.copy()
            
        commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])

        return goal, desired_goal, commanded_goal, desired_goal_state, commanded_goal_state, None

    def sample_trajectory(self, goal= None, greedy=False, starting_exploration=False,  save_video_trajectory=False, video_filename='traj_0'):
        is_eval = 'eval' in video_filename
        
        if self.use_images_in_policy:
            goal, desired_goal, new_commanded_goal, desired_goal_state, commanded_goal_state, actions_rollout, goal_image, desired_goal_image = self.get_goal_to_rollout_new(goal, is_eval)
        else:
            goal, desired_goal, new_commanded_goal, desired_goal_state, commanded_goal_state, actions_rollout = self.get_goal_to_rollout(goal, is_eval)
            
        # if np.linalg.norm(desired_goal - [0.25, -0.25]) < 0.1 and not is_eval:

        desired_goal = goal.copy()
        print("sampled goal: ", goal)
        states = []
        observations = []
        img_obs = []
        actions = []
        goals_commanded_state = []
        video = []
        poses = {}

        if is_eval:
            env = self.eval_env
            state = env.reset(poses)
        else:
            env = self.env
            if self.reset_free:
                if self.env_name == 'locobot_mujoco':
                    state = self.env.base_env.get_state()
                    state = np.array([state[0], state[1], state[2],
                                      state[0], state[1], state[2],
                                      state[0], state[1], state[2]])
                else:
                    state = env.get_obs()
            else:
                state = env.reset(poses)

        stopped = False
        t_stopped = self.max_path_length
        t = 0
        
        curr_max = self.curr_frontier

        if starting_exploration:
            t_stopped = 0
            stopped = True


        reached = False
        previous_action = None
        print("curr_max", curr_max, self.full_iters)
        reachable_command_goal = new_commanded_goal
        reachable_commanded_goal_state = np.concatenate([reachable_command_goal.copy(), reachable_command_goal.copy(), reachable_command_goal.copy()])
        while t < curr_max: #self.curr_frontier: #self.max_path_length:
            if (curr_max - t == self.explore_length) and not stopped:
                stopped = True
                t_stopped = t
                print("Stopped to explore", t)

            if self.render:
                env.render()

            if save_video_trajectory and not self.not_save_videos: #and False: # TODO: remove
                video.append(env.render_image())
            
            if not self.use_images_in_policy:
                if t - t_stopped  > self.explore_length :
                    break

            if stopped and is_eval:
                t = curr_max

            if stopped and self.explore_length <= t - t_stopped:
                print("Stopped exploring", t)
                break
                
            states.append(state)

            observation = env.observation(state)
            observations.append(observation)

            observation_image = self.env.render_image()
            observation_image = cv2.resize(observation_image, (self.input_image_size,self.input_image_size))
            observation_image = observation_image.reshape(1, 3, self.input_image_size,self.input_image_size)
            img_obs.append(observation_image)

            if np.random.random() < self.epsilon_greedy_rollout and not stopped:
                action = self.policy.act_vectorized(observation[None], reachable_command_goal[None], greedy=True, noise=0)[0][0]
            else:
                if previous_action is not None and np.random.random() < self.repeat_previous_action_prob:
                    action = previous_action
                else:
                    action = np.random.randint(env.action_space.n)        

            #print("Added action is ", action)
            actions.append(action)
            states_are_close = self.states_close(states[-1], goal)
            trajectory_stopped = self.traj_stopped(states)
            goals_commanded_state.append(reachable_commanded_goal_state)

            if not is_eval and not stopped and (states_are_close and self.check_if_close or self.check_if_stopped and trajectory_stopped):#  or self.traj_stopped(states)):
                reached = True #self.states_close(states[-1], goal) 
                stopped = True

                t_stopped = t

                print("Stopped at ", t)

                if trajectory_stopped:
                    print("Trajectory got stuck")
                if states_are_close:
                    print("states are close")
                    print(env.observation(states[-1]), goal)
                    wandb.log({"StatesClose":np.linalg.norm(env.observation(states[-1])-goal)})
                if trajectory_stopped:
                    states = states[:-self.remove_last_k_steps]# TODO: hardcoded
                    actions = actions[:-self.remove_last_k_steps]
                    t-=self.remove_last_k_steps
                

            wandb.log({'Deterministic/puck_distanceprev':np.linalg.norm(env.observation(state)[2:] - goal[2:])})
            wandb.log({'Deterministic/endeff_distanceprev':np.linalg.norm(env.observation(state)[:2] - goal[:2])})
            state, _, _, _ = env.step(action)
            previous_action = action
            t+=1

            if "ravens" in self.env_name:
                wandb.log({"Control/CommandedActionDiff_state": np.linalg.norm(env.observation(state)[:2]- action)})
            
        final_dist = self.env_distance(states[-1], desired_goal)
        final_dist_commanded = self.env_distance(states[-1], goal)
            
        if save_video_trajectory and not self.not_save_videos:
            self.create_video(video, f"{video_filename}_{final_dist}")

        return np.stack(states), np.array(actions), np.array(img_obs), np.array(goals_commanded_state), desired_goal_state, reached
    
    def compute_reachable_set_cont(self, observation, horizon, greedy, thres=0.1):
        centers = self.reachable_set.get_reachable_set_center_state([(i, j) for i in range(50) for j in range(50)])
        var_list = []
        for c in centers:
            if self.env_name == 'pusher_hard':
                dist_vec = self.env.compute_shaped_distance(observation, c) 
            else:
                dist_vec = c - observation
            if np.linalg.norm(dist_vec) < 0.5:
                for i in range(len(self.policy_list)):
                    probs_list = []
                    _, probs = self.policy_list[i].act_vectorized(observation[None], c[None], horizon=horizon[None], greedy=greedy, noise=0)
                    probs_list.append(probs.to('cpu').detach().numpy())
                var_list.append((c, np.var(probs_list)))
        reachable_set = []
        for i, c in enumerate(var_list):
            if var_list[i][1] < thres:
                reachable_set.append(var_list[i][0])
        return reachable_set

    def filter_close_candidates(self, curr_state, candidates, thresh):
        filter = np.linalg.norm(self.env.observation(curr_state) - self.env.observation(candidates), axis=-1) > thresh
        return  candidates[filter]
    
    def sample_trajectory_random(self, goal= None, greedy=False, noise=0, with_preferences=False, exploration_enabled=False,save_video_trajectory=False, video_filename='traj_0', eval_traj=False):
        if goal is None:
            goal_state = self.env.sample_goal()
            desired_goal_state = goal_state.copy()
            desired_goal = self.env.extract_goal(goal_state.copy())
            commanded_goal_state = goal_state.copy()
            commanded_goal = self.env.extract_goal(goal_state.copy())

            # Get closest achieved state
            # TODO: this might be too much human querying, except if we use the reward model
            if with_preferences:
                goal = self.get_closest_achieved_state([commanded_goal], device=self.device,)[0]
                print("goal", goal)
                #print(f"goal {goal}, commanded_goal {commanded_goal}")
                if np.linalg.norm(commanded_goal - goal) < self.goal_threshold:
                    goal = commanded_goal
                    exploration_enabled = False
                    print("Goals too close, preferences disabled")
                else:
                    commanded_goal = goal.copy()
                    print("Using preferences")
            else:
                goal = commanded_goal

        else:
            # TODO: URGENT should fix this
            commanded_goal = goal.copy()
            desired_goal = goal.copy()
            commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])
            desired_goal_state = commanded_goal_state.copy()

        commanded_goal_state = np.concatenate([goal.copy(), goal.copy(), goal.copy()])

        states = []
        observations = []
        img_obs = []
        actions = []
        video = []

        # for pointmass rooms
        if self.env_name == 'pointmass':
            room_num = random.randint(1, 4)
            if room_num == 1:
                x = np.random.uniform(low=0, high=1, size=(1,))
                y = np.random.uniform(low=-1, high=0, size=(1,))
                state = np.array([int(x), int(y)])
            elif room_num == 2:
                x = np.random.uniform(low=-1, high=0, size=(1,))
                y = np.random.uniform(low=-1, high=0, size=(1,))
                state = np.array([int(x), int(y)])
            elif room_num == 3:
                x = np.random.uniform(low=-1, high=0, size=(1,))
                y = np.random.uniform(low=0, high=1, size=(1,))
                state = np.array([int(x), int(y)])
            elif room_num == 4:
                x = np.random.uniform(low=0, high=1, size=(1,))
                y = np.random.uniform(low=0, high=1, size=(1,))
                state = np.array([int(x), int(y)])
        
        # #for locobot
        if self.env_name == 'locobot_mujoco':
            # state = np.array([2.25, -1.5, 0.5]) #for locobot
            x, y = random.choice([[2.25, -1.5], [2.25, 0], [2.25, 1.5], [0, 1.5], [-1.5, 1.5], [-2, 1.5], [2.25, -1.5], [0, -1.5], [-2.25, -1.5]])
            z = np.random.uniform(low=-1, high=1, size=(1,))
            state = np.array([x, y, z])
            state = {'observation': state, 'achieved_goal': state,  'state_achieved_goal': state}
            # state = self.env._base_obs_to_state(state)
        

        # for pusher
        if self.env_name == 'pusher_hard':
            state = np.array(random.choice([
                                    [-0.17,  0.45, -0.15,  0.55],
                                    # [-0.12,  0.55, -0.1,  0.55],
                                    # [-0.02,  0.55, 0,  0.55],
                                    # [0.08,  0.55, 0.1,  0.55],
                                    # [0.18,  0.55, 0.2,  0.55]
                                    ]))

        # state = np.array([0, 0, 0, 0])
        state = {'observation': state, 'achieved_goal': state,  'state_achieved_goal': state}
        state = self.env._base_obs_to_state(state)
        

        t = 0
            
        while t < self.max_path_length:
            if self.render:
                self.env.render()

            states.append(state)

            observation = self.env.observation(state)
            observations.append(observation)

            observation_image = self.env.render_image()
            observation_image = cv2.resize(observation_image, (self.input_image_size,self.input_image_size))
            observation_image = observation_image.reshape(1, 3, self.input_image_size,self.input_image_size)
            img_obs.append(observation_image)


            horizon = np.arange(self.max_path_length) >= (self.max_path_length - 1 - t) # Temperature encoding of horizon
            action = np.random.randint(self.env.action_space.n)
            
            actions.append(action)
            state, _, _, _ = self.env.step(action)
            t+=1

        if self.use_reachable_set:
            # self.reachable_set.grid_update_traj(np.stack(observations))
            pass

        commanded_goal_state = states[-1]
        return np.stack(states), np.array(actions), np.array(img_obs), commanded_goal_state, desired_goal_state
    
    def take_policy_step_new(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()
        for _ in range(self.n_accumulations):
            if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                observations, actions, img_obs, goals, img_goal, _, horizons, weights = buffer.sample_batch(self.batch_size)
            else:
                observations, actions, goals, _, horizons, weights = buffer.sample_batch(self.batch_size)

            if self.use_images_in_policy:
                loss = self.loss_fn(img_obs, img_goal, actions, horizons, weights)
            else:
                loss = self.loss_fn(observations, goals, actions, horizons, weights)

            loss.backward()
            avg_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip)
        self.policy_optimizer.step()

        return avg_loss / self.n_accumulations
    
    def take_policy_step(self, buffer=None):
        if buffer is None:
            buffer = self.replay_buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()
        for i in range(len(self.policy_list)):
            self.policy_optimizer_list[i].zero_grad()
        for _ in range(self.n_accumulations):
            if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                observations, actions, img_obs, goals, img_goal, _, horizons, weights = buffer.sample_batch(self.batch_size)
            else:
                observations, actions, goals, _, horizons, weights = buffer.sample_batch(self.batch_size)
            if self.use_images_in_policy:
                loss, loss_list = self.loss_fn(img_obs, img_goal, actions, horizons, weights)
            else:
                loss, loss_list = self.loss_fn(observations, goals, actions, horizons, weights)

            loss.backward()
            for i in range(len(loss_list)):
                loss_list[i].backward()
            avg_loss += ptu.to_numpy(loss.cpu())
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip)
        for i in range(len(self.policy_list)):
            torch.nn.utils.clip_grad_norm_(self.policy_list[i].parameters(), self.clip)
        self.policy_optimizer.step()
        for i in range(len(self.policy_list)):
            self.policy_optimizer_list[i].step()

        return avg_loss / self.n_accumulations

    def validation_loss(self, buffer=None):

        if buffer is None:
            buffer = self.validation_buffer

        if buffer is None or buffer.current_buffer_size == 0:
            return 0, 0

        avg_loss = 0
        avg_goal_selector_loss = 0
        for _ in range(self.n_accumulations):
            if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                observations, actions, img_obs, goals, img_goal, _, horizons, weights = buffer.sample_batch(self.batch_size)
            else:
                observations, actions, goals, _, horizons, weights = buffer.sample_batch(self.batch_size)

            if self.use_images_in_policy:
                loss, loss_list = self.loss_fn(img_obs, img_goal, actions, horizons, weights)
            else:
                loss, loss_list = self.loss_fn(observations, goals, actions, horizons, weights)

            # observations, actions, goals, lengths, horizons, weights = buffer.sample_batch(self.batch_size)
            # loss, _ = self.loss_fn(observations, goals, actions, horizons, weights)
            # eval_data = self.generate_pref_labels(observations, actions, [goals], extract=False)
            #print("eval data", eval_data)
            # loss_goal_selector =self.eval_goal_selector(eval_data)
            # TODO: implement eval loss
            loss_goal_selector = torch.tensor(0)
            avg_loss += ptu.to_numpy(loss)
            avg_goal_selector_loss += ptu.to_numpy(loss_goal_selector)

        return avg_loss / self.n_accumulations, avg_goal_selector_loss / self.n_accumulations

    def pretrain_demos(self, demo_replay_buffer=None, demo_validation_replay_buffer=None, demo_train_steps=0):
        if demo_replay_buffer is None:
            return

        self.policy.train()
        for i in range(len(self.policy_list)):
            self.policy_list[i].train()
        running_loss = None
        running_validation_loss = None
        losses = []
        val_losses = []
        with tqdm.trange(self.demo_pretrain_epochs) as looper:
            for _ in looper:
                loss = self.take_policy_step(buffer=demo_replay_buffer)
                validation_loss, goal_selector_val_loss = self.validation_loss(buffer=demo_validation_replay_buffer)

                if running_loss is None:
                    running_loss = loss
                else:
                    running_loss = 0.99 * running_loss + 0.01 * loss
                if running_validation_loss is None:
                    running_validation_loss = validation_loss
                else:
                    running_validation_loss = 0.99 * running_validation_loss + 0.01 * validation_loss

                looper.set_description('Loss: %.03f curr Loss: %.03f'%(running_loss, loss))
                losses.append(loss)
                val_losses.append(validation_loss)

        plt.plot(losses)
        plt.plot(val_losses)
        plt.savefig("loss.png")
        
    def test_goal_selector(self, itr, save=True, size=50):
        if "ravens" in self.env_name or "locobot" in self.env_name:
            return
        goal = self.env.sample_goal()#np.random.uniform(-0.5, 0.5, size=(2,))
        goal_pos =  self.env.extract_goal(goal)
        #goal_pos = goal
        #TODO: remove
        #goal_pos = np.array([0.3,0.3])
        if "maze" in self.env_name:
            #states = np.concatenate([np.random.uniform( size=(10000, 2)), np.random.uniform(-1,1, size=(10000,2))], axis=1)
            pos = np.meshgrid(np.linspace(0, 11.5,size), np.linspace(0, 12.5,size))
            vels = np.meshgrid(np.random.uniform(-1,1, size=(size)),np.zeros((size)))
            
            pos = np.array(pos).reshape(2,-1).T
            vels = np.array(vels).reshape(2,-1).T
            states = np.concatenate([pos, vels], axis=-1)
            goals = np.repeat(goal_pos[None], size*size, axis=0)

        elif "pusher" in self.env_name:
            pos = np.meshgrid(np.linspace(-0.4, 0.4,size), np.linspace(0.4, 0.8,size))
            puck_pos = pos.copy()
            
            pos = np.array(pos).reshape(2,-1).T
            puck_pos = np.array(puck_pos).reshape(2,-1).T

            states = np.concatenate([pos, puck_pos], axis=-1)
            goals = np.repeat(goal_pos[None], size*size, axis=0)
        else:
            # goal_pos = np.array([0.4, 0.3])
            states = np.meshgrid(np.linspace(-.6,.6,200), np.linspace(-.6,.6,200))
            states = np.array(states).reshape(2,-1).T
            goals = np.repeat(goal_pos[None], 200*200, axis=0)

        
        states_t = torch.Tensor(states).cuda()
        goals_t = torch.Tensor(goals).cuda()
        r_val = self.goal_selector(states_t, goals_t)
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
        elif self.env_name == "pusher_hard":
            from matplotlib.patches import Rectangle
            center1 = np.array([-0.1, 0.5])
            center2 = np.array([0.1, 0.7])
            dim = np.array([0.025,0.15])*2
            #plt.Rectangle(center1 - dim/2, dim[0], dim[1], edgecolor='black', facecolor="black", lw=0)
            plt.gca().add_patch(Rectangle((center1[0]-dim[0]/2, center1[1]-dim[1]/2) , dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
            plt.gca().add_patch(Rectangle((center2[0]-dim[0]/2, center2[1]-dim[1]/2), dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
            center_board = np.array([0,0.6])
            dim_board = np.array([0.4, 0.2])*2

            plt.xlim((center_board[0]-dim_board[0]/2, center_board[0]+dim_board[0]/2))
            plt.ylim((center_board[1]-dim_board[1]/2, center_board[1]+dim_board[1]/2))
            plt.gca().set_aspect('equal', adjustable='box')
        else:
            if self.env_name == "complex_maze":
                self.display_wall_maze()
            else:
                self.display_wall()
            plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')

        filename = self.env_name+"/goal_selector_test/test_goal_selector_itr%d.png"%itr
        from PIL import Image
        
        wandb.log({"rewardmodel": wandb.Image(plt)})

        r_val = self.oracle_model(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        #self.display_wall(plt)
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)
        if self.env_name == "pusher":
            self.display_wall_pusher()

            plt.scatter(goal_pos[2], goal_pos[3], marker='o', s=100, color='black')
        elif self.env_name == "pusher_hard":
            from matplotlib.patches import Rectangle
            center1 = np.array([-0.1, 0.5])
            center2 = np.array([0.1, 0.7])
            dim = np.array([0.025,0.15])*2
            #plt.Rectangle(center1 - dim/2, dim[0], dim[1], edgecolor='black', facecolor="black", lw=0)
            plt.gca().add_patch(Rectangle((center1[0]-dim[0]/2, center1[1]-dim[1]/2) , dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
            plt.gca().add_patch(Rectangle((center2[0]-dim[0]/2, center2[1]-dim[1]/2), dim[0], dim[1], edgecolor='black', facecolor="none", lw=1))
            center_board = np.array([0,0.6])
            dim_board = np.array([0.4, 0.2])*2

            plt.xlim((center_board[0]-dim_board[0]/2, center_board[0]+dim_board[0]/2))
            plt.ylim((center_board[1]-dim_board[1]/2, center_board[1]+dim_board[1]/2))
            plt.gca().set_aspect('equal', adjustable='box')
        else:
            if self.env_name == "complex_maze":
                self.display_wall_maze()
            else:
                self.display_wall()
            plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')
        wandb.log({"oraclemodel": wandb.Image(plt)})
        
        # plt.savefig(self.env_name+"/goal_selector_test/test_oracle_itr%d.png"%itr)
        
    def plot_visit_freq(self, itr):
        pos = np.random.uniform(-0.5, 0.5, size=(2,))
        #goals = np.repeat(goal_pos[None], 10000, axis=0)
        #states = np.random.uniform(-0.5, 0.5, size=(10000, 2))
        #states_t = torch.Tensor(states).cuda()
        #goals_t = torch.Tensor(goals).cuda()
        #r_val = self.goal_selector(states_t, goals_t, goals_t)
        r_val = np.zeros(pos.shape)
        #r_val = r_val.cpu().detach().numpy()
        os.makedirs('goal_selector_test', exist_ok=True)
        plt.clf()
        plt.cla()
        self.display_wall()
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)
        plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')
        plt.savefig("goal_selector_test/test_goal_selector_itr%d.png"%itr)

    def full_grid_evaluation(self, itr):
        grid_size = 20
        goals = np.linspace(-0.6, 0.6, grid_size)
        distances = np.zeros((grid_size,grid_size))

        for x in range(len(goals)):
            for y in range(len(goals)):
                goal = np.array([goals[x],goals[y]])

                if self.use_images_in_policy:
                    states, actions, img_obs, goal_state, _, _ = self.sample_trajectory(goal=goal, greedy =True)
                else:
                    states, actions, _, goal_state, _, _ = self.sample_trajectory(goal=goal, greedy =True)
                distance =  np.linalg.norm(goal - states[-1][-2:])
                distances[x,y]= distance 

        plot = sns.heatmap(distances, xticklabels=goals, yticklabels=goals)
        fig = plot.get_figure()
        fig.savefig(f'heatmap_performance/eval_{itr}.png')
        plot = sns.heatmap(distances < self.goal_threshold, xticklabels=goals, yticklabels=goals)
        fig = plot.get_figure()
        fig.savefig(f'heatmap_accuracy/eval_{itr}.png')
    
    def get_distances(self, state, goal):
        obs = self.env.observation(state)

        if not isinstance(self.env.wrapped_env, KitchenSequentialGoalEnv):
            return None, None, None, None, None, None

        per_pos_distance, per_obj_distance = self.env.success_distance(obs)
        distance_to_slide = per_pos_distance['slide_cabinet']
        distance_to_hinge = per_pos_distance['hinge_cabinet']
        distance_to_microwave = per_pos_distance['microwave']
        distance_joint_slide = per_obj_distance['slide_cabinet']
        distance_joint_hinge = per_obj_distance['hinge_cabinet']
        distance_microwave = per_obj_distance['microwave']

        return distance_to_slide, distance_to_hinge, distance_to_microwave, distance_joint_slide, distance_joint_hinge, distance_microwave

    def plot_trajectories(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        if isinstance(self.env.wrapped_env, PointmassGoalEnv):
            return self.plot_trajectories_rooms(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "pointmass/" + filename)
        if isinstance(self.env.wrapped_env, SawyerPushGoalEnv):
            return self.plot_trajectories_pusher(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "pusher/" + filename)
        if isinstance(self.env.wrapped_env, SawyerHardPushGoalEnv):
            return self.plot_trajectories_pusher_hard(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "pusher_hard/" + filename)
        if self.env_name == "complex_maze":
            #if 'train' in filename:
            #    self.plot_trajectories_complex_maze(self.replay_buffer._states.copy(), traj_accumulated_goal_states, extract, "complex_maze/"+f"train_states_preferences/replay_buffer{self.total_timesteps}.png")

            return self.plot_trajectories_complex_maze(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "complex_maze/"+filename)
        if "ravens" in self.env_name:
            return self.plot_trajectories_ravens(traj_accumulated_states.copy(), traj_accumulated_goal_states.copy(), extract, "complex_maze/"+filename)

    def display_wall_maze(self):
        from matplotlib.patches import Rectangle

        maze_arr = self.env.wrapped_env.base_env.maze_arr
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
            plt.plot(self.env.observation(states_plot[j ])[:,0], self.env.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
        from PIL import Image
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
            plt.plot(self.env.observation(states_plot[j ])[:,0], self.env.observation(states_plot[j])[:, 1], color=color, zorder = -1)
            
            plt.scatter(traj_accumulated_goal_states[j][0],
                    traj_accumulated_goal_states[j][1], marker='o', s=20, color=color, zorder=1)
            box_position_end = self.env.observation(states_plot[j])[-1,3:]
            plt.scatter(box_position_end[0],
                        box_position_end[1], marker='s', s=20, color=color)
            if len(box_position_end) > 2:
                plt.scatter(box_position_end[2],
                    box_position_end[3], marker='^', s=20, color=color)
            if len(box_position_end) > 4:
                plt.scatter(box_position_end[4],
                    box_position_end[5], marker='D', s=20, color=color)
                    
        box_position = self.env.observation(states_plot[j])[0,3:]
        
        goal_position = self.env.sample_goal()
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
        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def plot_trajectories_rooms(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        
        colors = sns.color_palette('hls', (len(traj_accumulated_states)))
        for j in range(len(traj_accumulated_states)):
            color = colors[j]
            plt.plot(self.env.observation(traj_accumulated_states[j ])[:,0], self.env.observation(traj_accumulated_states[j])[:, 1], color=color, zorder = -1)
            #if 'train_states_preferences' in filename:
            #    color = 'black'
            
            plt.scatter(traj_accumulated_goal_states[j][:,-2],
                    traj_accumulated_goal_states[j][:,-1], marker='o', s=20, color=color, zorder=1)
        
        plt.savefig(filename)

        from PIL import Image
        plt.savefig(filename)
        
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def plot_trajectories_pusher(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        plt.cla()
        self.display_wall_pusher()
        #if extract:

        states_plot =  self.env._extract_sgoal(traj_accumulated_states)
        traj_accumulated_goal_states =  self.env._extract_sgoal(traj_accumulated_goal_states)

        #else:
        #    states_plot = traj_accumulated_states
        #shutil.rmtree("train_states_preferences")
        colors = sns.color_palette('hls', (states_plot.shape[0]))
        for j in range(states_plot.shape[0]):
            color = colors[j]
            plt.plot(states_plot[j ][:,2], states_plot[j][:, 3], color=color)
            plt.scatter(traj_accumulated_goal_states[j][:,2],
                    traj_accumulated_goal_states[j][:,3], marker='o', s=20, color=color)
        
        from PIL import Image
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def plot_trajectories_pusher_hard(self,traj_accumulated_states, traj_accumulated_goal_states, extract=True, filename=""):
        # plot added trajectories to fake replay buffer
        plt.clf()
        plt.cla()
        self.display_wall_pusher_hard()
        #if extract:

        states_plot =  traj_accumulated_states

        #else:
        #    states_plot = traj_accumulated_states
        #shutil.rmtree("train_states_preferences")
        colors = sns.color_palette('hls', (len(states_plot)))
        for j in range(len(states_plot)):
            color = colors[j]
            plt.plot(self.env.observation(states_plot[j ])[:,0], self.env.observation(states_plot[j])[:, 1], color=color, zorder=1)

            plt.scatter(traj_accumulated_goal_states[j][:,2],
                    traj_accumulated_goal_states[j][:,3], marker='+', s=20, color=color, zorder=2)
            plt.scatter(traj_accumulated_goal_states[j][:,0],
                    traj_accumulated_goal_states[j][:,1], marker='o', s=20, color=color, zorder=2)
            plt.scatter(self.env.observation(states_plot[j ])[:,2], self.env.observation(states_plot[j])[:, 3], marker='x', s=20, color=color, zorder=1)
                    
        goals = np.array([[0.15, 0.63 , 0.15, 0.65], [-0.15,  0.45,-0.15,  0.5]])
        plt.scatter(goals[:,2],
                    goals[:,3], marker='o', s=20, color="black", zorder=2)
        if 'eval' in filename:
            wandb.log({"trajectory_eval": wandb.Image(plt)})
        else:
            wandb.log({"trajectory": wandb.Image(plt)})

    def display_collected_labels(self, state_1, state_2, goals, is_oracle=False):
        if self.env_name == "complex_maze" and not is_oracle:
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
                    plt.scatter(self.env.observation(state_1[j])[0], self.env.observation(state_1[j])[1], color=color, zorder = -1)
                else:
                    plt.scatter(self.env.observation(state_1[j])[0], self.env.observation(state_1[j])[1], color=color, zorder = -1)
                    plt.scatter(self.env.observation(state_2[j])[0], self.env.observation(state_2[j])[1], color=color, zorder = -1)
                
                if not is_oracle:
                    plt.scatter(goals[j][0],
                        goals[j][1], marker='+', s=20, color=color, zorder=1)
            if is_oracle:
                plt.scatter(goals[0],
                        goals[1], marker='+', s=20, color=color, zorder=1)
            from PIL import Image
            filename = self.env_name+f"/train_states_preferences/goal_selector_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            if is_oracle:
                wandb.log({"goal_selector_candidates": wandb.Image(plt)})
            else:
                wandb.log({"goal_selector_labels": wandb.Image(plt)})

    def display_collected_labels_complex_maze(self, state_1, state_2, goals):
            # plot added trajectories to fake replay buffer
            plt.clf()
            self.display_wall_maze()
            
            colors = sns.color_palette('hls', (state_1.shape[0]))
            for j in range(state_1.shape[0]):
                color = colors[j]
                plt.scatter(self.env.observation(state_1[j])[0], self.env.observation(state_1[j])[1], color=color, zorder = -1)
                plt.scatter(self.env.observation(state_2[j])[0], self.env.observation(state_2[j])[1], color=color, zorder = -1)
                
                plt.scatter(goals[j][0],
                        goals[j][1], marker='o', s=20, color=color, zorder=1)
            from PIL import Image
            
            filename = "complex_maze/"+f"train_states_preferences/goal_selector_labels_{self.total_timesteps}_{np.random.randint(10)}.png"
            wandb.log({"goal_selector_labels": wandb.Image(plt)})

    def display_goal_selection(self, states, goal, commanded_goal):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.test_goal_selector(-1, False)

        self.display_wall_maze()
        
        for j in range(states.shape[0]):
            plt.scatter(self.env.observation(states[j])[0], self.env.observation(states[j])[1], color="black")
            
        plt.scatter(goal[0],
                goal[1], marker='o', s=20, color="yellow", zorder=1)

        plt.scatter(commanded_goal[0],
                commanded_goal[1], marker='o', s=20, color="green", zorder=1)
        from PIL import Image
        
        filename = "complex_maze/"+f"train_states_preferences/goal_selection_candidates_{self.total_timesteps}_{np.random.randint(10)}.png"
        
        wandb.log({"goal_selector_labels_and_state": wandb.Image(plt)})

    def collect_and_train_goal_selector(self, desired_goal_states_goal_selector,total_timesteps):
        if len(desired_goal_states_goal_selector) == 0 or self.no_training_goal_selector or self.total_timesteps > self.stop_training_goal_selector_after:
            return 0, 0

        # TODO: add validation buffer
        if self.full_iters % self.display_trajectories_freq == 0 and ("maze" in self.env_name or "ravens" in self.env_name or "pusher" in self.env_name):
            self.test_goal_selector(self.total_timesteps)

        # Train reward model
        if not self.use_oracle:
            # Generate labels with preferences
            losses_goal_selector, eval_loss_goal_selector = self.train_goal_selector()

            print("Computing reward model loss ", np.mean(losses_goal_selector), "eval loss is: ", eval_loss_goal_selector)
            if self.summary_writer:
                self.summary_writer.add_scalar('Lossesgoal_selector/Train', np.mean(losses_goal_selector), total_timesteps)
            wandb.log({'Lossesgoal_selector/Train':np.mean(losses_goal_selector), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
            wandb.log({'Lossesgoal_selector/Eval':eval_loss_goal_selector, 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})

            self.train_loss_goal_selector_arr.append((np.mean(losses_goal_selector), total_timesteps))

            torch.save(self.goal_selector.state_dict(), f"checkpoint/goal_selector_model_intermediate_{self.total_timesteps}.h5")
        
        return losses_goal_selector, eval_loss_goal_selector

    def dump_data(self):
        metrics = {
            'success_ratio_eval_arr':self.success_ratio_eval_arr,
            'train_loss_arr':self.train_loss_arr,
            'distance_to_goal_eval_arr':self.distance_to_goal_eval_arr,
            'success_ratio_relabelled_arr':self.success_ratio_relabelled_arr,
            'eval_trajectories_arr':self.eval_trajectories_arr,
            'train_loss_goal_selector_arr':self.train_loss_goal_selector_arr,
            'eval_loss_arr':self.eval_loss_arr,
            'distance_to_goal_eval_relabelled':self.distance_to_goal_eval_relabelled,
        }
        with open(os.path.join(self.data_folder, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

    def dump_trajectories(self):
        
        with open(os.path.join(self.data_folder, f'eval_trajectories/traj_{self.traj_num_file}.pkl'), 'wb') as f:
            pickle.dump(self.collected_trajs_dump, f)
        self.traj_num_file +=1

        self.collected_trajs_dump = []

    def train(self):
        start_time = time.time()
        last_time = start_time

        self.full_iters = 0


        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0
        timesteps_since_reset = 0

        iteration = 0
        running_loss = None
        running_validation_loss = None
        goal_selector_running_val_loss = None

        losses_goal_selector_acc = None

        if self.pretrain_policy or self.pretrain_goal_selector:
            print("Pretraining")

            for i in range(self.num_demos//2):
                actions = np.load(f"demos/{self.demos_folder_name}/demo_{i}_actions.npy")
                states = np.load(f"demos/{self.demos_folder_name}/demo_{i}_states.npy")

                self.replay_buffer.add_trajectory(states, actions, states[-1])
                self.goal_selector_buffer.add_multiple_data_points(states,  goal=-1)


            for i in range(self.num_demos//2):
                actions = np.load(f"demos/{self.demos_folder_name}/demo_{self.offset + i}_actions.npy")
                states = np.load(f"demos/{self.demos_folder_name}/demo_{self.offset + i}_states.npy")

                self.replay_buffer.add_trajectory(states, actions, states[-1])
                self.goal_selector_buffer.add_multiple_data_points(states,  goal=-1)

        if self.pretrain_policy and self.num_demos > 0:
            self.pretrain_goal_selector_func()

            self.pretrain_demos(self.replay_buffer)
            self.evaluate_policy(self.eval_episodes, greedy=False, prefix="DemosEval")
            self.evaluate_policy(self.eval_episodes, greedy=False, prefix="Eval")

        os.makedirs('checkpoint', exist_ok=True)

        if self.display_plots:
            os.makedirs("train_states_preferences", exist_ok=True)
            os.makedirs("relabeled_states_preferences", exist_ok=True)
            os.makedirs("explore_states_trajectories", exist_ok=True)
            os.makedirs("train_states_preferences", exist_ok=True)
            #shutil.rmtree("explore_states_trajectories")
            os.makedirs("heatmap_performance", exist_ok=True)
            os.makedirs("explore_states_trajectories", exist_ok=True)
            #shutil.rmtree("heatmap_performance")
            os.makedirs("heatmap_accuracy", exist_ok=True)
            os.makedirs("heatmap_performance", exist_ok=True)
            #shutil.rmtree("heatmap_accuracy")
            os.makedirs(self.env_name+'/goal_selector_test', exist_ok=True)        
            os.makedirs("heatmap_accuracy", exist_ok=True)
            os.makedirs('preferences_distance', exist_ok=True)
            #shutil.rmtree(self.env_name+"/goal_selector_test")
            os.makedirs(self.env_name+'/goal_selector_test', exist_ok=True)        
            #shutil.rmtree("preferences_distance")
            #os.makedirs('preferences_distance', exist_ok=True)

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        os.makedirs(f'{self.env_name}', exist_ok=True)
        self.trajectories_videos_folder = f'{self.env_name}/trajectories_videos_{dt_string}'
        os.makedirs(self.trajectories_videos_folder, exist_ok=True)

        
        
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

        if logger.get_snapshot_dir() and self.log_tensorboard:
            info = self.comment
            if self.train_with_preferences:
                info+="preferences"
            info+= f"_start_policy_{self.start_policy_timesteps}"
            info+= f"_use_oracle_{self.use_oracle}"
            info+= f"_lr_{self.lr}"
            info+= f"_batch_size_{self.batch_size}"
            info+= f"_select_best_sample_size_{self.select_best_sample_size}"
            info+= f"_max_path_length_{self.max_path_length}"
            

            tensorboard_path = osp.join(logger.get_snapshot_dir(), info)

            print("tensorboard directory", tensorboard_path)
            self.summary_writer = SummaryWriter(tensorboard_path)
        else:
            print("Tensorboard failed", logger.get_snapshot_dir(), self.log_tensorboard)

        # Evaluation Code
        self.policy.eval()
        for i in range(len(self.policy_list)):
            self.policy_list[i].eval()
        if self.train_with_preferences and self.display_plots:
            #if os.path.exists(self.env_name+"/train_states_preferences"):
                #shutil.rmtree(self.env_name+"/train_states_preferences")

            os.makedirs(self.env_name+"/train_states_preferences", exist_ok=True)

            os.makedirs(self.env_name+"/plots_preferences", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_preferences")
            os.makedirs(self.env_name+"/plots_preferences", exist_ok=True)
            os.makedirs(self.env_name+"/plots_preferences_requested", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_preferences_requested")
            os.makedirs(self.env_name+"/plots_preferences_requested", exist_ok=True)
            plots_folder = "plots_preferences"
            plots_folder_requested = "plots_preferences_requested"

        elif self.display_plots:
            os.makedirs(self.env_name+"/plots", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots")
            os.makedirs(self.env_name+"/plots", exist_ok=True)
            plots_folder = self.env_name+"/plots"
            os.makedirs(self.env_name+"/plots_requested", exist_ok=True)
            #shutil.rmtree(self.env_name+"/plots_requested")
            os.makedirs(self.env_name+"/plots_requested", exist_ok=True)
            #if os.path.exists(self.env_name+"/train_states"):
                #shutil.rmtree(self.env_name+"/train_states")

            os.makedirs(self.env_name+"/train_states", exist_ok=True)

            plots_folder = "/plots"
            plots_folder_requested = "/plots_requested"
        else:
            plots_folder = ""
            plots_folder_requested = ""

        self.evaluate_policy(self.eval_episodes, total_timesteps=0, greedy=True, prefix='Eval', plots_folder=plots_folder)
        logger.record_tabular('policy loss', 0)
        logger.record_tabular('reward model train loss', 0)
        logger.record_tabular('reward model eval loss', 0)
        logger.record_tabular('timesteps', total_timesteps)
        logger.record_tabular('epoch time (s)', time.time() - last_time)
        logger.record_tabular('total time (s)', time.time() - start_time)
        last_time = time.time()
        logger.dump_tabular()
        # End Evaluation Code

        # Trajectory states being accumulated
        traj_accumulated_states = []
        traj_accumulated_actions = []
        traj_accumulated_goal_states = []
        desired_goal_states_goal_selector = []
        traj_accumulated_desired_goal_states = []
        goal_states_goal_selector = []
        traj_accumulated_states_droped = []
        traj_accumulated_goal_states_dropped = []

        
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:
                self.total_timesteps = total_timesteps
                self.full_iters +=1
                if self.save_buffer != -1 and total_timesteps > self.save_buffer:
                    self.save_buffer = -1
                    self.replay_buffer.save(self.buffer_filename)
                    self.validation_buffer.save(self.val_buffer_filename)


                #print("total timesteps", total_timesteps, "max timesteps", self.max_timesteps)
                # Interact in environmenta according to exploration strategy.
                # TODO: we can probably skip this in preferences or use it to learn a goal_selector
                if self.full_iters < self.explore_episodes:
                    #print("Sample trajectory noise")
                    # states, actions, goal_state, desired_goal_state, _ = self.sample_trajectory(starting_exploration=True)
                    if self.use_images_in_policy:
                        states, actions, img_obs, goal_state, desired_goal_state, _ = self.sample_trajectory(starting_exploration=True)
                    else:
                        states, actions, _, goal_state, desired_goal_state, _ = self.sample_trajectory(starting_exploration=True)
                    traj_accumulated_states.append(states)
                    traj_accumulated_desired_goal_states.append(desired_goal_state)
                    traj_accumulated_actions.append(actions)
                    traj_accumulated_goal_states.append(goal_state)

                    if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                        if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                            self.validation_buffer.add_trajectory(states, actions, img_obs, goal_state)
                        else:
                            self.validation_buffer.add_trajectory(states, actions, goal_state)
                    else:
                        if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                            self.replay_buffer.add_trajectory(states, actions, img_obs, goal_state)
                        else:
                            self.replay_buffer.add_trajectory(states, actions, goal_state)
                            self.goal_selector_buffer.add_multiple_data_points(states,  goal=-1)

                    # if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                    #     self.validation_buffer.add_trajectory(states, actions, goal_state)
                    # else:
                    #     self.replay_buffer.add_trajectory(states, actions, goal_state)

                elif not self.train_with_preferences:
                    assert not self.use_oracle and not self.sample_softmax
                    #print("sample trajectory greedy")
                    if self.use_images_in_policy:
                        states, actions, img_obs, goal_states, _, _ = self.sample_trajectory(greedy=False)
                    else:
                        states, actions, _, goal_states, _, _ = self.sample_trajectory(greedy=False)
                    traj_accumulated_states.append(states)
                    traj_accumulated_desired_goal_states.append(desired_goal_state)
                    traj_accumulated_actions.append(actions)
                    traj_accumulated_goal_states.append(goal_states)
                    #desired_goal_states_goal_selector.append(desired_goal_state)
                    #goal_states_goal_selector.append(goal_state)

                    if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                        if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                            self.validation_buffer.add_trajectory(states, actions, img_obs, goal_state)
                        else:
                            self.validation_buffer.add_trajectory(states, actions, goal_state)
                    else:
                        if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                            self.replay_buffer.add_trajectory(states, actions, img_obs, goal_state)
                        else:
                            self.replay_buffer.add_trajectory(states, actions, goal_state)
                            self.goal_selector_buffer.add_multiple_data_points(states,  goal=-1)

                    # if total_timesteps != 0 and self.validation_buffer is not None and np.random.rand() < 0.2:
                    #     self.validation_buffer.add_trajectory(states, actions, goal_state)
                    # else:
                    #     self.replay_buffer.add_trajectory(states, actions, goal_state)
                
                
                # Interact in environmenta according to exploration strategy.
                # TODO: should we try increasing the explore timesteps?
                if self.full_iters > self.explore_episodes:
                    print('_______________________________________________________')
                    if self.train_with_preferences and total_timesteps > self.explore_episodes:
                        save_video_trajectory = self.full_iters % self.display_trajectories_freq == 0
                        video_filename = f"traj_{total_timesteps}"
                        start = time.time()

                        if self.full_iters != 0 and self.full_iters % self.frontier_expansion_freq == 0:
                            self.curr_frontier = min(self.curr_frontier + self.frontier_expansion_rate, self.max_path_length)
                        
                        explore_states, explore_actions, explore_img_obs, explore_goal_state, desired_goal_state, stopped = self.sample_trajectory(greedy=self.greedy_before_stopping, save_video_trajectory=save_video_trajectory, video_filename=video_filename)
                        if stopped or not self.throw_trajectories_not_reaching_goal:
                            print("Sampling trajectory took", time.time() - start)
                            traj_accumulated_states.append(explore_states)
                            traj_accumulated_desired_goal_states.append(desired_goal_state)
                            traj_accumulated_actions.append(explore_actions)
                            traj_accumulated_goal_states.append(explore_goal_state)
                            desired_goal_states_goal_selector.append(desired_goal_state)
                            goal_states_goal_selector.append(explore_goal_state)

                            if self.validation_buffer is not None and np.random.rand() < 0.2:
                                if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                                    self.validation_buffer.add_trajectory(explore_states, explore_actions, explore_img_obs, explore_goal_state)
                                else:
                                    self.validation_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state)
                            else:
                                if self.use_images_in_policy or self.use_images_in_reward_model or self.use_images_in_stopping_criteria:
                                    self.replay_buffer.add_trajectory(explore_states, explore_actions, explore_img_obs, explore_goal_state)
                                else:
                                    self.replay_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state)
                                    self.goal_selector_buffer.add_multiple_data_points(explore_states,  goal=-1)

                            # if self.validation_buffer is not None and np.random.rand() < 0.2:
                            #     self.validation_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state)
                            # else:
                            #     self.replay_buffer.add_trajectory(explore_states, explore_actions, explore_goal_state)
                        else:
                            traj_accumulated_states_droped.append(explore_states)
                            traj_accumulated_goal_states_dropped.append(explore_goal_state)
                
                    if  self.train_with_preferences and self.full_iters % self.train_goal_selector_freq == 0 and total_timesteps > self.explore_episodes:
                        #print("total timesteps", total_timesteps)
                        desired_goal_states_goal_selector = np.array(desired_goal_states_goal_selector)
                        goal_states_goal_selector = np.array(goal_states_goal_selector)
                        dist = np.array([
                                self.env_distance(self.env.extract_goal(goal_states_goal_selector)[i, -1], self.env.extract_goal(desired_goal_states_goal_selector)[i])
                                for i in range(desired_goal_states_goal_selector.shape[0])
                        ])

                        if self.summary_writer:
                            #print(dist, np.mean(dist))
                            self.summary_writer.add_scalar("Preferences/DistanceCommandedToDesiredGoal", np.mean(dist), total_timesteps)
                        wandb.log({'Preferences/DistanceCommandedToDesiredGoal':np.mean(dist), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
                        
                        self.distance_to_goal_eval_arr.append((np.mean(dist), total_timesteps))
                        if self.display_plots:
                            plt.clf()
                            #self.display_wall()
                            
                            colors = sns.color_palette('hls', (goal_states_goal_selector.shape[0]))
                            for j in range(desired_goal_states_goal_selector.shape[0]):
                                color = colors[j]
                                plt.scatter(desired_goal_states_goal_selector[j][-2],
                                        desired_goal_states_goal_selector[j][-1], marker='o', s=20, color=color)
                                plt.scatter(goal_states_goal_selector[j][:,-2],
                                        goal_states_goal_selector[j][:,-1], marker='x', s=20, color=color)
                            
                            wandb.log({'GoalSelection':wandb.Image(plt)})
                            
                        # relabel and add to buffer
                        if not self.use_oracle :
                            losses_goal_selector, eval_loss_goal_selector = self.collect_and_train_goal_selector(desired_goal_states_goal_selector, total_timesteps)
                            print('---------------------------train goal selector-------------------------------------')
                    
                    desired_goal_states_goal_selector = []
                    goal_states_goal_selector = []

                    if self.classifier_model is not None and self.full_iters % self.train_classifier_freq == 0:
                        training_loss, training_acc = self.train_image_classifier(self.device, batch_size = self.classifier_batch_size)
                        eval_loss, eval_acc = self.eval_image_classifier(self.device, batch_size = self.classifier_batch_size)

                        if self.summary_writer:
                            self.summary_writer.add_scalar('LossesClassifier/Train_loss', training_loss, total_timesteps)
                            self.summary_writer.add_scalar('LossesClassifier/Eval_loss', eval_loss, total_timesteps)
                            self.summary_writer.add_scalar('LossesClassifier/Train_acc', training_acc, total_timesteps)
                            self.summary_writer.add_scalar('LossesClassifier/Eval_acc', eval_acc, total_timesteps)

                        wandb.log({'Classifier/Train_loss': training_loss, 'timesteps': total_timesteps, 'num_labels_queried': self.num_labels_queried})
                        wandb.log({'Classifier/Eval_loss': eval_loss, 'timesteps': total_timesteps, 'num_labels_queried': self.num_labels_queried})
                        wandb.log({'Classifier/Train_acc': training_acc, 'timesteps': total_timesteps, 'num_labels_queried': self.num_labels_queried})
                        wandb.log({'Classifier/Eval_acc': eval_acc, 'timesteps': total_timesteps, 'num_labels_queried': self.num_labels_queried})
                    
                if len(traj_accumulated_goal_states_dropped) != 0 and len(traj_accumulated_goal_states_dropped) % self.display_trajectories_freq == 0:
                    traj_accumulated_states_droped = np.array(traj_accumulated_states_droped)
                    traj_accumulated_goal_states_dropped = np.array(traj_accumulated_goal_states_dropped)
                    if self.display_plots:
                        if self.train_with_preferences:
                            self.plot_trajectories(traj_accumulated_states_droped, traj_accumulated_goal_states_dropped, filename=f'train_states_preferences/train_trajectories_dropped_{total_timesteps}_{np.random.randint(100)}.png')
                        else:
                            self.plot_trajectories(traj_accumulated_states_droped, traj_accumulated_goal_states_dropped, filename=f'train_states/train_trajectories_dropped_{total_timesteps}_{np.random.randint(100)}.png')
                    traj_accumulated_states_droped = []
                    traj_accumulated_goal_states_dropped = []

                if len(traj_accumulated_actions) != 0 and len(traj_accumulated_actions) % self.display_trajectories_freq == 0:
                    traj_accumulated_states = np.array(traj_accumulated_states)
                    traj_accumulated_actions = np.array(traj_accumulated_actions)
                    traj_accumulated_goal_states = np.array(traj_accumulated_goal_states)
                    if self.display_plots:
                        if self.train_with_preferences:
                            self.plot_trajectories(traj_accumulated_states, traj_accumulated_goal_states, filename=f'train_states_preferences/train_trajectories_{total_timesteps}_{np.random.randint(100)}.png')
                        else:
                            self.plot_trajectories(traj_accumulated_states, traj_accumulated_goal_states, filename=f'train_states/train_trajectories_{total_timesteps}_{np.random.randint(100)}.png')


                    if self.train_with_preferences and not self.use_oracle and self.display_plots:
                        self.test_goal_selector(total_timesteps)               

                    self.dump_data()

                    avg_success = 0.
                    avg_distance_total = 0.
                    avg_distance_commanded_total = 0.0
                    num_values = 0.
                    traj_accumulated_desired_goal_states = np.array(traj_accumulated_desired_goal_states)
                    traj_accumulated_goal_states = np.array(traj_accumulated_goal_states)
                    for i in range(traj_accumulated_desired_goal_states.shape[0]):
                        success = self.env.compute_success(self.env.observation(traj_accumulated_states[i][-1]), self.env.extract_goal(traj_accumulated_desired_goal_states[i]))
                        distance_total = self.env.compute_shaped_distance(self.env.observation(traj_accumulated_states[i][-1]), self.env.extract_goal(traj_accumulated_desired_goal_states[i]))
                        distance_commanded_total = self.env.compute_shaped_distance(self.env.observation(traj_accumulated_states[i][-1]), self.env.extract_goal(traj_accumulated_goal_states[i][-1]))

                        avg_success += success
                        avg_distance_total += distance_total
                        avg_distance_commanded_total += distance_commanded_total
                        num_values += 1
                    if num_values != 0:
                        avg_success = avg_success / num_values
                        avg_distance_total = avg_distance_total / num_values
                        avg_distance_commanded_total = avg_distance_commanded_total / num_values

                        wandb.log({'TrainingSuccess':avg_success, 
                                   'TrainingDistance':avg_distance_total,
                                   'TrainingDistanceCommanded':avg_distance_commanded_total,
                                    'timesteps':self.total_timesteps, 'num_labels_queried':self.num_labels_queried})

                    traj_accumulated_states = []
                    traj_accumulated_actions = []
                    traj_accumulated_goal_states = []
                    traj_accumulated_desired_goal_states = []

                total_timesteps += self.max_path_length
                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length
                
                # Take training steps
                #print(f"timesteps since train {timesteps_since_train}, train policy freq {self.train_policy_freq}, total_timesteps {total_timesteps}, start policy timesteps {self.start_policy_timesteps}")
                if self.full_iters % self.train_policy_freq == 0 and self.full_iters >= self.start_policy_timesteps:
                    self.policy.train()
                    for i in range(len(self.policy_list)):
                        self.policy_list[i].train()
                    for idx in range(int(self.policy_updates_per_step)): # TODO: modify this
                        loss = self.take_policy_step()
                        validation_loss, goal_selector_val_loss = self.validation_loss()

                        if running_loss is None:
                            running_loss = loss
                        else:
                            running_loss = 0.9 * running_loss + 0.1 * loss

                        if running_validation_loss is None:
                            running_validation_loss = validation_loss
                        else:
                            running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss

                        if goal_selector_running_val_loss is None:
                            goal_selector_running_val_loss = goal_selector_val_loss
                        else:
                            goal_selector_running_val_loss = 0.9 * goal_selector_running_val_loss + 0.1 * goal_selector_val_loss

                    self.policy.eval()
                    for i in range(len(self.policy_list)):
                        self.policy_list[i].eval()
                    ranger.set_description('Loss: %s Validation Loss: %s'%(running_loss, running_validation_loss))
                    
                    if self.summary_writer:
                        self.summary_writer.add_scalar('Losses/Train', running_loss, total_timesteps)
                        self.summary_writer.add_scalar('Losses/Validation', running_validation_loss, total_timesteps)
                    wandb.log({'Losses/Train':running_loss, 'timesteps':total_timesteps,  'num_labels_queried':self.num_labels_queried})
                    wandb.log({'Losses/Validation':running_validation_loss, 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
                    
                    self.train_loss_arr.append((running_loss, total_timesteps))
                    self.eval_loss_arr.append((running_validation_loss, total_timesteps))
                    self.train_loss_goal_selector_arr.append((goal_selector_running_val_loss, total_timesteps))

                
                # Evaluate, log, and save to disk
                if timesteps_since_eval >= self.eval_freq:


                    timesteps_since_eval %= self.eval_freq
                    iteration += 1
                    # Evaluation Code
                    self.policy.eval()
                    for i in range(len(self.policy_list)):
                        self.policy_list[i].eval()
                    print("evaluate policy")
                    self.evaluate_policy(self.eval_episodes, total_timesteps=total_timesteps, greedy=True, prefix='Eval', plots_folder=plots_folder)
                    if self.use_images_in_policy:
                        observations, actions, image_observations, goals, img_goals, lengths, horizons, weights = self.replay_buffer.sample_batch(self.eval_episodes)
                    else:
                        _, _, goals, _, _, _ = self.replay_buffer.sample_batch(self.eval_episodes)
                    #self.evaluate_policy_requested(goals, total_timesteps=total_timesteps, greedy=True, prefix='EvalRequested', plots_folder=plots_folder_requested)

                    logger.record_tabular('policy loss', running_loss or 0) # Handling None case
                

                    # "model.h5" is saved in wandb.run.dir & will be uploaded at the end of training
                    #torch.save(self.policy.state_dict(), os.path.join(wandb.run.dir, "model.h5"))
                    now = datetime.now()
                    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
                    #torch.save(self.policy.state_dict(), f"checkpoint/model_{dt_string}.h5")
                    # Save a model file manually from the current directory:
                    #wandb.save('model.h5')

                    #with open( f'checkpoint/buffer_{dt_string}.pkl', 'wb') as f:
                    #    pickle.dump(self.replay_buffer.state_dict(), f)

                    if self.logger_dump:
                        logger.record_tabular('reward model train loss', 0)
                        logger.record_tabular('reward model eval loss', 0)
                            
                        logger.record_tabular('timesteps', total_timesteps)
                        logger.record_tabular('epoch time (s)', time.time() - last_time)
                        logger.record_tabular('total time (s)', time.time() - start_time)
                        last_time = time.time()
                        logger.dump_tabular()

                        
                        # Logging Code
                        if logger.get_snapshot_dir():
                            modifier = str(iteration) if self.save_every_iteration else ''
                            torch.save(
                                self.policy.state_dict(),
                                osp.join(logger.get_snapshot_dir(), 'policy%s.pkl'%modifier)
                            )
                            if hasattr(self.replay_buffer, 'state_dict'):
                                with open(osp.join(logger.get_snapshot_dir(), 'buffer%s.pkl'%modifier), 'wb') as f:
                                    pickle.dump(self.replay_buffer.state_dict(), f)

                            full_dict = dict(env=self.env, policy=self.policy)
                            with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl'%modifier), 'wb') as f:
                                pickle.dump(full_dict, f)
                        
                        ranger.reset()

    def pretrain_goal_selector_func(self):
        observations, _, goals, final_states = self.replay_buffer.sample_batch_with_final_states(self.num_demos_goal_selector)
        self.goal_selector_buffer.add_multiple_data_points(observations, -1) #goals, final_states, np.ones(goals.shape[0]))
        self.train_goal_selector(epochs=self.demo_pretrain_epochs)
        self.test_goal_selector(0)

    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval', total_timesteps=0, plots_folder="plots"):
        print("Evaluate policy")
        env = self.env
        self.eval_env.reset()
        
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            video_filename = f"eval_traj_{total_timesteps}"
            goal = self.eval_env.extract_goal(self.eval_env.sample_goal())

            states, actions, _, goal_states, desired_goal_state, _ = self.sample_trajectory(goal=goal, greedy=True, save_video_trajectory=index==0, video_filename=video_filename)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_states)
            final_dist = self.env_distance(self.env.observation(states[-1]), self.env.extract_goal(desired_goal_state)) # TODO: should we compute shaped distance?
            
            final_dist_vec[index] = final_dist
            success_vec[index] = self.env.compute_success(self.env.observation(states[-1]),  self.env.extract_goal(desired_goal_state)) #(final_dist < self.goal_threshold)
        
        print("Eval goals:", all_goal_states)
        #all_states = np.stack(all_states)
        #all_goal_states = np.stack(all_goal_states)
        print('%s num episodes'%prefix, len(all_goal_states))
        print('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        print('%s success ratio'%prefix, np.mean(success_vec))

        logger.record_tabular('%s num episodes'%prefix, eval_episodes)
        logger.record_tabular('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio'%prefix, np.mean(success_vec))
        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist'%prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio'%prefix,  np.mean(success_vec), total_timesteps)

        wandb.log({'%s/avg final dist'%prefix:np.mean(final_dist_vec), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
        wandb.log({'%s/success ratio'%prefix:np.mean(success_vec), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})

        self.success_ratio_eval_arr.append((np.mean(success_vec), total_timesteps))
        self.distance_to_goal_eval_arr.append((np.mean(final_dist_vec), total_timesteps))
        
        #diagnostics = env.get_diagnostics(all_states, all_goal_states)
        #for key, value in diagnostics.items():
        #    print('%s %s'%(prefix, key), value)
        #    logger.record_tabular('%s %s'%(prefix, key), value)
        
        if self.display_plots:
            self.plot_trajectories(all_states, all_goal_states, extract=False, filename=f'{plots_folder}/eval_{total_timesteps}_{np.random.randint(100)}.png')

        return all_states, all_goal_states

    def display_wall(self):
        walls = self.env.base_env.room.get_walls()
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
        return 

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

    def evaluate_policy_requested(self, requested_goals, greedy=True, prefix='Eval', total_timesteps=0, plots_folder="plots"):
        env = self.env
        
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(len(requested_goals))
        success_vec = np.zeros(len(requested_goals))

        for index, goal in enumerate(requested_goals):

            states, actions, _, goal_state, desired_goal_state, _ = self.sample_trajectory(goal, greedy=True)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[-1], self.env.extract_goal(desired_goal_state))
            
            final_dist_vec[index] = final_dist
            success_vec[index] = self.env.compute_success(self.env.observation(states[-1]),self.env.extract_goal(desired_goal_state)) #(final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        """
        logger.record_tabular('%s num episodes'%prefix, len(requested_goals))
        logger.record_tabular('%s avg final dist requested goals'%prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio requested goals'%prefix, np.mean(success_vec))
        
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s'%(prefix, key), value)
        """
        print('%s num episodes'%prefix, len(requested_goals))
        print('%s avg final dist relabelled goals'%prefix,  np.mean(final_dist_vec))
        print('%s success ratio relabelled goals'%prefix, np.mean(success_vec))

        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist relabelled goals'%prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio relabelled goals'%prefix,  np.mean(success_vec), total_timesteps)
        wandb.log({'%s/avg final dist relabelled goals'%prefix:np.mean(final_dist_vec), 'timesteps':total_timesteps,'num_labels_queried':self.num_labels_queried})
        wandb.log({'%s/success ratio relabelled goals'%prefix:np.mean(success_vec), 'timesteps':total_timesteps, 'num_labels_queried':self.num_labels_queried})
        
        self.success_ratio_relabelled_arr.append((np.mean(success_vec), total_timesteps))
        self.distance_to_goal_eval_relabelled.append((np.mean(success_vec), total_timesteps))
        #diagnostics = env.get_diagnostics(all_states, all_goal_states)
        #for key, value in diagnostics.items():
        #    print('%s %s'%(prefix, key), value)

        if self.display_plots:
            self.plot_trajectories(all_states, all_goal_states, extract=False, filename=f'{plots_folder}/eval_requested_{total_timesteps}_{np.random.randint(100)}.png'%total_timesteps)


        return all_states, all_goal_states
