import math
import random
import numpy as np
import torch
from gear.envs.room_env import PointmassGoalEnv
from gear.envs.locobot_env_mujoco import LoCoBotEnvMujoco
from autoregressive_example import autoregressive_model
import matplotlib.pyplot as plt 

import matplotlib.cm as cm
import wandb 

class ReachableSetAutoregressive():
    def __init__(self, env, replay_buffer,  env_name,max_diff = 0,epochs=30000, num_buckets=20, batch_size=4096, width_bin=5, height_bin=5):
        self.env = env
        self.env_name = env_name
        self.width_bin= width_bin
        self.height_bin= height_bin
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.epochs = epochs
        self.grid = [] 
        self.num_buckets = num_buckets
        self.replay_buffer = replay_buffer
        self.max_diff=max_diff
        if "pointmass" in env_name:
            self.obs_space_width_low, self.obs_space_height_low = [-.6,-.6]
            self.obs_space_width_high, self.obs_space_height_high = [.6,.6]
        if "pusher" in env_name:
            self.width_bin= 5
            self.height_bin= 5
            self.obs_space_low= [-0.25, -0, -0.25, -0]#self.env.observation_space.low[:2]
            self.obs_space_high = [0.45,0.75, 0.45,0.75]#self.env.observation_space.high[:2]
        
        self.model = autoregressive_model.AutoRegressiveModel(self.env.observation_space.shape[0], self.env.observation_space.shape[0], 1024, 3, num_buckets = num_buckets, ac_low = np.array([-.6, -.6]), ac_high = np.array([.6,.6])).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.grid_init()

    def sample_visited_batch(self, batch_size):
        """
        Retruns pairs of observations that are at most max_diff timesteps appart
        """
        traj_idxs = np.random.choice(self.replay_buffer.current_buffer_size, batch_size)

        time_idxs_1 = np.random.randint(0, self.replay_buffer._length_of_traj[traj_idxs] - 1)
        time_idxs_2 = np.random.randint(1 + time_idxs_1, np.minimum(self.replay_buffer._length_of_traj[traj_idxs], time_idxs_1 + self.max_diff + 1))

        observations1 = self.env.observation(self.replay_buffer._states[traj_idxs, time_idxs_1])

        num_buckets = 20
        ac_low = self.model.ac_low.cpu().numpy()
        ac_high = self.model.ac_high.cpu().numpy()
        bucket_size = (ac_high - ac_low)/num_buckets
        bucket_idx = (observations1 - ac_low) // (bucket_size + 1e-9)
        bucket_idx = np.clip(bucket_idx, 0, num_buckets - 1)
        
        observations1 = bucket_idx
        
        observations2 = self.env.observation(self.replay_buffer._states[traj_idxs, time_idxs_2])

        return observations1, observations2

    def train_autoregressive_model(self, freq_print = None):
        """
        Trains the autoregressive model by sampling pairs of visited states
        """

        if freq_print is None:
            freq_print = self.num_epochs / 5
        
        self.model.trunks.train()
        running_loss = 0.0

        for epoch in range(self.num_epochs):
            states_1, states_2 = self.sample_visited_batch(self.batch_size)
            self.optimizer.zero_grad()

            states_1 = torch.Tensor(states_1).cuda()
            states_2 = torch.Tensor(states_2).cuda()

            loss = -torch.mean(self.model.log_prob(states_1, states_2))
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            
            if epoch == 0:
                print(f"At epoch {epoch} the value of the loss is {loss}")
        
        print(f"At epoch {epoch} the value of the loss is {loss}")

        self.model.trunks.eval()
        return running_loss / self.num_epochs

    def encode(self, s):
        ac_low = self.model.ac_low.cpu().numpy()
        ac_high = self.model.ac_high.cpu().numpy()
        bucket_size = (ac_high - ac_low)/self.num_buckets
        bucket_idx = (s[:3] - ac_low) // (bucket_size + 1e-9)
        bucket_idx = np.clip(bucket_idx, 0, self.num_buckets - 1)
    #     print(bucket_idx)
        return "$".join([str(int(x)) for x in bucket_idx])

    def grid_init(self):
        if "pointmass" in self.env_name:
            self.grid = np.zeros((self.num_buckets, self.num_buckets, self.num_buckets, self.num_buckets))
        elif "pusher" in self.env_name:
            self.grid = np.zeros((self.num_buckets, self.num_buckets, self.num_buckets, self.num_buckets, self.num_buckets, self.num_buckets, self.num_buckets, self.num_buckets))
        return

    def grid_update_traj(self, traj):
        if traj[0].shape[0] != self.env.observation_space.shape[0]*3:
            print(traj, traj[0].shape, self.env.observation_space.shape)
            assert False
        for i, start_state in enumerate(traj):
            s_idxs  = self.bin_check(start_state)
            for j, next_state in enumerate(traj[i+1:]):
                n_idxs = self.bin_check(next_state)
                self.grid[s_idxs][n_idxs] += 1
        return

    def forward_autoregressive_model(self, s1,s2):
        ac_low = self.model.ac_low.cpu().numpy()
        ac_high = self.model.ac_high.cpu().numpy()
        bucket_size = (ac_high - ac_low)/self.num_buckets
        bucket_idx = (s1 - ac_low) // (bucket_size + 1e-9)
        s1 = np.clip(bucket_idx, 0, self.num_buckets - 1)
        
        states_1 = torch.Tensor(s1).cuda()
        states_2 = torch.Tensor(s2).cuda()

        with torch.no_grad():
            return self.model.log_prob(states_1, states_2).detach().cpu().numpy()
    def query(self, x, y):
        s_idxs = self.bin_check(x)
        density = self.grid[s_idxs] 

        n_idxs = self.bin_check(y)

        return density[n_idxs]
    
    def bin_check(self, obs):
        if "pointmass" in self.env_name:
            width_index = math.ceil((obs[0] - self.model.ac_low[0]) / (self.model.ac_high[0] - self.model.ac_low[0]) * self.num_buckets) - 1
            height_index = math.ceil((obs[1] - self.model.ac_low[1]) / (self.model.ac_high[1] - self.model.ac_low[1]) * self.num_buckets) - 1
            return (max(0, width_index), max(0, height_index))
        elif "pusher" in self.env_name:
            width_index = math.ceil((obs[0] - self.obs_space_low[0]) / (self.obs_space_high[0] - self.obs_space_low[0]) * self.num_buckets) - 1
            height_index = math.ceil((obs[1] - self.obs_space_low[1]) / (self.obs_space_high[1] - self.obs_space_low[1]) * self.num_buckets) - 1
            width_index2 = math.ceil((obs[2] - self.obs_space_low[2]) / (self.obs_space_high[2] - self.obs_space_low[2]) * self.num_buckets) - 1
            height_index2 = math.ceil((obs[3] - self.obs_space_low[3]) / (self.obs_space_high[3] - self.obs_space_low[3]) * self.num_buckets) - 1
            return (min(max(0, width_index), self.num_buckets-1), min(max(0, height_index), self.height_bin-1), min(max(0, width_index2), self.num_buckets-1), min(max(0, height_index2), self.num_buckets-1))
        
        
        ###
        states = np.array(np.meshgrid(np.linspace(-.6, .6, 100), np.linspace(-.6,.6,100)))
        states = np.array(states).reshape(2,-1).T
        states = np.array([[0, 1] for i in range(10000)])

        start = np.array(self.env.observation(self.env.get_obs()))
        starts = np.array([[0.25,0.25] for _ in range(100 * 100)])

    def plot_comparison(self, curr_state, all_states, thresh, size=20):
        states = np.array(np.meshgrid(np.linspace(-.6, .6, size), np.linspace(-.6,.6,size)))
        states = np.array(states).reshape(2,-1).T

        start = np.array(curr_state)
        starts = np.array([start for _ in range(size * size)])

        r_val = np.exp(np.array(self.forward_autoregressive_model(starts, states)))
        plt.clf()
        plt.cla()
        plt.scatter(states[:, 0], states[:, 1], c=r_val, cmap=cm.jet)
        plt.scatter(start[0], start[1], color="black")
        plt.savefig("autoregressive_res.png")
        autoregressive_plt = wandb.Image(plt)
        
        r_val = np.array([self.query(starts[i], states[i]) for i in range(len(starts))])
        plt.clf()
        plt.cla()
        plt.scatter(states[:, 0], states[:, 1], c=r_val, cmap=cm.jet)
        plt.savefig("query_res.png")
        query_res_plt = wandb.Image(plt)

        autoreg_set, r_val = self.get_reachable_set_cut(curr_state, all_states, threshold=thresh)
        plt.clf()
        plt.cla()
        self.display_wall()
        plt.scatter(autoreg_set[:, 0], autoreg_set[:, 1], c=r_val, cmap=cm.jet)
        plt.scatter(curr_state[0], curr_state[1], color="black")
        autoreg_cut = wandb.Image(plt)

        densities_set, r_val = self.get_reachable_set_cut_densities(curr_state, all_states, threshold=5)
        plt.clf()
        plt.cla()
        self.display_wall()
        plt.scatter(densities_set[:, 0], densities_set[:, 1], c=r_val, cmap=cm.jet)
        plt.scatter(curr_state[0], curr_state[1], color="black")
        densities_cut = wandb.Image(plt)

        wandb.log({
            "autoregressive_model_res":autoregressive_plt, 
            "query_res":query_res_plt,
            "reachable_set_cut":autoreg_cut,
            "densities_set_cut":densities_cut,
        })
    
    def get_reachable_set_from(self, curr_state, all_states, threshold=0.1):
        starts = np.repeat(curr_state[None], len(all_states), axis=0)
        states = np.array(all_states)
        densities = np.exp(np.array(self.forward_autoregressive_model(starts, states)))

        k = int(len(densities)*threshold)

        indices = np.argsort(densities)[-k:]

        reachable_set = states[indices]
        
        return reachable_set, np.array(densities[indices])
    
    def get_reachable_set_cut(self, curr_state, all_states, threshold=0.01):
        starts = np.repeat(curr_state[None], len(all_states), axis=0)
        states = np.array(all_states)
        densities = np.exp(np.array(self.forward_autoregressive_model(starts, states)))

        idxs = densities > threshold

        reachable_set = states[idxs]
        
        return np.array(reachable_set), np.array(densities[idxs])
    
    def get_reachable_set_cut_densities(self, curr_state, all_states, threshold=0.01):
        starts = np.repeat(curr_state[None], len(all_states), axis=0)
        states = np.array(all_states)
        densities = np.array([self.query(starts[i], states[i]) for i in range(len(starts))])

        idxs = densities > threshold

        reachable_set = states[idxs]
        
        return np.array(reachable_set), np.array(densities[idxs])
    
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

def env_distance(env, state, goal):
        obs = env.observation(state)
        # Original
        if isinstance(env.wrapped_env, PointmassGoalEnv):
            return env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return env.get_shaped_distance(obs, goal)
        return None


def get_closest_reachable_state(env, reachable_set, goal, goal_selector):
    # goal_selector.eval()
    closest_state = None
    if isinstance(env.wrapped_env, LoCoBotEnvMujoco):
        env_name = "locobot_mujoco"
    else:
        env_name = 'other_env'
    if reachable_set != []:
        closest_state = reachable_set[0]
    else:
        return None
    
    if env_name == "locobot_mujoco":
        success_thres = 0.25
    else:
        success_thres = 0.05
    for state in reachable_set:
        
        if env_distance(env, state, goal) <= success_thres: #locobot: 0.25, pointmass: 0.05
            closest_state = goal
            break
        # if env_distance(env, state, goal) <= env_distance(env, closest_state, goal):
        # if state[0] >= 0 and state[1] < 0:
        #     goal_query = [0, -0.3]
        # elif state[0] < 0 and state[1] <= 0:
        #     goal_query = [-0.3, 0]
        # elif state[0] <= 0 and state[1] > 0:
        #     goal_query = [0, 0.3]
        # elif state[0] > 0 and state[1] >= 0:
        #     goal_query = goal
        # use goal_selector 
        # if goal_selector(torch.Tensor(state), torch.Tensor(goal)).cpu().detach().numpy() >= goal_selector(torch.Tensor(closest_state), torch.Tensor(goal)).cpu().detach().numpy():
        if env_distance(env, state, goal) < env_distance(env, closest_state, goal):
            closest_state = state
    return closest_state

def get_closest_reachable_state_from_image(env, reachable_set_state_image, goal, goal_selector):
    closest_state = None
    if reachable_set_state_image != []:
        closest_state = reachable_set_state_image[0][0]
        closest_state_image = reachable_set_state_image[0]
    else:
        return None
    
    for (state, image) in reachable_set_state_image:
        # state = {'observation': state, 'achieved_goal': state,  'state_achieved_goal': state}
        if env.base_env.room.get_shaped_distance(state, goal) <= 0.05: #locobot: 0.25, pointmass: 0.05
            closest_state = goal
            closest_state_image = (goal, image)
            break
        if env.base_env.room.get_shaped_distance(state, goal) < env.base_env.room.get_shaped_distance(closest_state, goal):
            closest_state = state
            closest_state_image = (state, image)
    return closest_state_image

# def get_shortest_path_reachable_state(env, curr_state, reachable_set, goal):
#     closest_state_shortest = None
#     if reachable_set != []:
#         closest_state_shortest = reachable_set[0]
#     else:
#         return None
#     for state in reachable_set:
#         if env_distance(env, state, goal) <= 0.05:
#             closest_state_shortest = goal
#             break
#         if env_distance(env, curr_state, state) + env_distance(env, state, goal) <= env_distance(env, curr_state, closest_state) + env_distance(env, closest_state, goal):
#             closest_state_shortest = state
#     return closest_state_shortest

def get_min_reach_state(state, goal, reachable_set, depth=3, threshold=5, sample_rate=10):
    reach = reachable_set.get_reachable_set(state, threshold)
    reach_center = reachable_set.get_reachable_set_center_state(reach)
    reach_center = random.sample(reach_center, min(sample_rate, len(reach_center)))
    min_state = state
    min_dist = np.linalg.norm((min_state, goal))
    queue = []
    for state in reach_center:
        queue.append((state, state, 1))
        while queue != []:
            prev_state, curr_state, d = queue.pop(0)
            dist = np.linalg.norm(curr_state-goal)
            if dist < min_dist:
                min_state = prev_state
                min_dist = dist
            if d < depth:
                reach_curr = reachable_set.get_reachable_set(curr_state, threshold=10)
                reach_center_curr = reachable_set.get_reachable_set_center_state(reach_curr)
                reach_center_curr = random.sample(reach_center_curr, min(sample_rate, len(reach_center_curr)))
                for r in reach_center_curr:
                    queue.append((prev_state, r, d+1))
    return min_state

class ReachableSetEnsemble():
    pass