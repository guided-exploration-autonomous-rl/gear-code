import math
import random
import numpy as np
import torch
from gear.envs.room_env import PointmassGoalEnv
from gear.envs.locobot_env_mujoco import LoCoBotEnvMujoco

class ReachableSet():
    def __init__(self, env, env_name, width_bin=5, height_bin=5):
        self.env = env
        self.env_name = env_name
        self.width_bin= width_bin
        self.height_bin= height_bin
        self.grid = [] 
        if "pointmass" in env_name:
            self.obs_space_width_low, self.obs_space_height_low = [-0.6,-0.6]
            self.obs_space_width_high, self.obs_space_height_high = [0.6,0.6]
        if "pusher" in env_name:
            self.width_bin= 5
            self.height_bin= 5
            self.obs_space_low= [-0.25, -0, -0.25, -0]#self.env.observation_space.low[:2]
            self.obs_space_high = [0.45,0.75, 0.45,0.75]#self.env.observation_space.high[:2]
        if "locobot" in env_name:
            self.obs_space_low = [-3, -2, -1]
            self.obs_space_high = [3, 2, 1]
            self.length_bin = 15
            self.width_bin = 10
            self.angle_bin = 5
        
        self.grid_init()

    def grid_init(self):
        if "pointmass" in self.env_name:
            self.grid = np.zeros((self.width_bin, self.height_bin, self.width_bin, self.height_bin))
        elif "pusher" in self.env_name:
            self.grid = np.zeros((self.width_bin, self.height_bin, self.width_bin, self.height_bin, self.width_bin, self.height_bin, self.width_bin, self.height_bin))
        elif "locobot" in self.env_name:
            self.grid = np.zeros((self.length_bin, self.width_bin, self.angle_bin, self.length_bin, self.width_bin, self.angle_bin))
        return

    def bin_check(self, obs):
        if "pointmass" in self.env_name:
            width_index = math.ceil((obs[0] - self.obs_space_width_low) / (self.obs_space_width_high - self.obs_space_width_low) * self.width_bin) - 1
            height_index = math.ceil((obs[1] - self.obs_space_height_low) / (self.obs_space_height_high - self.obs_space_height_low) * self.height_bin) - 1
            return (max(0, width_index), max(0, height_index))
        elif "pusher" in self.env_name:
            width_index = math.ceil((obs[0] - self.obs_space_low[0]) / (self.obs_space_high[0] - self.obs_space_low[0]) * self.width_bin) - 1
            height_index = math.ceil((obs[1] - self.obs_space_low[1]) / (self.obs_space_high[1] - self.obs_space_low[1]) * self.height_bin) - 1
            width_index2 = math.ceil((obs[2] - self.obs_space_low[2]) / (self.obs_space_high[2] - self.obs_space_low[2]) * self.width_bin) - 1
            height_index2 = math.ceil((obs[3] - self.obs_space_low[3]) / (self.obs_space_high[3] - self.obs_space_low[3]) * self.height_bin) - 1
            return (min(max(0, width_index), self.width_bin-1), min(max(0, height_index), self.height_bin-1), min(max(0, width_index2), self.width_bin-1), min(max(0, height_index2), self.height_bin-1))
        elif "locobot" in self.env_name:
            length_index = math.ceil((obs[0] - self.obs_space_low[0]) / (self.obs_space_high[0] - self.obs_space_low[0]) * self.length_bin) - 1
            width_index = math.ceil((obs[1] - self.obs_space_low[1]) / (self.obs_space_high[1] - self.obs_space_low[1]) * self.width_bin) - 1
            angle_index = math.ceil((obs[2] - self.obs_space_low[2]) / (self.obs_space_high[2] - self.obs_space_low[2]) * self.angle_bin) - 1
            return (min(max(0, length_index), length_index-1), min(max(0, width_index), width_index-1), min(max(0, angle_index), angle_index-1))



    # def grid_update(self, curr_obs, next_obs):
    #     # Get the indices for the current and next observations
    #     c_w, c_h = self.bin_check(curr_obs)
    #     n_w, n_h = self.bin_check(next_obs)
    #     # Increment the counts for the current-to-next and next-to-next transitions
    #     self.grid[c_w][c_h][n_w][n_h] += 1
    #     self.grid[n_w][n_h][n_w][n_h] += 1

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

    # def get_reachable_set(self, state, threshold=10):
    #     reachable_set = []
    #     s_w, s_h = self.bin_check(state)
    #     for i in range(self.grid[s_w][s_h].shape[0]):
    #         for j in range(self.grid[s_w][s_h].shape[1]):
    #             if threshold > 0 and threshold <= 1:
    #                 if self.grid[s_w][s_h][i][j] / np.sum(self.grid[s_w][s_h]) >= threshold:
    #                     reachable_set.append((i, j))
    #             elif threshold > 1:
    #                 if self.grid[s_w][s_h][i][j] >= threshold:
    #                     reachable_set.append((i, j))
    #     return reachable_set

    # def grid_init_random(self, num_sample=10, horizon=50):
    #     buffer = []
    #     if self.env_name == 'pointmass_empty':
    #         for i in range(num_sample):
    #             buffer.append([])
    #             state = np.random.uniform(low=-1, high=1, size=(2,))
    #             state = {'observation': state, 'achieved_goal': state,  'state_achieved_goal': state}
    #             state = self.env._base_obs_to_state(state)
    #             for j in range(horizon):
    #                 obs = self.env.observation(state)
    #                 action = random.randint(0, 8)
    #                 state, _, _, _ = self.env.step(action)
    #                 buffer[i].append(obs)
    #     elif self.env_name == 'pointmass_rooms':
    #         for i in range(num_sample):
    #             buffer.append([])
    #             room_num = random.randint(1, 4)
    #             if room_num == 1:
    #                 x = np.random.uniform(low=0, high=1, size=(1,))
    #                 y = np.random.uniform(low=-1, high=0, size=(1,))
    #                 state = np.array([int(x), int(y)])
    #             elif room_num == 2:
    #                 x = np.random.uniform(low=-1, high=0, size=(1,))
    #                 y = np.random.uniform(low=-1, high=0, size=(1,))
    #                 state = np.array([int(x), int(y)])
    #             elif room_num == 3:
    #                 x = np.random.uniform(low=-1, high=0, size=(1,))
    #                 y = np.random.uniform(low=0, high=1, size=(1,))
    #                 state = np.array([int(x), int(y)])
    #             elif room_num == 4:
    #                 x = np.random.uniform(low=0, high=1, size=(1,))
    #                 y = np.random.uniform(low=0, high=1, size=(1,))
    #                 state = np.array([int(x), int(y)])
    #             state = {'observation': state, 'achieved_goal': state,  'state_achieved_goal': state}
    #             state = self.env._base_obs_to_state(state)
    #             for j in range(horizon):
    #                 obs = self.env.observation(state)
    #                 action = random.randint(0, 8)
    #                 state, _, _, _ = self.env.step(action)
    #                 buffer[i].append(obs)
    #         for traj in buffer:
    #             self.grid_update_traj(traj)
    
    def get_reachable_set_from(self, curr_state, all_states, threshold=0.1):
        s_idxs = self.bin_check(curr_state)
        density = self.grid[s_idxs] 

        reachable_set = []
        densities = []
        for state in all_states:
            n_idxs = self.bin_check(state)
            # if density[n_w, n_h] > threshold:
            if density[n_idxs] > 0:
                reachable_set.append(state)
                densities.append(density[n_idxs])
        
        if len(reachable_set) == 0:
            return reachable_set
        reachable_set = np.array(reachable_set)

        k = int(len(densities)*threshold)

        indices = np.argsort(densities)[-k:]

        reachable_set = reachable_set[indices]
        
        return reachable_set, densities[indices]
    
    def get_reachable_set_cut(self, curr_state, all_states, threshold=5):
        s_idxs = self.bin_check(curr_state)
        density = self.grid[s_idxs] 

        reachable_set = []
        densities = []
        for state in all_states:
            n_idxs = self.bin_check(state)
            # if density[n_w, n_h] > threshold:
            if density[n_idxs] >= threshold:
                reachable_set.append(state)
                densities.append(density[n_idxs])
        
        return np.array(reachable_set), np.array(densities)
    
    # def get_reachable_set_center_state(self, reachable_set_current):
    #     reachable_set_center_state = []
    #     for r_state in reachable_set_current:
    #         r_w, r_h = r_state
    #         s_w = (r_w + 0.5) / self.width_bin * (self.obs_space_width_high - self.obs_space_width_low) + self.obs_space_width_low
    #         s_h = (r_h + 0.5) / self.height_bin * (self.obs_space_height_high - self.obs_space_height_low) + self.obs_space_height_low
    #         state = (s_w, s_h)
    #         reachable_set_center_state.append(np.array(state))
    #     return reachable_set_center_state

# class ReachableSetPusher():
#     def __init__(self, env, env_name, width_bin=5, height_bin=5):
#         self.env = env
#         self.env_name = env_name
#         self.width_bin= width_bin
#         self.height_bin= height_bin
#         self.grid = [] 
#         self.obs_space_low= [-0.25, -0, -0.25, -0]#self.env.observation_space.low[:2]
#         self.obs_space_high = [0.45,0.75, 0.45,0.75]#self.env.observation_space.high[:2]
#         self.grid_init()
#     def grid_init(self):
#         self.grid = np.zeros((self.width_bin, self.height_bin, self.width_bin, self.height_bin, self.width_bin, self.height_bin, self.width_bin, self.height_bin))
#         return

#     def bin_check(self, obs):
#         width_index = math.ceil((obs[0] - self.obs_space_low[0]) / (self.obs_space_high[0] - self.obs_space_low[0]) * self.width_bin) - 1
#         height_index = math.ceil((obs[1] - self.obs_space_low[1]) / (self.obs_space_high[1] - self.obs_space_low[1]) * self.height_bin) - 1
#         width_index2 = math.ceil((obs[2] - self.obs_space_low[2]) / (self.obs_space_high[2] - self.obs_space_low[2]) * self.width_bin) - 1
#         height_index2 = math.ceil((obs[3] - self.obs_space_low[3]) / (self.obs_space_high[3] - self.obs_space_low[3]) * self.height_bin) - 1
#         return (max(0, width_index), max(0, height_index), max(0, width_index2), max(0, height_index2))

#     def grid_update(self, curr_obs, next_obs):
#         # Get the indices for the current and next observations
#         c_w, c_h, c_w2, c_h2 = self.bin_check(curr_obs)
#         n_w, n_h, n_w2, n_h2 = self.bin_check(next_obs)
#         # Increment the counts for the current-to-next and next-to-next transitions
#         self.grid[c_w,c_h,c_w2, c_h2][n_w,n_h, n_w2, n_h2] += 1
#         # self.grid[n_w][n_h][n_w][n_h] += 1

#     def grid_update_traj(self, traj):
#         if len(traj) == 0:
#             return
#         for i, start_state in enumerate(traj):
#             for j, next_state in enumerate(traj[i:]):
#                 self.grid_update(start_state, next_state)
#         return

#     def get_reachable_set_from(self, curr_state, all_states, threshold=5):
#         s_w, s_h, s_w2, s_h2 = self.bin_check(curr_state)
#         density = self.grid[s_w,s_h,s_w2,s_h2] 

#         reachable_set = []
#         for state in all_states:
#             n_w, n_h, n_w2, n_h2 = self.bin_check(state)
#             if density[n_w, n_h, n_w2, n_h2] > threshold:
#                 reachable_set.append(state)

#         return np.array(reachable_set)

        
#     # def get_reachable_set(self, state, threshold=10):
#     #     reachable_set = []
#     #     s_w, s_h, = self.bin_check(state)
#     #     for i in range(self.grid[s_w][s_h].shape[0]):
#     #         for j in range(self.grid[s_w][s_h].shape[1]):
#     #             if threshold > 0 and threshold <= 1:
#     #                 if self.grid[s_w][s_h][i][j] / np.sum(self.grid[s_w][s_h]) >= threshold:
#     #                     reachable_set.append((i, j))
#     #             elif threshold > 1:
#     #                 if self.grid[s_w][s_h][i][j] >= threshold:
#     #                     reachable_set.append((i, j))
#     #     return np.array(reachable_set)

#     # def grid_init_random(self, num_sample=10, horizon=50):
#     #     buffer = []
#     #     if self.env_name == 'pointmass_empty':
#     #         for i in range(num_sample):
#     #             buffer.append([])
#     #             state = np.random.uniform(low=-1, high=1, size=(2,))
#     #             state = {'observation': state, 'achieved_goal': state,  'state_achieved_goal': state}
#     #             state = self.env._base_obs_to_state(state)
#     #             for j in range(horizon):
#     #                 obs = self.env.observation(state)
#     #                 action = random.randint(0, 8)
#     #                 state, _, _, _ = self.env.step(action)
#     #                 buffer[i].append(obs)
#     #     elif self.env_name == 'pointmass_rooms':
#     #         for i in range(num_sample):
#     #             buffer.append([])
#     #             room_num = random.randint(1, 4)
#     #             if room_num == 1:
#     #                 x = np.random.uniform(low=0, high=1, size=(1,))
#     #                 y = np.random.uniform(low=-1, high=0, size=(1,))
#     #                 state = np.array([int(x), int(y)])
#     #             elif room_num == 2:
#     #                 x = np.random.uniform(low=-1, high=0, size=(1,))
#     #                 y = np.random.uniform(low=-1, high=0, size=(1,))
#     #                 state = np.array([int(x), int(y)])
#     #             elif room_num == 3:
#     #                 x = np.random.uniform(low=-1, high=0, size=(1,))
#     #                 y = np.random.uniform(low=0, high=1, size=(1,))
#     #                 state = np.array([int(x), int(y)])
#     #             elif room_num == 4:
#     #                 x = np.random.uniform(low=0, high=1, size=(1,))
#     #                 y = np.random.uniform(low=0, high=1, size=(1,))
#     #                 state = np.array([int(x), int(y)])
#     #             state = {'observation': state, 'achieved_goal': state,  'state_achieved_goal': state}
#     #             state = self.env._base_obs_to_state(state)
#     #             for j in range(horizon):
#     #                 obs = self.env.observation(state)
#     #                 action = random.randint(0, 8)
#     #                 state, _, _, _ = self.env.step(action)
#     #                 buffer[i].append(obs)
#     #         for traj in buffer:
#     #             self.grid_update_traj(traj)

#     def get_reachable_set_center_state(self, reachable_set_current):
#         reachable_set_center_state = []
#         for r_state in reachable_set_current:
#             r_w, r_h = r_state
#             s_w = (r_w + 0.5) / self.width_bin * (self.obs_space_width_high - self.obs_space_width_low) + self.obs_space_width_low
#             s_h = (r_h + 0.5) / self.height_bin * (self.obs_space_height_high - self.obs_space_height_low) + self.obs_space_height_low
#             state = (s_w, s_h)
#             reachable_set_center_state.append(np.array(state))
#         return reachable_set_center_state

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
