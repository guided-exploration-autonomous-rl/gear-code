import numpy as np
import copy

class ReplayBuffer:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """
        self.env = env
        self._actions = np.zeros(
            (buffer_size, max_trajectory_length, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        #print("state space shape", *env.state_space.shape)
        self._states = np.zeros(
            (buffer_size, max_trajectory_length, *env.state_space.shape),
            dtype=env.state_space.dtype
        )
        # self._desired_states = np.zeros(
        #     (buffer_size, *env.state_space.shape),
        #     dtype=env.state_space.dtype
        # )
        internal_goal_shape = env._extract_sgoal(env.sample_goal()).shape
        self._internal_goals = np.zeros(
            (buffer_size, max_trajectory_length, *internal_goal_shape),
            dtype=env.observation_space.dtype,
        )
        
        self._length_of_traj = np.zeros(
            (buffer_size,),
            dtype=int
        )
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size
        self.max_trajectory_length = max_trajectory_length
        
    def add_trajectory(self, states, actions, desired_state, length_of_traj=None):
        """
        Adds a trajectory to the buffer

        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """

        print("add trajectory buffer", states.shape, actions.shape)

        assert states.shape[0] == actions.shape[0]

        if length_of_traj is None:
            length_of_traj = states.shape[0]
            print(length_of_traj)

        self._actions[self.pointer, : length_of_traj] = actions
        self._states[self.pointer, : length_of_traj] = states
        self._internal_goals[self.pointer, : length_of_traj] = self.env._extract_sgoal(states)
        # self._desired_states[self.pointer] = desired_state

        self._length_of_traj[self.pointer] = length_of_traj

        self.pointer += 1
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size

    def add_multiple_trajectory(self, states, actions, desired_state, length_of_traj=None):
        """
        Adds a trajectory to the buffer

        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """

        for actions_t, states_t, desired_state_t in zip(actions, states, desired_state):
            self.add_trajectory(states_t, actions_t, desired_state_t, length_of_traj )
    
    def _sample_indices(self, batch_size):
        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)

        prop_idxs_1 = np.random.rand(batch_size)
        prop_idxs_2 = np.random.rand(batch_size)
        time_idxs_1 = np.floor(prop_idxs_1 * (self._length_of_traj[traj_idxs]-1)).astype(int)
        time_idxs_2 = np.floor(prop_idxs_2 * (self._length_of_traj[traj_idxs])).astype(int)
        time_idxs_2[time_idxs_1 == time_idxs_2] += 1

        time_state_idxs = np.minimum(time_idxs_1, time_idxs_2)
        time_goal_idxs = np.maximum(time_idxs_1, time_idxs_2)
        return traj_idxs, time_state_idxs, time_goal_idxs

    def _sample_indices_fixed(self, batch_size, len=30):
        traj_idxs = np.random.choice(self.current_buffer_size-1, batch_size)

        prop_idxs_1 = np.random.rand(batch_size)
        time_idxs_1 = np.floor(prop_idxs_1 * (self._length_of_traj[traj_idxs]-1)).astype(int)

        time_state_idxs = time_idxs_1
        time_goal_idxs = time_idxs_1 + len
        return traj_idxs, time_state_idxs, time_goal_idxs

    def sample_batch(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)
    
    def sample_batch_with_final_states(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        return self._get_batch_with_final_states(traj_idxs, time_state_idxs, time_goal_idxs)
    
    def sample_batch_fixed(self, batch_size, len=30):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices_fixed(batch_size, len)
        
        return self._get_batch_fixed(traj_idxs, time_state_idxs, time_goal_idxs, len)
    
    def _get_batch_with_final_states(self, traj_idxs, time_state_idxs, time_goal_idxs):
        batch_size = len(traj_idxs)
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        actions = self._actions[traj_idxs, time_state_idxs]
        goals = self.env.extract_goal(self._states[traj_idxs, time_goal_idxs])
        final_states = self.env.observation(self._states[traj_idxs, -1])
        lengths = time_goal_idxs - time_state_idxs
        horizons = np.tile(np.arange(self.max_trajectory_length), (batch_size, 1))
        horizons = horizons >= lengths[..., None]

        #norm_lengths = (lengths.copy() - np.mean(lengths)) / np.std(lengths)
        weights = np.ones(len(lengths)) #np.exp(-norm_lengths)

        return observations, actions, goals, final_states
    
    def _get_obs_batch(self, traj_idxs, time_state_idxs):
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        actions = [] 
        for i in range(len(time_state_idxs)):
            actions.append(self._actions[traj_idxs[i], :time_state_idxs[i]+1])

        return observations, actions

    def _get_batch(self, traj_idxs, time_state_idxs, time_goal_idxs):
        batch_size = len(traj_idxs)
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        actions = self._actions[traj_idxs, time_state_idxs]
        goals = self.env.extract_goal(self._states[traj_idxs, time_goal_idxs])
        
        lengths = time_goal_idxs - time_state_idxs
        horizons = np.tile(np.arange(self.max_trajectory_length), (batch_size, 1))
        horizons = horizons >= lengths[..., None]

        #norm_lengths = (lengths.copy() - np.mean(lengths)) / np.std(lengths)
        weights = np.ones(len(lengths)) #np.exp(-norm_lengths)

        return observations, actions, goals, lengths, horizons, weights

    def _get_batch_fixed(self, traj_idxs, time_state_idxs, time_goal_idxs, len=30):
        batch_size = traj_idxs.shape[0]
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        actions = self._actions[traj_idxs, time_state_idxs]
        traj_idxs_goal = copy.deepcopy(traj_idxs)
        for i in range(time_goal_idxs.shape[0]):
            if time_goal_idxs[i] >= self.max_trajectory_length:
                traj_idxs_goal[i] += 1
                time_goal_idxs[i] -= self.max_trajectory_length
        goals = self.env.extract_goal(self._states[traj_idxs_goal, time_goal_idxs])
        
        lengths = time_goal_idxs - time_state_idxs
        horizons = np.tile(np.arange(self.max_trajectory_length), (batch_size, 1))
        horizons = horizons >= lengths[..., None]

        weights = np.ones(lengths.shape[0]) #np.exp(-norm_lengths)

        return observations, actions, goals, lengths, horizons, weights

    def _sample_indices_last_steps(self, batch_size, k=10, last_k_trajectories = -1):
        if last_k_trajectories == -1:
            last_k_trajectories = self.max_buffer_size

        sampling_size = min(self.current_buffer_size, last_k_trajectories)

        traj_idxs = np.random.choice(sampling_size, batch_size)
        traj_idxs = (self.pointer - 1 - traj_idxs ) % self.current_buffer_size 
        
        prop_idxs_1 = np.random.rand(batch_size)
        prop_idxs_2 = np.random.rand(batch_size)
        
        ks = np.array([k for i in range(batch_size)])
        time_idxs_1 = np.floor(prop_idxs_1 * (np.minimum(self._length_of_traj[traj_idxs], ks).astype(int)-1)).astype(int) + np.maximum(self._length_of_traj[traj_idxs]-k, np.zeros(batch_size))
        time_idxs_2 = np.floor(prop_idxs_2 * np.minimum(self._length_of_traj[traj_idxs], ks).astype(int)) + np.maximum(self._length_of_traj[traj_idxs]-k, np.zeros(batch_size))
        time_idxs_2[time_idxs_1 == time_idxs_2] += 1

        time_state_idxs = np.minimum(time_idxs_1, time_idxs_2).astype(int)
        time_goal_idxs = np.maximum(time_idxs_1, time_idxs_2).astype(int)
        print(min(time_state_idxs), max(time_state_idxs), min(time_goal_idxs), max(time_goal_idxs))
        return traj_idxs, time_state_idxs, time_goal_idxs

    def _sample_indices_last_steps_obs(self, batch_size, k=10, last_k_trajectories = -1):
        if last_k_trajectories == -1:
            last_k_trajectories = self.max_buffer_size

        sampling_size = min(self.current_buffer_size, last_k_trajectories)

        traj_idxs = np.random.choice(sampling_size, batch_size)
        traj_idxs = (self.pointer - 1 - traj_idxs ) % self.current_buffer_size 
        
        prop_idxs = np.random.rand(batch_size)
        
        ks = np.array([k for i in range(batch_size)])
        time_idxs = np.floor(prop_idxs * np.minimum(self._length_of_traj[traj_idxs], ks).astype(int)) + np.maximum(self._length_of_traj[traj_idxs]-k, np.zeros(batch_size))

        time_state_idxs = time_idxs.astype(int)
        return traj_idxs, time_state_idxs

    def sample_batch_last_steps(self, batch_size, k=10, last_k_trajectories=-1):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices_last_steps(batch_size, k, last_k_trajectories)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)

    def sample_obs_last_steps(self, batch_size, k=10, last_k_trajectories=-1):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs = self._sample_indices_last_steps_obs(batch_size, k, last_k_trajectories)
        return self._get_obs_batch(traj_idxs, time_state_idxs)



    def save(self, file_name):
        np.savez(file_name,
            states=self._states[:self.current_buffer_size],
            actions=self._actions[:self.current_buffer_size],
            desired_states=self._desired_states[:self.current_buffer_size],
        )

    def load(self, file_name, replace=False):
        data = np.load(file_name, allow_pickle=True)
        states, actions, desired_states = data['states'], data['actions'], data['desired_states']
        n_trajectories = len(states)
        for i in range(n_trajectories):
            self.add_trajectory(states[i], actions[i], desired_states[i])

    def state_dict(self):
        d = dict(internal_goals=self._internal_goals[:self.current_buffer_size])
        if self._states.shape[2] < 100: # Not images
            d.update(dict(
                states=self._states[:self.current_buffer_size],
                actions=self._actions[:self.current_buffer_size],
            ))
        return d

class RewardModelBuffer:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """
        self.env = env
        self._states_1 = np.zeros(
            (buffer_size, env.observation_space.shape[0]),
            dtype=env.state_space.dtype
        )
        self._states_2 = np.zeros(
            (buffer_size, env.observation_space.shape[0]),
            dtype=env.state_space.dtype
        )
        internal_goal_shape = env._extract_sgoal(env.sample_goal()).shape
        self._goals = np.zeros(
            (buffer_size, *internal_goal_shape),
            dtype=env.state_space.dtype
        )
        self._labels = np.zeros(
            (buffer_size, ),
            dtype=float
        )
        
       
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size

    def add_multiple_data_points(self, state_1, state_2, goal, label):
        for s1, s2, g, l in zip(state_1, state_2, goal, label):
            self.add_data_point(s1, s2, g, l)


    def add_data_point(self, state_1, state_2, goal, label):
        self._states_1[self.pointer] = state_1
        self._states_2[self.pointer] = state_2

        self._goals[self.pointer] = goal

        self._labels[self.pointer] = label
       
        self.pointer += 1
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size
    
   
    def sample_batch(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        return self._get_batch(batch_size)

    def _get_batch(self, batch_size):
        idxs = np.random.choice(self.current_buffer_size, batch_size)
        observations_1 = self.env.observation(self._states_1[idxs])
        observations_2 = self.env.observation(self._states_2[idxs])
        goals = self._goals[idxs]
        labels = self._labels[idxs]

        return observations_1, observations_2, goals, labels


class ViceModelBuffer:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """
        self.env = env
        self._positives_1 = np.zeros(
            (buffer_size, env.observation_space.shape[0]),
            dtype=env.state_space.dtype
        )
        self._positives_2 = np.zeros(
            (buffer_size, env.observation_space.shape[0]),
            dtype=env.state_space.dtype
        )
        self._negatives = np.zeros(
            (buffer_size, env.observation_space.shape[0]),
            dtype=env.state_space.dtype
        )
       
        self.pointer_neg = 0
        self.pointer_pos1 = 0
        self.pointer_pos2 = 0
        self.current_buffer_size_neg = 0
        self.current_buffer_size_pos1 = 0
        self.current_buffer_size_pos2 = 0
        self.max_buffer_size = buffer_size
        self.goals = self.env.base_env.goals

    def add_multiple_data_points(self, state_1, goal=0):
        for s1 in state_1:
            self.add_data_point(s1, goal=goal)

    def add_data_point(self, state, goal=0):
        if goal == -1:
            self._negatives[self.pointer_neg] = self.env.observation(state)

            self.pointer_neg += 1
            self.current_buffer_size_neg = max(self.pointer_neg, self.current_buffer_size_neg)
            self.pointer_neg %= self.max_buffer_size
        elif goal == 0:
            self._positives_1[self.pointer_pos1] = self.env.observation(state)

            self.pointer_pos1 += 1
            self.current_buffer_size_pos1 = max(self.pointer_pos1, self.current_buffer_size_pos1)
            self.pointer_pos1 %= self.max_buffer_size
        else:
            self._positives_2[self.pointer_pos2] = self.env.observation(state)

            self.pointer_pos2 += 1
            self.current_buffer_size_pos2 = max(self.pointer_pos2, self.current_buffer_size_pos2)
            self.pointer_pos2 %= self.max_buffer_size
   
    def sample_batch(self, batch_size, goal=0):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """
        neg_batch = self._get_batch(batch_size//2, goal=-1)
        pos_batch = self._get_batch(batch_size//2, goal=goal)
        batch_obs = np.concatenate([neg_batch, pos_batch])
        labels = np.concatenate([np.zeros(len(neg_batch)), np.ones(len(pos_batch))])
        batch_goals = np.array([self.goals[goal] for i in range(len(neg_batch)+len(pos_batch))])
        return batch_obs, batch_goals, labels

    def _get_batch(self, batch_size, goal=0):
        if goal == -1:
            idxs = np.random.choice(self.current_buffer_size_neg, batch_size)
            observations_1 = self.env.observation(self._negatives[idxs])

        elif goal == 0:
            idxs = np.random.choice(self.current_buffer_size_pos1, batch_size)
            observations_1 = self.env.observation(self._positives_1[idxs])

        else:
            idxs = np.random.choice(self.current_buffer_size_pos2, batch_size)
            observations_1 = self.env.observation(self._positives_2[idxs])

        return observations_1

class FakeReplayBuffer:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """
        self.env = env
        self._actions = np.zeros(
            (buffer_size, max_trajectory_length, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        self._states = np.zeros(
            (buffer_size, max_trajectory_length, *env.state_space.shape),
            dtype=env.state_space.dtype
        )

        
        internal_goal_shape = env._extract_sgoal(env.sample_goal()).shape
        self._internal_goals = np.zeros(
            (buffer_size, max_trajectory_length, *internal_goal_shape),
            dtype=env.observation_space.dtype,
        )
        
        self._goal_states = np.zeros(
            (buffer_size, *internal_goal_shape),
            dtype=env.state_space.dtype
        )
        self._length_of_traj = np.zeros(
            (buffer_size,),
            dtype=int
        )
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size
        self.max_trajectory_length = max_trajectory_length
        
    def add_trajectory(self, states, actions, goal_states, length_of_traj=None):
        """
        Adds a trajectory to the buffer

        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """
        batch_size = len(states)

        pointer_idxs = np.arange(self.pointer, self.pointer + batch_size) % self.max_buffer_size
        self._actions[pointer_idxs] = actions
        self._states[pointer_idxs] = states
        self._internal_goals[pointer_idxs] = self.env._extract_sgoal(states)
        self._goal_states[pointer_idxs] = goal_states
        
        if length_of_traj is None:
            length_of_traj = self.max_trajectory_length
        self._length_of_traj[pointer_idxs] = length_of_traj
        
        self.pointer += batch_size
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size

    def _sample_indices(self, batch_size):
        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)
        prop_idxs_1 = np.random.rand(batch_size)
        time_idxs_1 = np.floor(prop_idxs_1 * (self._length_of_traj[traj_idxs]-1)).astype(int)
        return traj_idxs, time_idxs_1
        

    def sample_batch(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """
        traj_idxs, time_idxs = self._sample_indices(batch_size)
        return self._get_batch(traj_idxs, time_idxs)

    def _get_batch(self, traj_idxs, time_state_idxs):
        time_state_idxs = time_state_idxs % self._length_of_traj[traj_idxs]

        batch_size = len(traj_idxs)
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        actions = self._actions[traj_idxs, time_state_idxs]
        goals = self._goal_states[traj_idxs]
        
        horizons = None        
        weights = np.ones(batch_size)

        return observations, actions, goals, horizons, weights


    
    def save(self, file_name):
        np.savez(file_name,
            states=self._states[:self.current_buffer_size],
            actions=self._actions[:self.current_buffer_size],
            goal_states=self._goal_states[:self.current_buffer_size],
        )

    def load(self, file_name, replace=False):
        data = np.load(file_name)
        states, actions, goal_states = data['states'], data['actions'], data['goal_states']
        n_trajectories = len(states)
        for i in range(n_trajectories):
            self.add_trajectory(states[i], actions[i], goal_states[i])

    def state_dict(self):
        d = dict(internal_goals=self._internal_goals[:self.current_buffer_size])
        if self._states.shape[2] < 100: # Not images
            d.update(dict(
                states=self._states[:self.current_buffer_size],
                actions=self._actions[:self.current_buffer_size],
                goal_states=self._goal_states[:self.current_buffer_size],
            ))
        return d


#################################################################################
##################################### IMAGES ####################################
#################################################################################

class ReplayBufferImages:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                input_image_size,
                close_frames,
                far_frames,
                ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """

        self.close_frames = close_frames
        self.far_frames = far_frames
        self.env = env
        self._actions = np.zeros(
            (buffer_size, max_trajectory_length, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        self._states = np.zeros(
            (buffer_size, max_trajectory_length, *env.state_space.shape),
            dtype=env.state_space.dtype
        )
        self._desired_states = np.zeros(
            (buffer_size, *env.state_space.shape),
            dtype=env.state_space.dtype
        )
        self._image_observations = np.zeros(
            (buffer_size, max_trajectory_length, 3, input_image_size, input_image_size),
            dtype=np.uint8
        )
        
        internal_goal_shape = env._extract_sgoal(env.sample_goal()).shape
        self._internal_goals = np.zeros(
            (buffer_size, max_trajectory_length, *internal_goal_shape),
            dtype=env.observation_space.dtype,
        )
        
        self._length_of_traj = np.zeros(
            (buffer_size,),
            dtype=int
        )
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size
        self.max_trajectory_length = max_trajectory_length
        
    def add_trajectory(self, states, actions, image_observations, desired_state, length_of_traj=None):
        from sys import getsizeof
        """
        Adds a trajectory to the buffer
        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            image_observations (np.array): Rdendered states - Needs shape (self.max_path_length, 3, input_image_size, input_image_size)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """
        # print(actions.shape, self._actions[0].shape)
        # assert actions.shape == self._actions[0].shape
        print("add trajectory buffer", states.shape, actions.shape)
        assert states.shape[0] == actions.shape[0]

        if length_of_traj is None:
            length_of_traj = states.shape[0]
            print(length_of_traj)

        self._actions[self.pointer, : length_of_traj] = actions
        self._states[self.pointer, : length_of_traj] = states
        self._image_observations[self.pointer, : length_of_traj] = image_observations[:length_of_traj].squeeze()
        self._internal_goals[self.pointer, : length_of_traj] = self.env._extract_sgoal(states)
        # self._desired_states[self.pointer] = desired_state

        self._length_of_traj[self.pointer] = length_of_traj

        self.pointer += 1
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size

        # self._actions[self.pointer] = actions
        # self._states[self.pointer] = states
        # self._image_observations[self.pointer] = image_observations
        # self._internal_goals[self.pointer] = self.env._extract_sgoal(states)
        # self._desired_states[self.pointer] = desired_state

        # if length_of_traj is None:
        #     length_of_traj = self.max_trajectory_length
        # self._length_of_traj[self.pointer] = length_of_traj

        # self.pointer += 1
        # self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        # self.pointer %= self.max_buffer_size
    
    def _sample_indices(self, batch_size):
        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)

        prop_idxs_1 = np.random.rand(batch_size)
        prop_idxs_2 = np.random.rand(batch_size)
        time_idxs_1 = np.floor(prop_idxs_1 * (self._length_of_traj[traj_idxs]-1)).astype(int)
        time_idxs_2 = np.floor(prop_idxs_2 * (self._length_of_traj[traj_idxs])).astype(int)
        time_idxs_2[time_idxs_1 == time_idxs_2] += 1

        time_state_idxs = np.minimum(time_idxs_1, time_idxs_2)
        time_goal_idxs = np.maximum(time_idxs_1, time_idxs_2)
        return traj_idxs, time_state_idxs, time_goal_idxs

    def sample_batch(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            image_observations
            goals
            image_goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)

    def _get_batch(self, traj_idxs, time_state_idxs, time_goal_idxs):
        batch_size = len(traj_idxs)
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        image_observations = self._image_observations[traj_idxs, time_state_idxs]

        actions = self._actions[traj_idxs, time_state_idxs]
        goals = self.env.extract_goal(self._states[traj_idxs, time_goal_idxs])
        img_goals = self._image_observations[traj_idxs, time_goal_idxs]
        
        lengths = time_goal_idxs - time_state_idxs
        horizons = np.tile(np.arange(self.max_trajectory_length), (batch_size, 1))
        horizons = horizons >= lengths[..., None]

        weights = np.ones(batch_size)

        return observations, actions, image_observations, goals, img_goals, lengths, horizons, weights


    def sample_classification_batch(self, batch_size):
        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)

        prop_idxs_1 = np.random.randint(0, self._length_of_traj[traj_idxs], batch_size)
        labels = np.random.randint(0, 2, batch_size) # 1 = close

        prop_idxs_2 = []

        for i in range(batch_size):
            current = prop_idxs_1[i]

            if labels[i]:
                array = [x for x in range(current - self.close_frames, current + self.close_frames + 1) if (x >= 0 and x < self._length_of_traj[traj_idxs[i]])]
            else:
                array = [x for x in range(self._length_of_traj[traj_idxs[i]]) if (abs(x - current) >= self.far_frames)]

            prop_idxs_2.append(np.random.choice(array))

        observations_1 = self._image_observations[traj_idxs, prop_idxs_1]
        observations_2 = self._image_observations[traj_idxs, prop_idxs_2]

        return np.array(observations_1), np.array(observations_2), np.array(labels)


    def sample_batch_last_steps(self, batch_size, k=5):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            image_observations
            goals
            image_goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        time_state_idxs = np.maximum(time_state_idxs, self.max_trajectory_length - k)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)

    def _sample_indices_last_steps_obs(self, batch_size, k=10, last_k_trajectories = -1):
        if last_k_trajectories == -1:
            last_k_trajectories = self.max_buffer_size

        sampling_size = min(self.current_buffer_size, last_k_trajectories)

        traj_idxs = np.random.choice(sampling_size, batch_size)
        traj_idxs = (self.pointer - 1 - traj_idxs ) % self.current_buffer_size 
        
        prop_idxs = np.random.rand(batch_size)
        
        ks = np.array([k for i in range(batch_size)])
        time_idxs = np.floor(prop_idxs * np.minimum(self._length_of_traj[traj_idxs], ks).astype(int)) + np.maximum(self._length_of_traj[traj_idxs]-k, np.zeros(batch_size))

        time_state_idxs = time_idxs.astype(int)
        return traj_idxs, time_state_idxs

    def _get_obs_batch(self, traj_idxs, time_state_idxs):
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        image_observations = self._image_observations[traj_idxs, time_state_idxs]

        actions = [] 
        for i in range(len(time_state_idxs)):
            actions.append(self._actions[traj_idxs[i], :time_state_idxs[i]+1])

        return observations, image_observations, actions


    def sample_batch_last_steps(self, batch_size, k=10, last_k_trajectories=-1):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices_last_steps(batch_size, k, last_k_trajectories)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)

    def sample_obs_last_steps(self, batch_size, k=10, last_k_trajectories=-1):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs = self._sample_indices_last_steps_obs(batch_size, k, last_k_trajectories)
        return self._get_obs_batch(traj_idxs, time_state_idxs)

    def save(self, file_name):
        np.savez(file_name,
            states=self._states[:self.current_buffer_size],
            actions=self._actions[:self.current_buffer_size],
            image_observations=self._image_observations[:self.current_buffer_size],
            desired_states=self._desired_states[:self.current_buffer_size],
        )

    def load(self, file_name, replace=False):
        data = np.load(file_name)
        states, actions, desired_states = data['states'], data['actions'], data['desired_states']
        n_trajectories = len(states)
        for i in range(n_trajectories):
            self.add_trajectory(states[i], actions[i], desired_states[i])

    def state_dict(self):
        d = dict(internal_goals=self._internal_goals[:self.current_buffer_size])
        if self._states.shape[2] < 100: # Not images
            d.update(dict(
                states=self._states[:self.current_buffer_size],
                actions=self._actions[:self.current_buffer_size],
                image_observations=self._image_observations[:self.current_buffer_size],
                desired_states=self._desired_states[:self.current_buffer_size],
            ))
        return d



class RewardModelBufferImages:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                input_image_size,
                ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """
        self.env = env
        self._states_1 = np.zeros(
            (buffer_size, env.observation_space.shape[0]),
            dtype=env.state_space.dtype
        )
        self._states_2 = np.zeros(
            (buffer_size, env.observation_space.shape[0]),
            dtype=env.state_space.dtype
        )
        internal_goal_shape = env._extract_sgoal(env.sample_goal()).shape
        self._goals = np.zeros(
            (buffer_size, *internal_goal_shape),
            dtype=env.state_space.dtype
        )
        self._labels = np.zeros(
            (buffer_size, ),
            dtype=float
        )

        self._image_observations_1 = np.zeros(
            (buffer_size, 3, input_image_size, input_image_size),
            dtype=np.uint8
        )

        self._image_observations_2 = np.zeros(
            (buffer_size, 3, input_image_size, input_image_size),
            dtype=np.uint8
        )
        
        self._image_goals = np.zeros(
            (buffer_size, 3, input_image_size, input_image_size),
            dtype=np.uint8
        )
       
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size

    def add_multiple_data_points(self, state_1, state_2, goal, label, img1, img2, img_goal):
        for s1, s2, g, l, im1, im2, img in zip(state_1, state_2, goal, label, img1, img2, img_goal):
            self.add_data_point(s1, s2, g, l, im1, im2, img)


    def add_data_point(self, state_1, state_2, goal, label, img1, img2, image_goal):
        self._states_1[self.pointer] = state_1
        self._states_2[self.pointer] = state_2

        self._goals[self.pointer] = goal
        self._labels[self.pointer] = label
        
        self._image_observations_1[self.pointer] = img1
        self._image_observations_2[self.pointer] = img2
        self._image_goals[self.pointer] = image_goal
       
        self.pointer += 1
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size
    
   
    def sample_batch(self, batch_size):
        return self._get_batch(batch_size)

    def _get_batch(self, batch_size):
        idxs = np.random.choice(self.current_buffer_size, batch_size)
        observations_1 = self.env.observation(self._states_1[idxs])
        observations_2 = self.env.observation(self._states_2[idxs])
        goals = self._goals[idxs]
        labels = self._labels[idxs]
        images_1 = self._image_observations_1[idxs]
        images_2 = self._image_observations_2[idxs]
        image_goals = self._image_goals[idxs]

        return observations_1, observations_2, goals, labels, images_1, images_2, image_goals