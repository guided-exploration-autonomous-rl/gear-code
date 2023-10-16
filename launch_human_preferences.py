
from operator import ne
from PIL import Image

from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import rlutil.torch.pytorch_util as ptu
import seaborn as sns


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from gear import envs
from gear.algo import buffer, gear, variants, networks
import gym
import argparse
import wandb
import copy
import numpy as np
import torch
from gear.algo import human_preferences
from gear.algo.ppo_new import PPO
#from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

class SubProcVecEnvCustom(SubprocVecEnv):
    def __init__(self, env_fns, start_method=None):
        super().__init__(env_fns, start_method)
        self.timesteps=0
        self.collected_episodes = 0
    
    def display_wall(self):
        #walls = self._env.base_env.room.get_walls()
        walls = [[(-0.6, 0), (-0.35, 0)],
            [(-0.25, 0), (0.25, 0)],
            [(0, 0), (0.6, 0)],
            [(0, -0.6), (0, -0.35)],
            [(0, -0.25), (0, 0.25)],
            [(0, 0.35), (0, 0.6)],]
        walls.append([[0.6,-0.6], [0.6,0.6]])
        walls.append([[0.6,0.6], [-0.6,0.6]])
        walls.append([[-0.6,0.6], [-0.6,-0.6]])
        walls.append([[-0.6,-0.6], [0.6,-0.6]])
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='',  color = 'black', linewidth=4)

    # def plot_trajectory(self, trajs):

    #     # plot added trajectories to fake replay buffer
    #     plt.clf()
    #     self.display_wall()
    #     colors = sns.color_palette('hls', (len(trajs)))
    #     print("inside plot trajectory", np.shape(trajs))
    #     for i in range(len(trajs)):
    #         plt.plot(trajs[i][:,0], trajs[i][:, 1], color=colors[i], zorder = -1)

    #         plt.scatter(trajs[i][-1,2], trajs[i][-1, 3], color=colors[i], zorder = -1)
    #     #if 'train_states_preferences' in filename:
    #     #    color = 'black'
    
    #     wandb.log({"trajectory": wandb.Image(plt)})   
        

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        self.timesteps += len(obs)
        if np.any(dones):
            success = 0
            distance = 0
            paths = []
            for idx, i in enumerate(infos):
                if dones[idx]:
                    distance += i['info/distance']
                    success += int(i['info/success'])
                    paths.append(i['path'])
            
            # self.plot_trajectory(paths)
            distance/=sum(dones)
            success /= sum(dones)
            wandb.log({'timesteps':self.timesteps, 'Train/Success':success, 'Train/Distance':distance})
        return obs, rewards, dones, infos

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self):
        print("logger",self.logger.name_to_value)
        return True
    
class UnWrapper(gym.Env):
    def __init__(self, env, goal, max_path_legnth, env_name, dense_reward=False, reward_model=None, is_eval=False ):
        super(UnWrapper, self).__init__()
        self._env = env
        self.reward_model = reward_model

        self.state_space = self.observation_space

        self.env_name = env_name

        self.current_goal = self._env.sample_goal()

        self.is_eval = is_eval
        self.max_path_length = max_path_legnth

        self.total_timesteps = 0

        print("max path length inside wrapper", max_path_legnth)

        self.dense_reward = dense_reward

        self.reset()

    def __getattr__(self, attr):
        return getattr(self._env, attr)
    
    @property
    def action_space(self, ):
        return self._env.action_space

    @property
    def observation_space(self, ):
        dim = self._env.observation_space.shape[0]*2
        ones = np.ones(dim)
        return gym.spaces.Box(-ones, ones)

    def compute_shaped_distance(self, state, goal):
        if self.env_name == "pointmass_rooms":
            return self._env.compute_shaped_distance(np.array(state), np.array(goal))[0]

        return self._env.compute_shaped_distance(np.array(state), np.array(goal))
        
    def compute_success(self, state, goal):
        if self.env_name == "pointmass_rooms":
            return self._env.compute_success(np.array(state), np.array(goal))[0]
        return self._env.compute_success(np.array(state), np.array(goal))

    def render(self):
        self._env.render()

    def get_state(self, obs):
        return np.concatenate([self._env.observation(obs), self._env.extract_goal(self.current_goal)])
    def reset(self):
        """
        Resets the environment and returns a state vector
        Returns:
            The initial state
        """
        if self.is_eval or self.total_timesteps > 100000:
            self.total_timesteps = 0
            self._env.reset()
        self.current_goal = self._env.sample_goal()
        self.current_states = []
        self.current_timestep = 0
        obs = self.get_state(self._env.get_obs())
        return obs

    def step(self, a):
        """
        Runs 1 step of simulation
        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """
        self.current_timestep +=1
        new_state, reward, done, info = self._env.step(a)
        new_state = self.get_state(new_state)
        distance = self.compute_shaped_distance(self._env.observation(new_state), self._env.extract_goal(new_state))
        success = self.compute_success(self._env.observation(new_state), self._env.extract_goal(new_state))
        self.current_states.append(new_state)

        info['info/distance'] = distance
        info['info/success'] = success

        self.total_timesteps += 1

        done = self.current_timestep == self.max_path_length



        info['info/final_distance'] = distance
        info['info/final_success'] = success
        info['path'] = np.array(self.current_states)

        reward = distance#self.reward_model(torch.Tensor(new_state).to('cuda'), torch.Tensor(self.goal).to('cuda')).detach().cpu().numpy()[0]

        if done:
            self.current_goal = self._env.sample_goal()
            self.current_timestep = 0
            self.current_states = []
            done = False
        
        return new_state, reward, done, info


    def observation(self, state):
        """
        Returns the observation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        raise self._env.observation(state)
    
    def extract_goal(self, state):
        """
        Returns the goal representation for a given state
        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        raise self._env.extract_goal(state)

    def goal_distance(self, state, ):
        return self._env.goal_distance(state, self.goal)

    def sample_goal(self):
        return self._env.sample_goal() #self.goal_space.sample()

def make_env(env_name, env_params, goal, reward_model=None, dense_reward=False, task_config="slide_cabinet,microwave", maze_type=3, continuous_action_space=False, num_blocks=1, max_path_length=50, is_eval=False):
    print("maze type", maze_type)
        
    env = envs.create_env(env_name, task_config, num_blocks, True, maze_type, False, continuous_action_space, 0.05, goal_config="bimodal", env_version=None)

    wrapped_env, policy, _, replay_buffer, reward_model_buffer, gcsl_kwargs = variants.get_params_human_preferences(env, env_params)
    
    print("env action space", wrapped_env.action_space)
    info_keywords = ('info/distance', 'info/success', 'info/final_distance', 'info/final_success')
    unwrapped_env = UnWrapper(wrapped_env, goal, max_path_length, env_name, dense_reward, reward_model, is_eval)
    final_env = Monitor(unwrapped_env, filename='info.txt', info_keywords=info_keywords)

    return final_env


def experiment(wandb_run, env_name, task_config, label_from_last_k_steps=-1,normalize_rewards=False,reward_layers="400,600,600,300", 
label_from_last_k_trajectories=-1, gpu=0, entropy_coefficient= 0.01, num_envs=4, num_steps_per_policy_step=1000, explore_episodes=10, 
reward_model_epochs=400, reward_model_num_samples=1000, goal_threshold = 0.05, num_blocks=1, buffer_size=20000, use_oracle=False, 
display_plots=False, max_path_length=50, network_layers='128,128', train_rewardmodel_freq=2, fourier=False, 
use_wrong_oracle=False,
eval_episodes=5,
display_trajectories=False,
fourier_reward_model=False, normalize=False, max_timesteps=1e6, reward_model_name="", no_training=False, continuous_action_space=True, maze_type=3):
    ptu.set_gpu(gpu)
    
    print("here", ptu.CUDA_DEVICE)

    print("Using oracle", use_oracle)
    env = envs.create_env(env_name, task_config, num_blocks, maze_type=maze_type, continuous_action_space=continuous_action_space)
    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers # TODO: useless
    env_params['reward_layers'] = reward_layers
    env_params['buffer_size']=buffer_size
    env_params['fourier']=fourier
    env_params['fourier_goal_selector']=fourier_reward_model
    env_params['normalize'] = normalize
    env_params['env_name'] = env_name
    env_params['use_oracle'] = use_oracle
    env_params['reward_model_name']=reward_model_name
    env_params['use_horizon'] = False
    env_params['fourier_goal_selector']=fourier_reward_model
    env_params['maze_type']=maze_type
    env_params['goal_selector_name']=''
    env_params['reward_model_name']=reward_model_name
    env_params['continuous_action_space'] = continuous_action_space
    fake_env, policy, reward_model, replay_buffer, reward_model_buffer, gcsl_kwargs = variants.get_params_human_preferences(env, env_params)
    goal = fake_env.extract_goal(fake_env.sample_goal())

    if not use_oracle:
        reward_model.to(ptu.CUDA_DEVICE)

    env_kwargs = {
        'env_name':env_name, 
        'env_params':env_params,
        'task_config':task_config, 
        'num_blocks':num_blocks,
        'goal':goal,
        'reward_model':reward_model,
        'max_path_length':max_path_length,
        'maze_type':maze_type,
        'continuous_action_space':continuous_action_space,
        'is_eval':False
        }
    env = make_vec_env(make_env, vec_env_cls=SubProcVecEnvCustom, env_kwargs=env_kwargs, n_envs=num_envs)
    env_kwargs['is_eval'] = True
    env_eval = make_vec_env(make_env, vec_env_cls=SubProcVecEnvCustom, env_kwargs=env_kwargs, n_envs=1)


    policy_kwargs = dict()
    policy_kwargs['net_arch'] = variants.get_network_layers(env_params)

    n_steps = 2048

    if "ravens" in env_name:
        n_steps = 40

    model = PPO("MlpPolicy", env, verbose=2, n_steps=n_steps, tensorboard_log=f'runs/{wandb_run.id}', ent_coef=entropy_coefficient, device=ptu.CUDA_DEVICE, policy_kwargs=policy_kwargs)

    algo_kwargs = dict()
    algo_kwargs['explore_episodes']= explore_episodes
    algo_kwargs['goal_threshold']=goal_threshold
    algo_kwargs['reward_model_num_samples']=reward_model_num_samples
    algo_kwargs['train_reward_model_freq']=train_rewardmodel_freq
    algo_kwargs['reward_model_epochs']= reward_model_epochs
    algo_kwargs['eval_episodes']=eval_episodes
    algo_kwargs['display_trajectories_freq']=20#display_trajectories_freq,
    algo_kwargs['display_trajectories']=display_trajectories#display_trajectories_freq,
    algo_kwargs['reward_model_batch_size']=256#reward_model_batch_size,
    algo_kwargs['num_steps_per_policy_step'] = num_steps_per_policy_step
    algo_kwargs['num_envs'] = num_envs
    algo_kwargs['fake_env'] = fake_env

    import os 
    os.makedirs(env_name, exist_ok=True)
    os.makedirs(env_name + "/rewardmodel_test", exist_ok=True)

    print("cuda device", ptu.CUDA_DEVICE)
    algo = human_preferences.HumanPreferences(
        env,
        env_eval,
        model, # TODO
        reward_model, # TODO
        replay_buffer, # TODO
        reward_model_buffer, # TODO
        reward_model_buffer, # TODO
        env_name,
        max_path_length=max_path_length,
        max_timesteps=max_timesteps,
        use_oracle=use_oracle,
        display_plots=display_plots,
        goal=goal,
        wandb_run=wandb_run,
        entropy_coefficient=entropy_coefficient,
        label_from_last_k_trajectories=label_from_last_k_trajectories,
        label_from_last_k_steps=label_from_last_k_steps,
        normalize_rewards=normalize_rewards,
        no_training=no_training,
        use_wrong_oracle=use_wrong_oracle,
        device=ptu.CUDA_DEVICE,
        **algo_kwargs
    )

    algo.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--maze_type",type=int, default=3)
    parser.add_argument("--num_blocks",type=int, default=3)
    parser.add_argument("--num_envs",type=int, default=4)
    parser.add_argument("--max_timesteps",type=int, default=2e6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_path_length", type=int, default=50)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--network_layers",type=str, default='256,64')
    parser.add_argument("--task_config",type=str, default='slide_cabinet,microwave')
    parser.add_argument("--reward_layers",type=str, default='400,600,600,300')
    parser.add_argument("--train_rewardmodel_freq",type=int, default=1)
    parser.add_argument("--reward_model_num_samples",type=int, default=1000)
    parser.add_argument("--display_plots",action="store_true", default=False)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--use_oracle",action="store_true", default=False)
    parser.add_argument("--fourier_reward_model",action="store_true", default=False)
    parser.add_argument("--display_trajectories",action="store_true", default=False)
    parser.add_argument("--use_wrong_oracle",action="store_true", default=False)
    parser.add_argument("--continuous_action_space",action="store_true", default=False)
    parser.add_argument("--start_from_scratch_every_epoch",action="store_true", default=False)
    parser.add_argument("--normalize_rewards",action="store_true", default=False)
    parser.add_argument("--no_training",action="store_true", default=False)
    parser.add_argument("--fourier",action="store_true", default=False)
    parser.add_argument("--reward_model_epochs",type=int, default=400)
    parser.add_argument("--num_steps_per_policy_step",type=int, default=400)
    parser.add_argument("--goal_threshold",type=float, default=0.05)
    parser.add_argument("--entropy_coefficient",type=float, default=0.01)
    parser.add_argument("--label_from_last_k_steps",type=int, default=400)
    parser.add_argument("--label_from_last_k_trajectories",type=int, default=400)
    parser.add_argument("--reward_model_name", type=str, default='')
    parser.add_argument("--eval_episodes",type=int, default=5)
    parser.add_argument("--explore_episodes",type=int, default=5)


    args = parser.parse_args()

    wandb_suffix = "human_preferences"
    if args.use_oracle:
        wandb_suffix += "oracle"
    wandb_run = wandb.init(project=args.env_name+"gcsl_preferences", name=f"{args.env_name}_{wandb_suffix}_{args.seed}", config={
    'seed': args.seed, 
    'num_envs':args.num_envs,
    'lr':args.lr, 
    'max_path_length':args.max_path_length,
    'batch_size':args.batch_size,
    'max_timesteps':args.max_timesteps,
    'task_config':args.task_config,
    'train_rewardmodel_freq':args.train_rewardmodel_freq,
    'task_config':args.task_config,
    'display_plots':args.display_plots,
    'buffer_size':args.buffer_size,
    'use_oracle':args.use_oracle,
    'fourier_reward_model':args.fourier_reward_model,
    'fourier':args.fourier,
    'goal_threshold':args.goal_threshold,
    'reward_model_epochs':args.reward_model_epochs,
    'reward_model_num_samples':args.reward_model_num_samples,
    'num_steps_per_policy_step':args.num_steps_per_policy_step,
    'gpu':args.gpu,
    'entropy_coefficient':args.entropy_coefficient,
    'label_from_last_k_trajectories':args.label_from_last_k_trajectories,
    'label_from_last_k_steps':args.label_from_last_k_steps,
    'reward_layers':args.reward_layers,
    'normalize_rewards':args.normalize_rewards,
    'reward_model_name':args.reward_model_name,
    'no_training':args.no_training,
    'maze_type':args.maze_type,
    'num_blocks':args.num_blocks,
    'continuous_action_space':args.continuous_action_space,
    'use_wrong_oracle':args.use_wrong_oracle,
    'eval_episodes':args.eval_episodes,
    'explore_episodes':args.eval_episodes,
    'display_trajectories':args.display_trajectories,
    })


    #setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment( 
        wandb_run,
        args.env_name, 
        task_config=args.task_config, 
        max_timesteps=args.max_timesteps,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size, 
        max_path_length=args.max_path_length, 
        display_plots=args.display_plots, 
        train_rewardmodel_freq=args.train_rewardmodel_freq,
        use_oracle=args.use_oracle,
        fourier=args.fourier,
        fourier_reward_model=args.fourier_reward_model,
        goal_threshold=args.goal_threshold, 
        reward_model_epochs= args.reward_model_epochs,
        reward_model_num_samples=args.reward_model_num_samples,
        num_steps_per_policy_step=args.num_steps_per_policy_step,
        gpu=args.gpu,
        entropy_coefficient=args.entropy_coefficient,
        label_from_last_k_steps=args.label_from_last_k_steps,
        label_from_last_k_trajectories=args.label_from_last_k_trajectories,
        reward_layers=args.reward_layers,
        normalize_rewards=args.normalize_rewards,
        reward_model_name=args.reward_model_name,
        no_training=args.no_training,
        maze_type=args.maze_type,
        num_blocks=args.num_blocks,
        continuous_action_space=args.continuous_action_space,
        use_wrong_oracle=args.use_wrong_oracle,
        eval_episodes=args.eval_episodes,
        explore_episodes=args.explore_episodes,
        display_trajectories = args.display_trajectories,
        )