from gear.algo import gear
from numpy import VisibleDeprecationWarning
import doodad as dd
import gear.doodad_utils as dd_utils
import argparse
import wandb
import copy

def run( output_dir='/tmp',start_frontier = -1,
        frontier_expansion_rate=10,
        frontier_expansion_freq=-1,
        select_goal_from_last_k_trajectories=-1,
        throw_trajectories_not_reaching_goal=False,
        repeat_previous_action_prob=0.8,
        greedy_before_stopping=False, 
        reward_layers="600,600", 
        fourier=False,
        fourier_goal_selector=False,
        command_goal_if_too_close=False,
        display_trajectories_freq=20,
        label_from_last_k_steps=-1,
        label_from_last_k_trajectories=-1,
        contrastive=False,
        pick_or_place=False,
        k_goal=1, use_horizon=False, 
        sample_new_goal_freq=1, 
        weighted_sl=False, 
        buffer_size=20000, 
        stopped_thresh=0.05, 
        eval_episodes=200, 
        maze_type=0, 
        random_goal=False,
        explore_length=20, 
        desired_goal_sampling_freq=0.0,
        num_blocks=1, 
        deterministic_rollout=False,
        train_policy_freq=10, 
        network_layers="128,128", 
        epsilon_greedy_rollout=0, 
        epsilon_greedy_exploration=0.2, 
        remove_last_k_steps=8, 
        select_last_k_steps=8, 
        eval_freq=5e3, 
        expl_noise_mean = 0,
        expl_noise_std = 1,
        goal_selector_epochs=400,
        stop_training_goal_selector_after=-1,
        no_training_goal_selector=False,
        normalize=False,
        set_desired_when_stopped=True, 
        task_config="slide_cabinet,microwave",
        human_input=False,
        logger_dump=False, save_videos = True, 
        continuous_action_space=False,
        goal_selector_batch_size=64,
        goal_threshold=-1,
        check_if_stopped=False,
        not_save_videos=False,
        human_data_file='',
        goal_config="",
        num_demos=0,
        env_version=0,
        demos_folder_name=None,
        pretrain_policy=False,
        pretrain_goal_selector=False,
        demo_pretrain_epochs=5000,
        offset=500,
        no_cond=False,
        buffer_random_init=300,
        reset_free=False,
        use_reachable_set_densities=False,
        env_name='pointmass_empty',train_goal_selector_freq=10, 
        distance_noise_std=0,  exploration_when_stopped=True, 
        remove_last_steps_when_stopped=True,  
        goal_selector_num_samples=100, data_folder="data", display_plots=False, render=False,
        explore_episodes=5, gpu=0, sample_softmax=False, seed=0, load_goal_selector=False,
        batch_size=100, train_regression=False,load_buffer=False, save_buffer=-1, policy_updates_per_step=1,
        select_best_sample_size=1000, max_path_length=50, lr=5e-4, train_with_preferences=True,
        start_policy_timesteps=500, log_tensorboard=False, use_oracle=False, exploration_horizon=30, 
        use_wrong_oracle=False,
        use_prop=False,
        autoregressive_epochs=30000,
        use_reachable_set_autoregressive=False,
        autoregressive_freq=300,
        comment="", max_timesteps=2e-4, goal_selector_name='', use_reachable_set=True, reachable_sample_rate=300, reachable_thres=0.5, 
        input_image_size=64, use_images_in_policy=False, use_images_in_reward_model=False, use_images_in_stopping_criteria=False, close_frames=2, far_frames=10, autoregress=False,**kwargs):

    print("PRETRAIN", pretrain_policy, pretrain_goal_selector)

    import gym
    import numpy as np
    from rlutil.logging import log_utils, logger
    
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True, 0)

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from gear import envs
    from gear.envs.env_utils import DiscretizedActionEnv

    # Algo
    from gear.algo import buffer, variants, networks

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = envs.create_env(env_name, task_config, num_blocks, random_goal, maze_type, pick_or_place, continuous_action_space, goal_threshold, goal_config=goal_config, env_version=env_version)
    env_eval = envs.create_env(env_name, task_config, num_blocks, random_goal, maze_type, pick_or_place, continuous_action_space, goal_threshold, goal_config=goal_config, goal=0, env_version=env_version)

    env.reset()
    env_eval.reset()

    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers
    env_params['reward_layers'] = reward_layers
    env_params['buffer_size'] = buffer_size
    env_params['use_horizon'] = use_horizon
    env_params['fourier'] = fourier
    env_params['pick_or_place'] = pick_or_place
    env_params['fourier_goal_selector'] = fourier_goal_selector
    env_params['normalize']=normalize
    env_params['env_name'] = env_name

    print(env_params)
    env_params['goal_selector_name']=goal_selector_name
    env_params['continuous_action_space'] = continuous_action_space
    env_params['input_image_size'] = input_image_size
    env_params['use_images_in_policy'] = use_images_in_policy
    env_params['use_images_in_reward_model'] = use_images_in_reward_model
    env_params['use_images_in_stopping_criteria'] = use_images_in_stopping_criteria
    env_params['close_frames'] = close_frames
    env_params['far_frames'] = far_frames
    
    env_params['autoregress'] = autoregress

    policy_list = []
    env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, gcsl_kwargs = variants.get_params(env, env_params)
    env_eval, _, _, _, _, _, _ = variants.get_params(env_eval, env_params)

    policy_list.append(policy)
    for i in range(1):
        _, policy_new, _, _, _, _, _ = variants.get_params(env, env_params)
        policy_list.append(policy_new)

    gcsl_kwargs['lr']=lr
    gcsl_kwargs['max_timesteps']=max_timesteps
    gcsl_kwargs['batch_size']=batch_size
    gcsl_kwargs['max_path_length']=max_path_length
    gcsl_kwargs['policy_updates_per_step']=policy_updates_per_step
    gcsl_kwargs['explore_episodes']=explore_episodes
    gcsl_kwargs['eval_episodes']=eval_episodes
    gcsl_kwargs['eval_freq']=eval_freq
    gcsl_kwargs['remove_last_k_steps']=remove_last_k_steps
    gcsl_kwargs['select_last_k_steps']=select_last_k_steps
    gcsl_kwargs['train_policy_freq'] = train_policy_freq
    gcsl_kwargs['continuous_action_space']=continuous_action_space
    gcsl_kwargs['expl_noise_mean'] = expl_noise_mean
    gcsl_kwargs['expl_noise_std'] = expl_noise_std
    gcsl_kwargs['reset_free'] = reset_free
    gcsl_kwargs['check_if_stopped'] = check_if_stopped
    gcsl_kwargs['use_reachable_set'] = use_reachable_set
    gcsl_kwargs['reachable_sample_rate'] = reachable_sample_rate
    gcsl_kwargs['reachable_thres']=reachable_thres
    # Image related params
    gcsl_kwargs['input_image_size'] = input_image_size
    gcsl_kwargs['use_images_in_policy'] = use_images_in_policy
    gcsl_kwargs['use_images_in_reward_model'] = use_images_in_reward_model
    gcsl_kwargs['classifier_model'] = classifier_model
    gcsl_kwargs['use_images_in_stopping_criteria'] = use_images_in_stopping_criteria
    print(gcsl_kwargs)

    algo = gear.GEAR(
        env,
        env_eval,
        policy,
        policy_list,
        goal_selector,
        replay_buffer,
        goal_selector_buffer,
        log_tensorboard=log_tensorboard,
        train_with_preferences=train_with_preferences,
        use_oracle=use_oracle,
        save_buffer=save_buffer,
        train_regression=train_regression,
        load_goal_selector=load_goal_selector,
        sample_softmax = sample_softmax,
        display_plots=display_plots,
        render=render,
        data_folder=data_folder,
        goal_selector_num_samples=goal_selector_num_samples,
        train_goal_selector_freq=train_goal_selector_freq,
        remove_last_steps_when_stopped=remove_last_steps_when_stopped,
        exploration_when_stopped=exploration_when_stopped,
        distance_noise_std=distance_noise_std,
        save_videos=save_videos,
        logger_dump=logger_dump,
        human_input=human_input,
        epsilon_greedy_exploration=epsilon_greedy_exploration,
        epsilon_greedy_rollout=epsilon_greedy_rollout,
        set_desired_when_stopped=set_desired_when_stopped,
        explore_length=explore_length,
        greedy_before_stopping=greedy_before_stopping,
        stopped_thresh=stopped_thresh,
        weighted_sl=weighted_sl,
        sample_new_goal_freq=sample_new_goal_freq,
        k_goal=k_goal,
        frontier_expansion_freq=frontier_expansion_freq,
        frontier_expansion_rate=frontier_expansion_rate,
        start_frontier=start_frontier,
        select_goal_from_last_k_trajectories=select_goal_from_last_k_trajectories,
        throw_trajectories_not_reaching_goal=throw_trajectories_not_reaching_goal,
        command_goal_if_too_close=command_goal_if_too_close,
        display_trajectories_freq=display_trajectories_freq,
        label_from_last_k_steps=label_from_last_k_steps,
        label_from_last_k_trajectories=label_from_last_k_trajectories,
        contrastive=contrastive,
        deterministic_rollout=deterministic_rollout,
        repeat_previous_action_prob=repeat_previous_action_prob,
        desired_goal_sampling_freq=desired_goal_sampling_freq,
        goal_selector_batch_size=goal_selector_batch_size,
        goal_selector_epochs=goal_selector_epochs,
        not_save_videos=not_save_videos,
        use_wrong_oracle=use_wrong_oracle,
        human_data_file=human_data_file,
        no_training_goal_selector=no_training_goal_selector,
        stop_training_goal_selector_after=stop_training_goal_selector_after,
        num_demos=num_demos,
        pretrain_policy=pretrain_policy,
        pretrain_goal_selector=pretrain_goal_selector,
        demo_pretrain_epochs=demo_pretrain_epochs,
        demos_folder_name=demos_folder_name,
        use_reachable_set_densities=use_reachable_set_densities,
        offset=offset,
        buffer_random_init=buffer_random_init,
        use_prop=use_prop,
        autoregressive_epochs=autoregressive_epochs,
        use_reachable_set_autoregressive=use_reachable_set_autoregressive,
        autoregressive_freq=autoregressive_freq,
        no_cond=no_cond,
        **gcsl_kwargs
    )

    exp_prefix = 'example/%s/gcsl/' % (env_name,)
    #if logger_dump:
    #    log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir)
    algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--no_preferences", action="store_true", default=False)
    parser.add_argument("--log_tensorboard", action="store_true", default=False)
    parser.add_argument("--max_timesteps",type=int, default=2e6)
    parser.add_argument("--start_policy_timesteps",type=int, default=0)
    parser.add_argument("--train_without_preferences",action="store_true", default=False)
    parser.add_argument("--use_oracle",action="store_true", default=False)
    parser.add_argument("--exploration_horizon", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save_buffer", type=int, default=-1)
    parser.add_argument("--load_buffer",action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_path_length", type=int, default=50)
    parser.add_argument("--comment", type=str, default='')
    parser.add_argument("--goal_selector_name", type=str, default='')
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--select_best_sample_size", type=int, default=1000)
    parser.add_argument("--policy_updates_per_step", type=int, default=1)
    parser.add_argument("--train_regression", action='store_true', default=False)
    parser.add_argument("--load_goal_selector", action='store_true', default=False)
    parser.add_argument("--sample_softmax", action='store_true', default=False)
    parser.add_argument("--explore_episodes", type=int, default=5)
    parser.add_argument("--render", action='store_true', default=False)
    parser.add_argument("--display_plots", action='store_true', default=False)
    parser.add_argument("--goal_selector_num_samples", type=int, default=100)
    parser.add_argument("--train_goal_selector_freq", type=int, default=10)
    parser.add_argument("--not_remove_last_steps_when_stopped",  action='store_true', default=False)
    parser.add_argument("--not_exploration_when_stopped",  action='store_true', default=False)
    parser.add_argument("--distance_noise_std", type=float, default=0)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--save_no_videos",  action='store_true', default=False)
    parser.add_argument("--logger_dump",  action='store_true', default=False)
    parser.add_argument("--human_input",  action='store_true', default=False)
    parser.add_argument("--eval_freq", type=int, default=5e3)
    parser.add_argument("--task_config", type=str, default='slide_cabinet,microwave')
    parser.add_argument("--epsilon_greedy_exploration", type=float, default=1)
    parser.add_argument("--epsilon_greedy_rollout", type=float, default=0) # probability of getting the best action (argmax) during rollout instead of sampling from prob
    parser.add_argument("--no_set_desired_when_stopped",  action='store_true', default=False)
    parser.add_argument("--select_last_k_steps", type=int, default=-1)
    parser.add_argument("--remove_last_k_steps", type=int, default=8)
    parser.add_argument("--explore_length", type=int, default=20)
    parser.add_argument("--train_policy_freq", type=int, default=1)
    parser.add_argument("--network_layers",type=str, default='128,128')
    parser.add_argument("--reward_layers",type=str, default='600,600')
    parser.add_argument("--weighted_sl",  action='store_true', default=False)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--random_goal",  action='store_true', default=False)
    parser.add_argument("--maze_type", type=int, default=0)
    parser.add_argument("--greedy_before_stopping",  action='store_true', default=False)
    parser.add_argument("--stopped_thresh", type=float, default=0.05)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--use_horizon",  action='store_true', default=False)
    parser.add_argument("--sample_new_goal_freq",   type=int, default=1)
    parser.add_argument("--k_goal",   type=int, default=1)
    parser.add_argument("--start_frontier",   type=int, default=-1)
    parser.add_argument("--frontier_expansion_rate",   type=int, default=-1)
    parser.add_argument("--frontier_expansion_freq",   type=int, default=-1)
    parser.add_argument("--select_goal_from_last_k_trajectories",   type=int, default=-1)
    parser.add_argument("--throw_trajectories_not_reaching_goal",  action='store_true', default=False)
    parser.add_argument("--fourier",  action='store_true', default=False)
    parser.add_argument("--fourier_goal_selector",  action='store_true', default=False)
    parser.add_argument("--command_goal_if_too_close",  action='store_true', default=False)
    parser.add_argument("--display_trajectories_freq", type=int, default=20)
    parser.add_argument("--label_from_last_k_steps", type=int, default=-1)
    parser.add_argument("--label_from_last_k_trajectories", type=int, default=-1)
    parser.add_argument("--contrastive",  action='store_true', default=False)
    parser.add_argument("--goal_selector_buffer_size", type=int, default=-1)
    parser.add_argument("--pick_or_place",  action='store_true', default=False)
    parser.add_argument("--deterministic_rollout",  action='store_true', default=False)
    parser.add_argument("--repeat_previous_action_prob", type=float, default=0.8)
    parser.add_argument("--continuous_action_space",  action='store_true', default=False)
    parser.add_argument("--expl_noise_mean", type=float, default=0)
    parser.add_argument("--expl_noise_std", type=float, default=1)
    parser.add_argument("--normalize",  action='store_true', default=False)
    parser.add_argument("--desired_goal_sampling_freq", type=float, default=0.0)
    parser.add_argument("--goal_threshold", type=float, default=-1)
    parser.add_argument("--goal_selector_epochs", type=int, default=400)
    parser.add_argument("--goal_selector_batch_size", type=int, default=64)
    parser.add_argument("--check_if_stopped",  action='store_true', default=False)
    parser.add_argument("--not_save_videos",  action='store_true', default=False)
    parser.add_argument("--use_wrong_oracle",  action='store_true', default=False)
    parser.add_argument("--no_training_goal_selector",  action='store_true', default=False)
    parser.add_argument("--reset_free",  action='store_true', default=False)
    parser.add_argument("--human_data_file",type=str, default='')
    parser.add_argument("--goal_config",type=str, default='')
    parser.add_argument("--stop_training_goal_selector_after", type=int, default=-1)
    parser.add_argument("--use_reachable_set",  action='store_true', default=False)
    parser.add_argument("--reachable_sample_rate", type=int, default=300)
    parser.add_argument("--reachable_thres", type=float, default=0.5)
    parser.add_argument("--input_image_size", type=int, default=64)
    parser.add_argument("--use_images_in_policy",  action='store_true', default=False)
    parser.add_argument("--use_images_in_reward_model",  action='store_true', default=False)
    parser.add_argument("--use_images_in_stopping_criteria",  action='store_true', default=False)
    parser.add_argument("--pretrain_policy",  action='store_true', default=False)
    parser.add_argument("--demo_pretrain_epochs", type=int, default=5000)
    parser.add_argument("--num_demos", type=int, default=0)
    parser.add_argument("--offset", type=int, default=500)
    parser.add_argument("--autoregress",  action='store_true', default=False)
    parser.add_argument("--demos_folder_name", type=str, default=None)
    parser.add_argument("--env_version", type=int, default=0)
    parser.add_argument("--use_reachable_set_densities",  action='store_true', default=False)
    parser.add_argument("--buffer_random_init", type=int, default=300)
    parser.add_argument("--use_prop",  action='store_true', default=False)
    parser.add_argument("--use_reachable_set_autoregressive",  action='store_true', default=False)
    parser.add_argument("--autoregressive_epochs", type=int, default=30000)
    parser.add_argument("--autoregressive_freq", type=int, default=300)
    parser.add_argument("--no_cond",  action='store_true', default=False)
     

    #parser.add_argument("--start_hallucination",type=int, default=0)

    args = parser.parse_args()

    data_folder_name = f"{args.env_name}_"

    demos_folder_name = args.demos_folder_name
    if demos_folder_name is None:
        demos_folder_name = args.env_name
    wandb_suffix = ""
    if args.use_oracle:
        data_folder_name = data_folder_name+"_use_oracle_"
        wandb_suffix = "oracle"
    elif args.human_input:
        wandb_suffix = "human"
    elif args.train_without_preferences:
        data_folder_name = data_folder_name+"_standard_"
        wandb_suffix = "std"
    else:
        data_folder_name = data_folder_name + "_goal_selector_"
        wandb_suffix = "goal_selector"

    data_folder_name = data_folder_name + str(args.seed)
    
    params = {
        'seed': args.seed,
        'env_name': args.env_name, #'pointmass_rooms', #['lunar', 'pointmass_empty','pointmass_rooms', 'pusher', 'claw', 'door'],
        'gpu': 0,
        'use_preferences': not args.no_preferences,
        'log_tensorboard': True, #args.log_tensorboard,
        'train_with_preferences': not args.train_without_preferences,
        'use_oracle': args.use_oracle,
        'exploration_horizon': args.exploration_horizon,
        'lr': args.lr,
        'comment': args.comment, 
        'max_timesteps':args.max_timesteps,
        'batch_size':args.batch_size,
        'goal_selector_name':args.goal_selector_name,
        'max_path_length':args.max_path_length,
        'select_best_sample_size':args.select_best_sample_size,
        'policy_updates_per_step':args.policy_updates_per_step,
        'load_buffer':args.load_buffer,
        'load_goal_selector':args.load_goal_selector,
        'save_buffer':args.save_buffer,
        'train_regression':args.train_regression,
        'sample_softmax':args.sample_softmax,
        'explore_episodes':args.explore_episodes,
        'render':args.render,
        'display_plots':args.display_plots,
        'data_folder':data_folder_name,
        'goal_selector_num_samples':args.goal_selector_num_samples,
        'train_goal_selector_freq':args.train_goal_selector_freq,
        'remove_last_steps_when_stopped': not args.not_remove_last_steps_when_stopped,
        'exploration_when_stopped': not args.not_exploration_when_stopped,
        'eval_episodes':args.eval_episodes,
        'distance_noise_std': args.distance_noise_std,
        'save_videos': not args.save_no_videos,
        'logger_dump':args.logger_dump,
        'human_input':args.human_input,
        'task_config':args.task_config,
        'epsilon_greedy_exploration':args.epsilon_greedy_exploration,
        'epsilon_greedy_rollout':args.epsilon_greedy_rollout,
        'set_desired_when_stopped':not args.no_set_desired_when_stopped,
        'eval_freq':args.eval_freq,
        'remove_last_k_steps':args.remove_last_k_steps,
        'select_last_k_steps':args.select_last_k_steps,
        'explore_length':args.explore_length,
        'network_layers':args.network_layers,
        'weighted_sl':args.weighted_sl,
        'train_policy_freq':args.train_policy_freq,
        'num_blocks':args.num_blocks,
        'random_goal':args.random_goal,
        'maze_type':args.maze_type,
        'greedy_before_stopping':args.greedy_before_stopping,
        'stopped_thresh':args.stopped_thresh,
        'buffer_size':args.buffer_size,
        'use_horizon':args.use_horizon,
        'sample_new_goal_freq':args.sample_new_goal_freq,
        'k_goal':args.k_goal,
        'reward_layers':args.reward_layers,
        'start_frontier':args.start_frontier,
        'frontier_expansion_rate':args.frontier_expansion_rate,
        'frontier_expansion_freq':args.frontier_expansion_freq,
        'select_goal_from_last_k_trajectories':args.select_goal_from_last_k_trajectories,
        'throw_trajectories_not_reaching_goal':args.throw_trajectories_not_reaching_goal,
        'fourier':args.fourier,
        'fourier_goal_selector':args.fourier_goal_selector,
        'command_goal_if_too_close':args.command_goal_if_too_close,
        'display_trajectories_freq':args.display_trajectories_freq,
        'label_from_last_k_steps':args.label_from_last_k_steps,
        'label_from_last_k_trajectories':args.label_from_last_k_trajectories,
        'goal_selector_buffer_size':args.goal_selector_buffer_size,
        'contrastive':args.contrastive,
        'pick_or_place':args.pick_or_place,
        'deterministic_rollout':args.deterministic_rollout,
        'repeat_previous_action_prob':args.repeat_previous_action_prob,
        'continuous_action_space':args.continuous_action_space,
        'expl_noise_mean':args.expl_noise_mean,
        'expl_noise_std':args.expl_noise_std,
        'normalize':args.normalize,
        'desired_goal_sampling_freq':args.desired_goal_sampling_freq,
        'goal_threshold':args.goal_threshold,
        'goal_selector_epochs':args.goal_selector_epochs,
        'goal_selector_batch_size':args.goal_selector_batch_size,
        'check_if_stopped':args.check_if_stopped,
        'not_save_videos':args.not_save_videos,
        'human_data_file':args.human_data_file,
        'use_wrong_oracle':args.use_wrong_oracle,
        'no_training_goal_selector':args.no_training_goal_selector,
        #'start_hallucination': args.start_hallucination
        'stop_training_goal_selector_after':args.stop_training_goal_selector_after,
        'reset_free':args.reset_free, 
        'goal_config':args.goal_config,
        'use_reachable_set': args.use_reachable_set,
        'reachable_sample_rate':args.reachable_sample_rate,
        'reachable_thres':args.reachable_thres,
        'input_image_size': args.input_image_size,
        'use_images_in_policy': args.use_images_in_policy,
        'use_images_in_stopping_criteria': args.use_images_in_stopping_criteria,
        'use_images_in_reward_model': args.use_images_in_reward_model,
        'pretrain_policy':args.pretrain_policy,
        'demo_pretrain_epochs':args.demo_pretrain_epochs,
        'num_demos':args.num_demos,
        'offset':args.offset,
        'autoregress':args.autoregress,
        'demos_folder_name':demos_folder_name,
        'env_version':args.env_version,
        'use_reachable_set_densities':args.use_reachable_set_densities,
        'buffer_random_init':args.buffer_random_init,
        'use_prop':args.use_prop,
        'autoregressive_epochs':args.autoregressive_epochs,
        'use_reachable_set_autoregressive':args.use_reachable_set_autoregressive,
        'autoregressive_freq':args.autoregressive_freq,
        'no_cond':args.no_cond,
    }

    if args.use_wrong_oracle:
        wandb_suffix = wandb_suffix + "wrong_oracle"

    if args.use_reachable_set:
        wandb_suffix = wandb_suffix + "_reachable"
    if args.pretrain_policy:
        wandb_suffix = wandb_suffix + "_"+str(args.num_demos)
    wandb.init(project=args.env_name+"gcsl_preferences", name=f"{args.env_name}_{wandb_suffix}_{args.seed}", config=params, )

    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
