# Guided Exploration for Autonomous Reinforcement learning (GEAR)

## Citation
```
@inproceedings{pamies2023autonomous,
title={Autonomous Robotic Reinforcement Learning with Asynchronous Human Feedback},
author={Max Balsells I Pamies and Marcel Torne Villasevil and Zihan Wang and Samedh Desai and Pulkit Agrawal and Abhishek Gupta},
booktitle={7th Annual Conference on Robot Learning},
year={2023},
url={https://openreview.net/forum?id=z3D__-nc9y}
}
```


## Installation Setup
Conda

```
conda env create -f environment.yml
conda activate gear
pip install -e dependencies/Locobot_mujoco_sim
pip install -e dependencies/Locobot-sim
conda develop dependencies
conda develop dependencies/lexa_benchmark
conda develop dependencies/ravens
```

If you have, and would like to use, a GPU, you will need to additionally install a GPU-compiled version of PyTorch. To do so, simply run

```
pip uninstall torch && pip install torch==1.1.0
```

## Environments available
- Sawyer pusher: `pusher_hard`

- Locobot Navigation: `locobot_mujoco`

- Four room navigation: `pointmass_rooms`

- Franka Kitchen: `kitchenSeq`

## Sample Experiment
Pusher:
```
python launch_gear.py --env_name=pusher_hard --check_if_stopped --epsilon_greedy_rollout=1 --epsilon_greedy_exploration=0  --use_oracle --eval_episodes=200 --max_path_length=70 --policy_updates_per_step=100 --explore_length=20 --network_layers=400,600,600,300 --display_plots --buffer_size=5000 --explore_episodes=10 --sample_new_goal_freq=10 --select_best_sample_size=1000 --select_goal_from_last_k_trajectories=-1 --stopped_thresh=0.05 --select_last_k_steps=-1 --max_timesteps=2000000 --fourier --repeat_previous_action_prob=0.25 --reward_layers=400,600,600,300 --fourier_goal_selector --goal_selector_num_samples=1000 --seed=0 --reset_free --goal_config=bimodal --use_reachable_set_densities --reachable_thres=30 --remove_last_k_steps=9 --eval_episodes=10 --reachable_sample_rate=5 --buffer_random_init=1 --offset=500
```

HUGE

```
python launch_gear.py --check_if_stopped --epsilon_greedy_rollout=1 --epsilon_greedy_exploration=0 --env_name=pusher_hard --use_oracle --eval_episodes=200 --max_path_length=70 --policy_updates_per_step=100 --explore_length=20 --network_layers=400,600,600,300 --display_plots --buffer_size=5000 --explore_episodes=10 --sample_new_goal_freq=10 --select_best_sample_size=1000 --select_goal_from_last_k_trajectories=-1 --stopped_thresh=0.05 --select_last_k_steps=-1 --max_timesteps=2000000 --fourier --repeat_previous_action_prob=0.25 --reward_layers=400,600,600,300 --fourier_goal_selector --goal_selector_num_samples=1000 --reset_free --goal_config=bimodal --remove_last_k_steps=9 --eval_episodes=10 --buffer_random_init=1 --offset=500 --seed=0
```

DENSITIES

```
python launch_gear.py --check_if_stopped --epsilon_greedy_rollout=1 --epsilon_greedy_exploration=0 --env_name=pusher_hard --use_oracle --eval_episodes=200 --max_path_length=70 --policy_updates_per_step=100 --explore_length=20 --network_layers=400,600,600,300 --display_plots --buffer_size=5000 --explore_episodes=10 --sample_new_goal_freq=10 --select_best_sample_size=1000 --select_goal_from_last_k_trajectories=-1 --stopped_thresh=0.05 --select_last_k_steps=-1 --max_timesteps=2000000 --fourier --repeat_previous_action_prob=0.25 --reward_layers=400,600,600,300 --fourier_goal_selector --goal_selector_num_samples=1000 --reset_free --goal_config=bimodal --use_reachable_set_densities --reachable_thres=30 --remove_last_k_steps=9 --eval_episodes=10 --reachable_sample_rate=5 --buffer_random_init=1 --offset=500 --seed=0
```

AUTOREGRESSIVE

```
python launch_gear.py --check_if_stopped --epsilon_greedy_rollout=1 --epsilon_greedy_exploration=0 --env_name=pusher_hard --use_oracle --eval_episodes=200 --max_path_length=70 --policy_updates_per_step=100 --explore_length=20 --network_layers=400,600,600,300 --display_plots --buffer_size=5000 --explore_episodes=10 --sample_new_goal_freq=10 --select_best_sample_size=1000 --select_goal_from_last_k_trajectories=-1 --stopped_thresh=0.05 --select_last_k_steps=-1 --max_timesteps=2000000 --fourier --repeat_previous_action_prob=0.25 --reward_layers=400,600,600,300 --fourier_goal_selector --goal_selector_num_samples=1000 --reset_free --goal_config=bimodal --use_reachable_set_autoregressive --reachable_thres=30 --remove_last_k_steps=9 --eval_episodes=10 --reachable_sample_rate=5 --buffer_random_init=1 --offset=500  --no_cond --reachable_thres=0.01 --autoregressive_epochs=100 --autoregressive_freq=1 --seed=0
```

## Development Notes

The directory structure currently looks like this:

- gear (Contains all code)
    - envs (Contains all environment files and wrappers)
    - algo (Contains all GEAR code)
        - gear.py (implements high-level algorithm logic, e.g. data collection, policy update, evaluate, save data)
        - buffer.py (The replay buffer used to *relabel* and *sample* (s,g,a,h) tuples
        - networks.py (Implements neural network policies.)
        - variants.py (Contains relevant hyperparameters for GEAR)

- doodad (We require this old version of doodad)
- dependencies (Contains other libraries like rlkit, rlutil, room_world, multiworld, etc.)

Please file an issue if you have trouble running this code.

