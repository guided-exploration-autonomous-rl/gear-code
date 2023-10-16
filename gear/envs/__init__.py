"""Helper functions to create envs easily through one interface.

- create_env(env_name)
- get_goal_threshold(env_name)
"""


try:
    import mujoco_py
except:
    print('MuJoCo must be installed.')

from gear.envs.locobot_env import LoCoBotEnv
from gear.envs.locobot_env_mujoco import LoCoBotEnvMujoco
from gear.envs.room_env import PointmassGoalEnv
from gear.envs.sawyer_push import SawyerPushGoalEnv
from gear.envs.sawyer_push_hard import SawyerHardPushGoalEnv
from gear.envs.kitchen_env_sequential import KitchenSequentialGoalEnv


from gear.algo import variants
env_names = ['locobot_mujoco_simple', 'locobot_mujoco', 'locobot', 'ravens','ravens_stack_blocks', 'complex_maze', 'ravens_stack_block_continuous', 'ravens_pick_place', 'ravens_pick_or_place', 'ravens_reaching', 'ravens_simple','ravens_continuous', 'kitchen', 'kitchen3D', 'kitchenSeq', 'pointmass_rooms', 'pointmass_empty', 'pointmass_maze', 'pusher', 'pusher_hard', 'lunar', 'door', 'claw']


def create_env_parallel(env_name, env_params, task_config="slide_cabinet,microwave", num_blocks=1, deterministic_rollout=False):
    """Helper function."""
    assert env_name in env_names
    if env_name == 'pusher':
        env = SawyerPushGoalEnv()
    elif env_name == "locobot":
        env = LoCoBotEnv()   
    elif env_name == "locobot_mujoco":
        env = LoCoBotEnvMujoco("locobot_mujoco")   
    elif env_name == "locobot_mujoco_simple":
        env = LoCoBotEnvMujoco("locobot_mujoco_simple")   
    elif env_name == "kitchenSeq":
        env = KitchenSequentialGoalEnv(task_config=task_config)
    elif env_name == 'pointmass_empty':
        env = PointmassGoalEnv(room_type='empty')
    elif env_name == 'pointmass_rooms':
        print("Point mass rooms")
        env = PointmassGoalEnv(room_type='rooms', fixed_goal=True)
    elif env_name == 'pointmass_maze':
        print("Point mass maze")
        env = PointmassGoalEnv(room_type='maze')
    elif env_name == 'pointmass_rooms_large':
        print("Point mass rooms large")
        env = PointmassGoalEnv(room_type='rooms')
    elif env_name == 'pusher_hard':
        if deterministic_rollout:
            return SawyerHardPushGoalEnv( fixed_start=True, fixed_goal=True)
        else:
            return SawyerHardPushGoalEnv()


    return variants.discretize_environment(env, env_params)

"""
def create_env_parallel(env_name, num_envs=4, task_config="slide_cabinet, microwave"):
    type_env = None
    params = {}
    if env_name == 'pusher':
        type_env = SawyerPushGoalEnv
    elif env_name == "kitchen":
        params['task_config'] = task_config
        type_env =  KitchenGoalEnv   
    elif env_name == "kitchen3D":
        type_env =  Kitchen3DGoalEnv
    elif env_name == "kitchenSeq":
        params['task_config'] = task_config
        type_env =  KitchenSequentialGoalEnv
    elif env_name == 'door':
        type_env =  SawyerDoorGoalEnv
    elif env_name == 'pointmass_empty':
        params['room_type'] = 'empty'
        type_env =  PointmassGoalEnv
    elif env_name == 'pointmass_rooms':
        print("Point mass rooms")
        params['room_type'] = 'rooms'
        type_env =  PointmassGoalEnv
    elif env_name == 'pointmass_maze':
        print("Point mass maze")
        params['room_type'] = 'maze'
        type_env =  PointmassGoalEnv
    elif env_name == 'pointmass_rooms_large':
        print("Point mass rooms large")
        params['room_type'] = 'rooms'
        type_env = PointmassGoalEnv
    elif env_name == 'lunar':
        type_env =  LunarEnv
    elif env_name == 'claw':
        type_env =  ClawEnv
    elif env_name == 'pusher_hard':
        type_env = SawyerHardPushGoalEnv

    if type_env is None:
        print(f"Error: {env_name} not defined for parallel environment")
        assert False

    env = make_vec_env(type_env,vec_env_cls=SubprocVecEnv,  n_envs=num_envs)

    return env
"""


def create_env(env_name, task_config="slide_cabinet,microwave", num_blocks=1, random_goal=False, maze_type=0, pick_or_place=False,continuous_action_space=False, goal_threshold=-1, goal=None, deterministic_rollout=False, goal_config='all', fix_reset=True, env_version=0):
    """Helper function."""
    assert env_name in env_names
    if env_name == 'pusher':
        return SawyerPushGoalEnv()
    elif env_name == "locobot":
        return LoCoBotEnv()   
    elif env_name == "locobot_mujoco":
        return LoCoBotEnvMujoco("locobot_mujoco")   
    elif env_name == "locobot_mujoco_simple":
        return LoCoBotEnvMujoco("locobot_mujoco_simple")   
    elif env_name == "kitchenSeq":
        return KitchenSequentialGoalEnv(task_config=task_config)
    elif env_name == 'pointmass_empty':
        return PointmassGoalEnv(room_type='empty', goal_config=goal_config)
    elif env_name == 'pointmass_rooms':
        print("Point mass rooms")
        return PointmassGoalEnv(room_type='rooms', goal_config=goal_config)
    elif env_name == 'pointmass_maze':
        print("Point mass maze")
        return PointmassGoalEnv(room_type='maze', goal_config=goal_config)
    elif env_name == 'pointmass_rooms_large':
        print("Point mass rooms large")
        return PointmassGoalEnv(room_type='rooms', goal_config=goal_config)
    elif env_name == 'pusher_hard':
        if deterministic_rollout:
            return SawyerHardPushGoalEnv(fixed_start=fix_reset, goal=goal, env_version=env_version)
        else:
            return SawyerHardPushGoalEnv(fixed_start=fix_reset , goal=goal, env_version=env_version)

def get_env_params(env_name, images=False):
    assert env_name in env_names

    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_trajectory_length=50,
        max_timesteps=1e6,
    )

    if env_name == 'pusher':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'locobot':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'locobot_mujoco':
        env_specific_params = dict(
            goal_threshold=0.25,
        )
    elif env_name == 'locobot_mujoco_simple':
        env_specific_params = dict(
            goal_threshold=0.25,
        )
    elif env_name == 'pusher_hard':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'complex_maze':
        env_specific_params = dict(
            goal_threshold=0.2,
        )
    elif env_name == 'ravens':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ravens_continuous':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ravens_stack_block_continuous':
        env_specific_params = dict(
            goal_threshold=0.15,
        )
    elif env_name == 'ravens_simple':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ravens_reaching':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ravens_pick_place':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ravens_pick_or_place':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ravens_stack_blocks':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif 'pick' in env_name:
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'door':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif 'pointmass' in env_name:
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )
    elif env_name == 'lunar':
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )
    elif env_name == 'claw':
        env_specific_params = dict(
            goal_threshold=0.1,
        )
    elif env_name == 'kitchen':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'kitchen3D':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'kitchenSeq':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    else:
        raise NotImplementedError()
    
    base_params.update(env_specific_params)
    return base_params