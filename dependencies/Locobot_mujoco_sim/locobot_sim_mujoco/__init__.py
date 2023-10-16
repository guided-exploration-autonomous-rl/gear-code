from gym.envs.registration import register

register(
    id="LocobotSimMujoco-v0", 
    entry_point="locobot_sim_mujoco.envs:LocobotSimMujocoEnv"
)