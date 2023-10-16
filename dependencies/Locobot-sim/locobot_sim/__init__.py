from gym.envs.registration import register

register(
    id="LocobotSim-v0", 
    entry_point="locobot_sim.envs:LocobotSimEnv"
)