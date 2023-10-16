import gym
import numpy as np
import pybullet as p
from locobot_sim.resources.env import Environment
import time
from time import sleep
import yaml
import math
from locobot_sim.resources.controllers import Locobot, Viewer
from locobot_sim.resources.teleop import KeyboardController
from pathlib import Path

from gym.spaces import Discrete, Dict, box

class LocobotSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):
        self.action_space = Discrete(4)

        self.obs_space = box.Box(
            low=np.array([-3, -2, -3.15], dtype=np.float32),
            high=np.array([3, 2, 3.15], dtype=np.float32))
        
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.obs_space),
            ('achieved_goal', self.obs_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.obs_space),
            ('state_achieved_goal', self.obs_space),
        ])

        self.threshold = 0.05
        self.sample_goal()

        resources_path = str(Path.cwd()) + "/dependencies/Locobot-sim/locobot_sim/resources/"
        self.room_settings = yaml.load(open(resources_path + "room_settings.yaml", 'r'), Loader=yaml.Loader)
        print(self.room_settings)

        # Initiallize connection
        if self.room_settings["render"]:
            self.client = p.connect(p.GUI)
        else:
            print("Using direct mode. Don't train the policy on images when using this mode!!")
            self.client = p.connect(p.DIRECT)

        self.environment = Environment(self.room_settings)
        camera_w = self.room_settings["room"]["cameras"]["w"]
        camera_h = self.room_settings["room"]["cameras"]["h"]

        self.locobot = Locobot(pos = self.room_settings["robot"]["pos"], robot_urdf = resources_path + self.room_settings["robot"]["urdf"], w = camera_w, h = camera_h)
        for _ in range(3):
            self.locobot.action([0, 0, -1, 0, 0, 0, 0])

        self.initial_state = p.saveState()
        self.top_camera = Viewer(p, [0, 0, 5.5], [0, 0.1, -1], fov=60, near_pos=0.05, far_pos=20.0)

    def step(self, action):
        self.perform_action(action)

        state = self._get_obs()
        done = False
        reward = 0

        return state, reward, done, dict()

    def perform_action(self, action):
        if action == 0:
            self.locobot.rotate_01()  
        elif action == 1:
            self.locobot.rotate_02()          
        elif action == 2:
            self.locobot.advance_01()          
        elif action == 3:
            self.locobot.advance_02()          

    def reset(self):
        p.restoreState(self.initial_state) 
        return self._get_obs()

    def _get_obs(self):
        state_obs = self.get_state()
        achieved_state_goal = state_obs.copy()
        intended_state_goal = self.goal.copy()

        obs = state_obs.copy()
    
        achieved_goal = achieved_state_goal.copy()
        intended_goal = intended_state_goal.copy()
            
        return dict(
            observation=obs,
            desired_goal=intended_goal,
            achieved_goal=achieved_goal,
            state_observation=state_obs,
            state_desired_goal=intended_state_goal,
            state_achieved_goal=achieved_state_goal,
        )

    def get_state(self):
        return self.locobot.get_base_pos_and_yaw()

    def render(self, mode = "rgb_array", width = 512, height = 512, camera_id = 0):
        return self.top_camera.get_image(width=width, height=height)[:,:,:3]
    
    def close(self):
        p.disconnect(self.client)
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def render_image(self):
        return self.render()
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def observation(self, state):
        return state
    
    def extract_goal(self, state):
        return state
    
    def get_shaped_distance(self, state1, state2):
        # st = time.time()
        return self.environment.shortest_path_distance(state1[:2], state2[:2])
        print("DISTANCE TOOK", time.time() - st)
        return d
    
    def compute_shaped_distance(self, state1, state2):
        return self.get_shaped_distance(state1, state2)
    
    def sample_goal(self):
        """
        Returns a goal as [x, y, theta]
        """
        self.goal = np.array([-2.25, 1.25, 0])
        return self.goal
        
    def plot_trajectories(self):
        return

    def compute_success(self, state, goal):
        return self.get_shaped_distance(state, goal) < self.threshold