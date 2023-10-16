import math
import os
import pybullet as p
from time import sleep
import yaml
from locobot_sim.resources.controllers import Locobot
from locobot_sim.resources.controllers import Viewer
from locobot_sim.resources.teleop import KeyboardController
import numpy as np
import random

from queue import PriorityQueue

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

from collections import deque

CURR_PATH = os.path.dirname(os.path.abspath(__file__))

class Environment:
    def __init__(self, params):
        random.seed(1111)
        self.params = params
        self.render = params["render"]

        self.d = self.params["room"]["depth"] # x coord
        self.w = self.params["room"]["width"] # y coord
        self.resolution = self.params["room"]["resolution"] 
        
        self.camera_w = self.params["room"]["cameras"]["w"]
        self.camera_h = self.params["room"]["cameras"]["h"]

        # Set default parameters
        p.setGravity(0, 0, -9.8)
        self.default_ori = p.getQuaternionFromEuler([0,0,0])

        # Load room objects:
        room_params = params["room"]

        # - floor
        floor_params = room_params["floor"]
        self.floor = p.loadURDF(os.path.join(CURR_PATH, floor_params["urdf"]), floor_params["pos"], globalScaling=floor_params["scale"])
        p.changeVisualShape(self.floor, -1, textureUniqueId=p.loadTexture(os.path.join(CURR_PATH, floor_params["texture"])))

        # - walls
        walls_params = room_params["walls"]
        self.walls = p.loadURDF(os.path.join(CURR_PATH, walls_params["urdf"]), walls_params["pos"], globalScaling=walls_params["scale"])
        for i in range(len(p.getVisualShapeData(self.walls))):
            p.changeVisualShape(self.walls, i, rgbaColor=[0.58, 0.29, 0, 1])
            p.changeVisualShape(self.walls, i, textureUniqueId=p.loadTexture(os.path.join(CURR_PATH, walls_params["texture"])))

        # - robot
        # robot_params = params["robot"]
        # self.robot = p.loadURDF(robot_params["urdf"], robot_params["pos"], globalScaling=robot_params["scale"])

        # - other objects
        objects_params = room_params["objects"]

        self.objects = {}

        for object_name, object_params in objects_params.items():
            print("HERE", object_params)
            object_id = None 

            if "urdf" in object_params:
                object_id = p.loadURDF(os.path.join(CURR_PATH, object_params["urdf"]), object_params["pos"], baseOrientation=p.getQuaternionFromEuler(object_params["ori"]), globalScaling=object_params["scale"])
            else:
                object_id = p.loadSDF(os.path.join(CURR_PATH, object_params["sdf"]), object_params["pos"], baseOrientation=p.getQuaternionFromEuler(object_params["ori"]), globalScaling=object_params["scale"])
            

            if "texture" in object_params:
                p.changeVisualShape(object_id, -1, textureUniqueId=p.loadTexture(os.path.join(CURR_PATH, object_params["texture"])))
                for i in range(len(p.getVisualShapeData(object_id))):
                    p.changeVisualShape(object_id, i, textureUniqueId=p.loadTexture(os.path.join(CURR_PATH, object_params["texture"])))

            if "colour" in object_params:
                p.changeVisualShape(object_id, -1, rgbaColor=object_params["colour"])
                for i in range(len(p.getVisualShapeData(object_id))):
                    p.changeVisualShape(object_id, i, rgbaColor=object_params["colour"])

            self.objects[object_name] = object_id

        self.objects["walls"] = self.walls

        self.r = -0.1 #-0.2 #0.181 * params["robot"]["scale"] * 1.5 # radius of robot base

        # carefull with high resolutions 
        self.initialize_grid()
        self.initialize_destinations()
        self.initialize_cameras()
        self.init_borders()

    def initialize_grid(self):
        """
        Initiallizes a grid that tells which positions of the map are empty (traversable)
        and which aren't. In particular self.grid[0, 0] represents the position with the least
        x and least y and, self.grid[x+1][y+1] is the corresponding cell to self.grid[x][y]
        after moving self.params["room"]["resolution"]  
        """

        d = self.d # x coord
        w = self.w # y coord
        resolution = self.resolution
        x_start = round(d / resolution)
        y_start = round(w / resolution)

        self.grid = np.full((2 * x_start + 1, 2 * y_start + 1), 1)

        for object_name, object_id in self.objects.items():
            for i in range(-1, p.getNumJoints(object_id)):
                
                if object_name == "walls" and i == -1:
                    continue

                minimum, maximum = p.getAABB(object_id, i)

                min_x, min_y, _ = minimum
                max_x, max_y, _ = maximum

                r = self.r
                if object_name == "walls":
                    r = 0

                min_x = max(min_x - r, -d)
                max_x = min(max_x + r, d)
                min_y = max(min_y - r, -w)
                max_y = min(max_y + r, w)


                first_x = max(0, x_start -1 + int(min_x/resolution))
                last_x = min(len(self.grid), x_start + 1 + int(max_x/resolution))
                first_y = max(0, y_start - 1 + int(min_y/resolution))
                last_y = min(len(self.grid[0]), y_start + 1 + int(max_y/resolution))

                for x in range(first_x, last_x):
                    for y in range(first_y, last_y):
                        self.grid[x][y] = 0


        #         print(object_name, i, first_x, last_x, first_y, last_y)

        # print(self.grid)

    def init_borders(self):
        self.borders = []
        x_length = len(self.grid)
        y_length = len(self.grid[0])

        for x in range(x_length):
            found = False
            y = 0

            while not found and y < y_length:
                if self.grid[x][y]:
                    self.borders.append([-self.d + x * self.resolution, -self.w + y * self.resolution])
                    found = True
                y += 1
        
        for x in range(x_length):
            found = False
            y = y_length - 1

            while not found and y >= 0:
                if self.grid[x][y]:
                    self.borders.append([-self.d + x * self.resolution, -self.w + y * self.resolution])
                    found = True
                y -= 1

        for y in range(y_length):
            found = False
            x = 0

            while not found and x < x_length:
                if self.grid[x][y]:
                    self.borders.append([-self.d + x * self.resolution, -self.w + y * self.resolution])
                    found = True
                x += 1

        for y in range(y_length):
            found = False
            x = x_length - 1

            while not found and x >= 0:
                if self.grid[x][y]:
                    self.borders.append([-self.d + x * self.resolution, -self.w + y * self.resolution])
                    found = True
                x -= 1
            
    def init_rewards(self, start): # Uses Manhattan distance
        """"
        returns vector with the positions in the shortest path between start and end.
        """
        
        queue = deque()

        start_x = round((start[0] + self.d) / self.resolution)
        start_y = round((start[1] + self.w) / self.resolution)

        self.rewards = np.full(self.grid.shape, -1e9)
        self.rewards[start_x][start_y] = 0

        queue.append([start_x, start_y, 0])
        movex = [1, -1, 0, 0]
        movey = [0, 0, 1, -1]

        while len(queue) > 0:
            x, y, d = queue.popleft()

            for i in range(len(movex)):
                new_x = x + movex[i]
                new_y = y + movey[i]

                # borders whould always be 0 (closed space)
                if (not self.grid[new_x][new_y]):
                    continue

                if (self.rewards[new_x][new_y] > -1e8):
                    continue

                self.rewards[new_x][new_y] = -(d + 1) * self.resolution
                queue.append([new_x, new_y, d + 1])

    def current_reward(self, start):
        start_x = round((start[0] + self.d) / self.resolution)
        start_y = round((start[1] + self.w) / self.resolution)

        return self.rewards[start_x][start_y]

    def shortest_path_distance(self, start, end):
        """"
        returns vector with the positions in the shortest path between start and end.
        """

        visited = np.full(self.grid.shape, 0)
        sh = list(list(self.grid.shape) + [2])
        dirs = np.full((sh), 0) 
        queue = PriorityQueue()

        start_x = int(round((start[0] + self.d) / self.resolution))
        start_y = int(round((start[1] + self.w) / self.resolution))
        final_x = int(round((end[0] + self.d) / self.resolution))
        final_y = int(round((end[1] + self.w) / self.resolution))

        # print(start, start_x, start_y, self.grid[start_x, start_y])
        # print("and....")
        # print(end, final_x, final_y, self.grid[final_x, final_y])

        queue.put([0, start_x, start_y, 0, 0])
        movex = [-1, 0, 1]
        movey = [-1, 0, 1]

        sq = math.sqrt(2)
        it = 1

        while it > 0:
            d, x, y, pdx, pdy = queue.get()
            it -= 1

            # print("trying", x, y)
            if visited[x][y]:
                continue

            visited[x][y] = 1
            dirs[x][y] = [pdx, pdy]

            if (final_x == x and final_y == y):
                return d * self.resolution

            for dx in movex:
                for dy in movey:
                    if (dx == 0 and dy == 0):
                        continue

                    dd = 1
                    if (dx * dy != 0):
                        dd = sq

                    new_x = x + dx
                    new_y = y + dy
                    newd = d + dd

                    # borders whould always be 0 (closed space)
                    if (not self.grid[new_x][new_y]):
                        continue

                    if (visited[new_x][new_y]):
                        continue

                    it += 1

                    queue.put([newd, new_x, new_y, dx, dy])


        return 1e6

    def shortest_path(self, start, end):
        """"
        returns vector with the positions in the shortest path between start and end.
        """
        visited = np.full(self.grid.shape, 0)
        sh = list(list(self.grid.shape) + [2])
        dirs = np.full((sh), 0) 
        queue = PriorityQueue()

        start_x = round((start[0] + self.d) / self.resolution)
        start_y = round((start[1] + self.w) / self.resolution)
        final_x = round((end[0] + self.d) / self.resolution)
        final_y = round((end[1] + self.w) / self.resolution)

        queue.put([0, start_x, start_y, 0, 0])
        movex = [-1, 0, 1]
        movey = [-1, 0, 1]

        sq = math.sqrt(2)
        it = 1

        while it > 0:
            d, x, y, pdx, pdy = queue.get()
            it -= 1

            if visited[x][y]:
                continue

            visited[x][y] = 1
            dirs[x][y] = [pdx, pdy]

            if (final_x == x and final_y == y):
                queue = PriorityQueue()
                break

            for dx in movex:
                for dy in movey:
                    if (dx == 0 and dy == 0):
                        continue

                    dd = 1
                    if (dx * dy != 0):
                        dd = sq

                    new_x = x + dx
                    new_y = y + dy
                    newd = d + dd

                    # borders whould always be 0 (closed space)
                    if (not self.grid[new_x][new_y]):
                        continue

                    if (visited[new_x][new_y]):
                        continue

                    it += 1

                    queue.put([newd, new_x, new_y, dx, dy])


        if (not visited[final_x][final_y]):
            # The position is not reachable
            return None 

        cx = final_x
        cy = final_y 

        ans = []

        last_dx = 0
        last_dy = 0
        first = True
        while (cx != start_x or cy != start_y):
            dx, dy = dirs[cx][cy]
            pt = [-self.d + cx * self.resolution, -self.w + cy * self.resolution]
            dist = 0
            if (len(ans)):
                dist = math.sqrt((ans[-1][0] - pt[0]) ** 2 + (ans[-1][1] - pt[1]) ** 2)


            if (first or dist >= 0.5):
                ans.append(pt)

            first = False
            
            last_dx = dx
            last_dy = dy

            cx -= dx
            cy -= dy

        ans.reverse()
        return ans

    def initialize_destinations(self):
        self.tasks = []

        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                if (self.grid[x][y]):
                    self.tasks.append([-self.d + x * self.resolution, -self.w + y * self.resolution])

    def get_random_task(self):
        return self.tasks[random.randrange(len(self.tasks))]

    def get_random_border(self):
        return self.borders[random.randrange(len(self.borders))]

    def initialize_cameras(self):
        # self.cameras = [Viewer(p, [-5, -5, 5], [5, 5, -5], fov=60, near_pos=0.05, far_pos=20.0),
        #                 Viewer(p, [-5, 5, 5], [5, -5, -5], fov=60, near_pos=0.05, far_pos=20.0),
        #                 Viewer(p, [5, -5, 5], [-5, 5, -5], fov=60, near_pos=0.05, far_pos=20.0),
        #                 Viewer(p, [5, 5, 5], [-5, -5, -5], fov=60, near_pos=0.05, far_pos=20.0)]
        self.cameras = [Viewer(p, [-3, 2, 2], [3, -2, -2], fov=60, near_pos=0.05, far_pos=20.0),
                        Viewer(p, [3, -2, 2], [-3, 2, -2], fov=60, near_pos=0.05, far_pos=20.0)]

    def get_obs(self):
        return [x.get_image(width=self.camera_w, height=self.camera_h)[:,:,:3] for x in self.cameras]


    def find_closest_reachable_point(self, x, y):
        """
        At least with a distance of 0.1 to the object
        """

        visited = np.full(self.grid.shape, -1) 
        queue = deque()

        start_x = round((x + self.d) / self.resolution)
        start_y = round((y + self.w) / self.resolution)

        queue.append([start_x, start_y])
        movex = [1, -1, 0, 0]
        movey = [0, 0, 1, -1]

        while len(queue) > 0:
            x, y = queue.popleft()

            for i in range(len(movex)):
                new_x = x + movex[i]
                new_y = y + movey[i]

                if (new_x < 0 or new_y < 0 or new_x == self.d or new_y == self.w):
                    continue

                # reachable point
                if (self.grid[new_x][new_y]):
                    if (self.resolution * math.sqrt((new_x - start_x) ** 2 + (new_y - start_y) ** 2) > 0.1):
                        return -self.d + new_x * self.resolution, -self.w + new_y * self.resolution

                if (visited[new_x][new_y] != -1):
                    continue

                visited[new_x][new_y] = 0
                queue.append([new_x, new_y])

        return None

    def get_object_grabbing_pos(self, object_name, link = -1):
        """
        Returns three values pos, grab, angle
        Where
            * pos is the position that the robot should go to grab the object
            * grab is the tuple (x, y, z) of the grabbing point
            * angle is the angle between e3 and the bounding box of the link when located at pos
            * height is the maximum value of z
        """

        object_id = self.objects[object_name]
        minimum, maximum = p.getAABB(object_id) # bounding box

        min_x, min_y, min_z = minimum
        max_x, max_y, max_z = maximum

        # Find exact grabbing location

        # grab = 0.5 * (np.array(minimum) + np.array(maximum))

        base_pos, base_ori = p.getBasePositionAndOrientation(object_id)
        base_ori = p.getEulerFromQuaternion(base_ori)[2]

        # Do BFS to find closest point from where to grab it
        
        x, y = self.find_closest_reachable_point(base_pos[0], base_pos[1])

        return [x, y], base_pos, base_ori, max_z




# client = p.connect(p.GUI)
# room_settings = yaml.load(open(os.path.join(CURR_PATH, "room_settings.yaml"), 'r'), Loader=yaml.Loader)
# print(room_settings)
# environment = Environment(room_settings)

# robot = Locobot(robot_urdf = os.path.join(CURR_PATH, room_settings["robot"]["urdf"]))

# def interact():
#     keyboard = KeyboardController()
#     while True:
#         action = keyboard.get_teleop_action()
#         robot.action(action)

# interact()



# p.connect(p.GUI)
# p.setGravity(0, 0, -10)


# plane_id = p.loadURDF("data/structure/floor_patch.urdf", [0, 0, 0.05], globalScaling=10)
# p.changeVisualShape(plane_id, -1)
# p.changeVisualShape(plane_id, -1, textureUniqueId=load_texture("data/textures/floor2.png"))


# angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
# throttle = p.addUserDebugParameter('Throttle', 0, 20, 0)
# car = p.loadURDF('data/robots/locobot.urdf', [0, 0, 0.1])
# plane = p.loadURDF('plane.urdf')

# walls_id = p.loadURDF("data/structure/walls.urdf", globalScaling=10)
# # p.changeVisualShape(walls_id, -1, textureUniqueId=load_texture("data/textures/wood_texture.png"))

# for i in range(len(p.getVisualShapeData(walls_id))):
#     p.changeVisualShape(walls_id, i, rgbaColor=[0.58, 0.29, 0, 1])
#     p.changeVisualShape(walls_id, i, textureUniqueId=load_texture("data/textures/wood_texture.png"))


# sleep(1)

# wheel_indices = [1, 3, 4, 5]
# hinge_indices = [0, 2]

# while True:
#     user_angle = p.readUserDebugParameter(angle)
#     user_throttle = p.readUserDebugParameter(throttle)
#     for joint_index in wheel_indices:
#         p.setJointMotorControl2(car, joint_index,
#                                 p.VELOCITY_CONTROL,
#                                 targetVelocity=user_throttle)
#     for joint_index in hinge_indices:
#         p.setJointMotorControl2(car, joint_index,
#                                 p.POSITION_CONTROL, 
#                                 targetPosition=user_angle)
#     p.stepSimulation()