import mujoco_py
import numpy as np
import time
import random
import math

from queue import PriorityQueue
from pathlib import Path
from collections import deque 
import sys
np.set_printoptions(threshold=sys.maxsize)

class Environment:
    def __init__(self, env_name = "locobot"):
        path = str(Path.cwd()) + f"/dependencies/Locobot_mujoco_sim/locobot_sim_mujoco/resources/data/{env_name}.xml"

        self.model = mujoco_py.load_model_from_path(path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None #mujoco_py.MjViewer(self.sim)
        self.sim.step()

        self.w = 2.5
        self.d = 3.5
        self.resolution = 0.05

        self.r = 0
        self.num_bodies = self.model.nbody

        self.initialize_grid()
        self.init_rewards([-2.25, 1.25])

        self.init_state = self.sim.get_state()

    def reset(self):
        self.sim.set_state(self.init_state)
        self.sim.forward()

    def get_image(self, w = 640, h = 480):
        image = self.sim.render(w, h, camera_name='camera')
        return np.flip(image, axis = 0)

    def initialize_grid(self):
        """
        Initiallizes a grid that tells which positions of the map are empty (traversable)
        and which aren't. In particular self.grid[0, 0] represents the position with the least
        x and least y and, self.grid[x+1][y+1] is the corresponding cell to self.grid[x][y]
        after moving self.resolution  
        """

        d = self.d # x coord
        w = self.w # y coord
        resolution = self.resolution
        x_start = round(d / resolution)
        y_start = round(w / resolution)

        self.grid = np.full((2 * x_start + 1, 2 * y_start + 1), 1)

        # Do it for radius r
        for i in range(len(self.grid)):
            self.grid[i][0] = 0
            self.grid[i][-1] = 0

        for i in range(len(self.grid[0])):
            self.grid[0][i] = 0
            self.grid[-1][i] = 0

        body_to_geom = [[] for _ in range(self.num_bodies)]

        for i in range(self.model.ngeom):
            body_to_geom[self.model.geom_bodyid[i]].append(i)
            
        for i in range(self.num_bodies):
            # print(self.model)
            body_name = self.model.body_id2name(i)
            
            if body_name in ["world", "locobot", "walls"] or "locobot" in body_name:
                continue

            for geom_id in body_to_geom[i]:
                bounding_radius = self.model.geom_rbound[geom_id]
                x, y = self.sim.data.geom_xpos[geom_id][:2]

                # print(body_name, geom_id, x, y, bounding_radius, "discrete = ", self.grid.shape, x_start -1 + int(x/resolution), y_start - 1 + int(y/resolution))

                min_x = x - bounding_radius * 0.5
                min_y = y - bounding_radius

                max_x = x + bounding_radius * 0.5
                max_y = y + bounding_radius

                min_x = max(min_x - self.r, -d)
                max_x = min(max_x + self.r, d)
                min_y = max(min_y - self.r, -w)
                max_y = min(max_y + self.r, w)

                first_x = max(0, x_start -1 + int(min_x/resolution))
                last_x = min(len(self.grid), x_start + 1 + int(max_x/resolution))
                first_y = max(0, y_start - 1 + int(min_y/resolution))
                last_y = min(len(self.grid[0]), y_start + 1 + int(max_y/resolution))

                for x in range(first_x, last_x):
                    for y in range(first_y, last_y):
                        self.grid[x][y] = 0

        # print(self.grid)

    def init_rewards(self, start): # Uses Manhattan distance
        d = self.d # x coord
        w = self.w # y coord
        resolution = self.resolution
        x_start = round(d / resolution)
        y_start = round(w / resolution)

        self.rewards = np.full((2 * x_start + 1, 2 * y_start + 1), 10.0)

        start_x = int(round((start[0] + self.d) / self.resolution))
        start_y = int(round((start[1] + self.w) / self.resolution))

        visited = np.full(self.grid.shape, 0)
        queue = PriorityQueue()

        queue.put([0, start_x, start_y])
        movex = [-1, 0, 1]
        movey = [-1, 0, 1]

        sq = math.sqrt(2)
        it = 1

        while it > 0:
            d, x, y = queue.get()
            it -= 1

            if visited[x][y]:
                continue

            visited[x][y] = 1
            self.rewards[x][y] = float(d) * resolution

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

                    queue.put([newd, new_x, new_y])

        dist = self.rewards
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.rewards[i][j] == 10:
                    best_approx = self.compute_approx_reward(i, j)
                    dist[i][j] = best_approx

        self.rewards = dist

    def compute_approx_reward(self, xs, ys):
        visited = np.full(self.grid.shape, 0)
        queue = deque()

        queue.append([xs, ys])
        visited[xs][ys] = 1
        movex = [-1, 0, 1]
        movey = [-1, 0, 1]

        while len(queue) > 0:
            x, y = queue.popleft()

            for i in range(len(movex)):
                new_x = x + movex[i]
                new_y = y + movey[i]

                if new_x < 0 or new_x == self.grid.shape[0] or new_y < 0 or new_y == self.grid.shape[1]:
                    continue

                # borders whould always be 0 (closed space)
                if (self.rewards[new_x][new_y] != 10):
                    return self.rewards[new_x][new_y]

                if (visited[new_x][new_y]):
                    continue

                visited[new_x][new_y] = 1
                queue.append([new_x, new_y])

        return 10

    def shaped_reward(self, start, end):
        # print(start, end, flush = True)
        start_x = int(round((start[0] + self.d) / self.resolution))
        start_y = int(round((start[1] + self.w) / self.resolution))
        final_x = int(round((end[0] + self.d) / self.resolution))
        final_y = int(round((end[1] + self.w) / self.resolution))
        return abs(self.rewards[start_x][start_y] - self.rewards[final_x][final_y])

    def shortest_path_distance(self, start, end):
        """"
        returns vector with the positions in the shortest path between start and end.
        """

        start_x = int(round((start[0] + self.d) / self.resolution))
        start_y = int(round((start[1] + self.w) / self.resolution))
        final_x = int(round((end[0] + self.d) / self.resolution))
        final_y = int(round((end[1] + self.w) / self.resolution))

        # You have to start at a position that is not inside a bounding box
        if not self.grid[start_x][start_y]:
            if not self.grid[final_x][final_y]:
                # return "inf"
                return 10
            else: 
                return self.shortest_path_distance(end, start)

        visited = np.full(self.grid.shape, 0)
        sh = list(list(self.grid.shape) + [2])
        dist = np.full((sh), -1) 
        queue = PriorityQueue()

        queue.put([0, start_x, start_y])
        movex = [-1, 0, 1]
        movey = [-1, 0, 1]

        sq = math.sqrt(2)
        it = 1

        while it > 0:
            d, x, y = queue.get()
            it -= 1

            if visited[x][y]:
                continue

            visited[x][y] = 1
            dist[x][y] = d

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

                    queue.put([newd, new_x, new_y])

        # If not found, do a BFS to the closest position that has an assigned distance
        queue = deque()

        queue.append([final_x, final_y])
        visited[final_x][final_y] = 1

        while len(queue) > 0:
            x, y, d = queue.popleft()

            for i in range(len(movex)):
                new_x = x + movex[i]
                new_y = y + movey[i]

                if new_x < 0 or new_x == self.grid.shape[0] or new_y < 0 or new_y == self.grid.shape[1]:
                    continue

                # borders whould always be 0 (closed space)
                if (dist[new_x][new_y] != -1):
                    return dist[new_x][new_y] * self.resolution

                if (visited[new_x][new_y]):
                    continue

                visited[new_x][new_y] = 1
                queue.append([new_x, new_y])

        return 10