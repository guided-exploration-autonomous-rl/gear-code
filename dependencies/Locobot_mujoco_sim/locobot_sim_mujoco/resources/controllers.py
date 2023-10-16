import mujoco_py
import numpy as np
import math
import time
from scipy.spatial.transform import Rotation

class LoCoBot:
    def __init__(self, sim, viewer, model):
        self.sim = sim
        self.viewer = viewer
        self.model = model

        self.id = self.model.body_name2id("locobot")
        self.joint_ids = []

        for i in range(len(self.model.joint_names)):
            if self.model.jnt_bodyid[i] == self.id:
                self.joint_ids.append(i)


        # self.slide_joint_id = self.model.joint_name2id("advance")

    def get_current_pos(self):
        return np.array(self.sim.data.get_body_xpos("locobot")[:2])

    def get_current_angle(self):
        orientation = self.sim.data.get_body_xquat("locobot")
        rot = Rotation.from_quat(orientation)
        return rot.as_euler('xyz', degrees=True)[0]

    def smallest_angle(self, theta1, theta2):
        diff = abs(theta1 - theta2)
        return min(diff, 360 - diff)

    def rotate(self, theta, show = False):
        """
        Rotate theta degrees. We assume that with one step the angle of rotation
        is small enough (< min(180, theta)).
        """
        theta %= 360
        if (theta == 0):
            return

        if theta > 180:
            theta = -360 + theta

        elif theta < -180:
            theta = 360 + theta

        cw = theta > 0

        last_angle = self.get_current_angle()
        delta = 0
        iter = 0
        while delta < abs(theta):
            if cw:
                self.sim.data.ctrl[:3] = [0, 0, -15]
            else:
                self.sim.data.ctrl[:3] = [0, 0, 15]

            self.sim.step()

            # if show:
            #     self.viewer.render()

            current_angle = self.get_current_angle()
            iter +=1

            # Prevent getting stuck
            if iter > 25:
                return
            
            delta += self.smallest_angle(last_angle, current_angle)
            last_angle = current_angle

        # print("ANGLE", iter, delta)

    def advance(self, dist, show = False):
        """
        Move dist units. If dist is negative, then move backwards abs(dist) units
        """

        theta = self.get_current_angle() * math.pi / 180

        # if euler = pi/2 if it is 0 then is -cos, sin

        acty = 20 * math.cos(theta)
        actx = 20 * math.sin(theta)

        if dist == 0:
            return
        
        forward = dist > 0

        dist = abs(dist)
        
        last_pos = self.get_current_pos()
        iter = 0
        delta = 0
        while delta < dist:
            if forward:
                self.sim.data.ctrl[:3] = [actx, acty, 0]
            else:
                self.sim.data.ctrl[:3] = [-actx, -acty, 0]

            self.sim.step()

            # if show:
            #     self.viewer.render()

            current_pos = self.get_current_pos()
            iter +=1

            # Prevent getting stuck
            if iter > 50:
                return
            
            delta += np.linalg.norm(last_pos - current_pos) 
            last_pos = current_pos

        # print("DIST", iter, delta)

    def reduce_friction(self):
        for joint_id in self.joint_ids:
            self.sim.data.qvel[joint_id] = 0


    def do_action(self, action):
        if action == 0:
            self.advance(0.25)
        if action == 1:
            self.advance(-0.25)
        if action == 2:
            self.rotate(30)
        if action == 3:
            self.rotate(-30)

        self.reduce_friction()

    def get_base_pos_and_yaw(self):
        """
        Returns [x, y, theta] (theta in (-1, 1])
        """
        pos = self.get_current_pos()
        angle = self.get_current_angle()

        return np.array([pos[0], pos[1], angle/180.0])