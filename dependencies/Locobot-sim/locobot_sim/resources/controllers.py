
import os
import time
from os.path import expanduser
import sys
import pprint
from numbers import Number
import math
import gym
import numpy as np
import pybullet as p

import moviepy.editor as mpy

import functools
import inspect

from gym import spaces
from gym.utils import seeding
import cv2
from skimage.transform import rescale


from transforms3d.quaternions import quat2mat

class Viewer:
    """ Wrapper for a pybullet camera. """
    def __init__(self, p, camera_pos, look_pos, fov=25, near_pos=.02, far_pos=1.):
        self.p = p
        self.proj_matrix = p.computeProjectionMatrixFOV(fov, 1, near_pos, far_pos)
        self.update(camera_pos, look_pos)

    def update(self, camera_pos, look_pos):
        self.camera_pos = np.array(camera_pos)
        self.look_pos = np.array(look_pos)
        self.look_pos[-1] += 0.2
        self.view_matrix = p.computeViewMatrix(self.camera_pos, self.look_pos, [0,0,1])

    def get_image(self, width, height, **kwargs):
        render_coefficient = 1

        _, _, unscaled_image, _, _ = p.getCameraImage(width * render_coefficient, height * render_coefficient, 
                                                  self.view_matrix, self.proj_matrix, **kwargs, shadow = False, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=0, flags = p.ER_NO_SEGMENTATION_MASK)

        image = rescale(unscaled_image, 1.0 / render_coefficient, 
                        multichannel=True, anti_aliasing=True, preserve_range=True).astype(np.uint8)

        return image #image[:,:,:3], unscaled_image.astype(np.uint8)


class Locobot:

    # From the view of the robot pointing forward: 
    ARM_JOINTS = [13, # arm base rotates left (+) and right (-), in radians 
                  14, # 1st joint (controls 1st arm segment). 0 points up, + bends down
                  15, # 2nd joint (controls 2nd arm segment). 0 perpendicular to 1st segment, + bends down (towards 1st segment)
                  16] # 3rd joint (controls wrist/hand arm segment). 0 inline with 2nd segment, + bends down (towards 2nd segment)
    WRIST_JOINT = 17  # + rotates left, - rotates right
    LEFT_GRIPPER = 19  # 0 is middle, +0.02 is extended
    RIGHT_GRIPPER = 18 # 0 is middle, -0.02 is extended
    LEFT_WHEEL = 1
    RIGHT_WHEEL = 2
    CAMERA_LINK = 25
    AUX_CAMERA_LINK = 26

    GRIPPER_LENGTH_FROM_WRIST = 0.115

    def __init__(self, pos = None, **params):
        defaults = {
            "renders": False, # whether we use GUI mode or not
            "grayscale": False, # whether we render in grayscale
            "step_duration": 1/60, # when in render mode, how long in seconds does each step take
            "start_arm_joints": np.array([0., -1.3, 1.58, 0.8]), # joint values for the neutral start position
            "pregrasp_pos": np.array([0.42, 0, 0.185]), # local coord for the end-effector pos to go to before grasping
            "down_quat": np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865476]), # quaternion for gripper to point downwards
            "camera_look_pos": np.array([0.5, 0., .2]), # local pos that the camera looks at
            "camera_fov": 60, # FOV of the camera
            "urdf_name": "locobot",
            "load_plane": True,
            "video_fps": 10,
            "w": 128,
            "h": 128,
        }
        scale = 1
        defaults.update(params)
        self.params = defaults

        self.w = self.params["w"]
        self.h = self.params["h"]
        print("W is", self.w)
        # self.camera_w = self.params["camera"]["width"]
        # self.camera_h = self.params["camera"]["height"]

        self.wheel_radius = 0.035 * scale
        self.wheel_axle_length = 0.356 * scale
        self.wheel_axle_halflength = self.wheel_axle_length/2.0

        print()
        print("LocobotInterface params:")
        pprint.pprint(dict(
            self=self,
            **self.params,
        ))
        print()

        self.renders = self.params["renders"]
        print("RENDERSS!!!!!", self.renders)
        self.grayscale = self.params["grayscale"]
        self.step_duration = self.params["step_duration"]

        # set up pybullet simulation
        # if self.renders:
        #     p.connect(p.GUI)
        #     print("LOCOBOT: STARTED RENDER MODE")
        # else:
        #     p.connect(p.DIRECT)
        
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setGravity(0, 0, -9.8)
        self.default_ori = p.getQuaternionFromEuler([0,0,0])

        if pos is None:
            pos = [2, 0, 0]

        # Load robot
        self.robot_urdf = self.params["robot_urdf"]
        self.robot = p.loadURDF(self.robot_urdf, pos, useFixedBase=0, globalScaling=scale)#, baseOrientation=p.getQuaternionFromEuler([0, 0, 4.65]))

        _, _, _, _, self.base_pos, _ = p.getLinkState(self.robot, 0)
        _, _, _, _, self.camera_pos, _ = p.getLinkState(self.robot, self.CAMERA_LINK)
        self.base_pos = np.array(self.base_pos)
        self.camera_pos = np.array(self.camera_pos)

        # print("-----")

        # print("pos", p.getBasePositionAndOrientation(self.robot))
        # print(p.getNumJoints(self.robot))
        # for i in range(-1, p.getNumJoints(self.robot)):
        #     print(p.getAABB(self.robot, i))

        # print("-----")

        # Create viewers
        self.camera = Viewer(p, self.camera_pos, self.params["camera_look_pos"], 
                                fov=self.params["camera_fov"], 
                                near_pos=0.05, far_pos=10.0)

        #self.Lidar = ScanSensor(self.robot)
        
        # create the second auxilary camera if specified
        if self.params.get("use_aux_camera", False):
            if "aux_camera_look_pos" not in self.params:
                self.params["aux_camera_look_pos"] = self.params["camera_look_pos"]
            if "aux_camera_fov" not in self.params:
                self.params["aux_camera_fov"] = self.params["camera_fov"]
            _, _, _, _, self.axu_camera_pos, _ = p.getLinkState(self.robot, self.AUX_CAMERA_LINK)
            self.axu_camera_pos = np.array(self.axu_camera_pos)
            self.aux_camera = Viewer(p, self.axu_camera_pos, self.params["aux_camera_look_pos"], 
                                    fov=self.params["aux_camera_fov"],
                                    near_pos=0.05, far_pos=20.0)

        self.total_sim_steps = 0

        self.video_fps = self.params["video_fps"]
        self.frames = []
        self.video_index = 0

        # while True:
        #     self.move_to_point([3, 1])

    def save_state(self):
        self.saved_state = p.saveState()
        jointStates = p.getJointStates(self.robot, self.ARM_JOINTS + [self.WRIST_JOINT, self.LEFT_GRIPPER, self.RIGHT_GRIPPER])
        self.saved_joints = [state[0] for state in jointStates]

    def reset(self):
        if self.p: 
            p.restoreState(stateId=self.saved_state)
            for i, joint in enumerate(self.ARM_JOINTS + [self.WRIST_JOINT, self.LEFT_GRIPPER, self.RIGHT_GRIPPER]):
                p.setJointMotorControl2(self.robot,joint,p.POSITION_CONTROL, self.saved_joints[i])
                p.setJointMotorControl2(self.robot,joint,p.VELOCITY_CONTROL,0)


    # ----- OBSERVATION METHODS -----

    def get_observation(self):
        img = self.render_camera()

        # img = self.camera.get_image(self.w, self.h) # TODO ADD DIMENSION
        
        rgb = img[:,:,:3]
        # depth = img[:,:,3]
        return rgb


    # ----- BASE METHODS -----

    def reset_robot(self, pos=[0, 0], yaw=0, left=0, right=0, steps=180):
        """ Reset the robot's position and move the arm back to start.
        Args:
            pos: (2,) vector. Assume that the robot is on the floor.
            yaw: float. Rotation of the robot around the z-axis.
            left: left wheel velocity.
            right: right wheel velocity.
        """
        self.set_base_pos_and_yaw(pos=pos, yaw=yaw)
        self.set_wheels_velocity(left, right)
        p.resetJointState(self.robot, self.LEFT_WHEEL, targetValue=0, targetVelocity=left)
        p.resetJointState(self.robot, self.RIGHT_WHEEL, targetValue=0, targetVelocity=right)
        self.move_arm_to_start(steps=steps, max_velocity=8.0)

    def get_base_pos_and_yaw(self):
        """ Get the base position and yaw. (x, y, yaw). """
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        return np.array([base_pos[0], base_pos[1], p.getEulerFromQuaternion(base_ori)[2]])

    def get_base_pos(self):
        """ Get the base position (x, y). """
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        return np.array([base_pos[0], base_pos[1]])

    def set_base_pos_and_yaw(self, pos=np.array([0.0, 0.0]), yaw=0.0):
        """ Set the base's position (x, y) and yaw. """
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        new_pos = [pos[0], pos[1], base_pos[2]]
        new_rot = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot, new_pos, new_rot)

    def set_wheels_velocity(self, left, right):
        p.setJointMotorControl2(self.robot, self.LEFT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=left)
        p.setJointMotorControl2(self.robot, self.RIGHT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=right)

    def get_wheels_velocity(self):
        _, left, _, _ = p.getJointState(self.robot, 1)
        _, right, _, _ = p.getJointState(self.robot, 2)
        return np.array([left, right])
    
    def reset_base(self):
        p.setJointMotorControl2(self.robot, self.LEFT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        p.setJointMotorControl2(self.robot, self.RIGHT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        self.do_steps(10)

    def move_base(self, left, right):
        """ Move the base by some amount. """
        p.setJointMotorControl2(self.robot, self.LEFT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=left, force=1e4)
        p.setJointMotorControl2(self.robot, self.RIGHT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=right, force=1e4)
        # self.do_steps(55)
        # p.setJointMotorControl2(self.robot, self.LEFT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        # p.setJointMotorControl2(self.robot, self.RIGHT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        # self.do_steps(65)

    def differential_drive(self, action):
        lin_vel, ang_vel = action

        # Convert to wheel velocities
        left_wheel_joint_vel = (lin_vel - ang_vel * self.wheel_axle_halflength) / self.wheel_radius
        right_wheel_joint_vel = (lin_vel + ang_vel * self.wheel_axle_halflength) / self.wheel_radius

        # Return desired velocities
        self.move_base(left_wheel_joint_vel, right_wheel_joint_vel)

    def current_pos_angle(self):
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        base_angle = p.getEulerFromQuaternion(base_ori)[-1]
        return base_pos[0], base_pos[1], base_angle

    def rotate(self, destination):
        x0, y0, base_angle = self.current_pos_angle()
        x1, y1 = destination

        dx = x1 - x0
        dy = y1 - y0

        look_x = math.cos(base_angle)
        look_y = math.sin(base_angle)

        rot_angle = math.atan2(dy, dx) - math.atan2(look_y, look_x) 

        if (rot_angle > math.pi):
            rot_angle -= 2 * math.pi
        elif (rot_angle < -math.pi):
            rot_angle += 2 * math.pi

        # w = 10 with 50 steps: theta = 0.989856220294992
        # w = 1 with 100 steps: theta = 0.578420832443751
        # w = 1 with 200 steps: theta = 1.2210246651933707

        w = rot_angle / 1.22
        steps = 200

        self.differential_drive([0, w])
        self.do_steps(steps) 

        return [0, w]

    def angle_distance(self, destination):
        x0, y0, base_angle = self.current_pos_angle()
        x1, y1 = destination

        dx = x1 - x0
        dy = y1 - y0

        look_x = math.cos(base_angle)
        look_y = math.sin(base_angle)

        rot_angle = math.atan2(dy, dx) - math.atan2(look_y, look_x) 

        if (rot_angle > math.pi):
            rot_angle -= 2 * math.pi
        elif (rot_angle < -math.pi):
            rot_angle += 2 * math.pi

        dist = math.sqrt(dx**2 + dy**2)

        return rot_angle, dist

    # rotates 0.2
    def rotate_01(self):
        self.differential_drive([0, 0.42])
        self.do_steps(95)
        self.reset_base()

    def rotate_02(self):
        self.differential_drive([0, -0.42])
        self.do_steps(95)
        self.reset_joints()
        self.reset_base()

    # moves 0.2
    def advance_01(self):
        self.differential_drive([0.31, 0])
        self.do_steps(170)
        self.reset_joints()
        self.reset_base()

    def advance_02(self):
        self.differential_drive([-0.31, 0])
        self.do_steps(170)
        self.reset_joints()
        self.reset_base()


    def advance(self, destination):
        x0, y0, _ = self.current_pos_angle()
        x1, y1 = destination

        dx = x1 - x0
        dy = y1 - y0

        advance = math.sqrt(dx**2 + dy**2)

        # With v = 0.25 it advances 0.00095 units per step 
        # With steps = 50 and v = 10 it advances 0.17888346905
        # With steps = 100 and v = 1 it advances 0.2695474998
        # with steps = 200 and v = 1 it advances 0.6635

        v = advance / 0.6635 #np.clip(advance / 0.6635, -0.5, 0.5)
        steps = 200

        self.differential_drive([v, 0])
        self.do_steps(steps) 

        return [v, 0]

    def move_to_point(self, destination, iterations = 1):
        for _ in range(iterations):
            x0, y0, base_angle = self.current_pos_angle()
            x1, y1 = destination

            dx = x1 - x0
            dy = y1 - y0
            
            look_x = math.cos(base_angle)
            look_y = math.sin(base_angle)

            rot_angle = math.atan2(dy, dx) - math.atan2(look_y, look_x) 

            if (rot_angle > math.pi):
                rot_angle -= 2 * math.pi
            elif (rot_angle < -math.pi):
                rot_angle += 2 * math.pi

            # With w = 1 it rotates around 0.0056 rad per step.

            w = math.copysign(1, rot_angle)
            steps = round(abs(rot_angle) / (iterations * 0.0056))

            self.differential_drive([0, w])
            self.do_steps(steps) 

            advance = math.sqrt(dx**2 + dy**2)

            # With v = 0.25 it advances 0.00095 units per step 

            v = 0.25
            steps = round(advance / (iterations * 0.00095))

            self.differential_drive([v, 0])
            self.do_steps(steps) 
        
        
    def follow_path(self, path):
        for point in path:
            self.move_to_point(point)


    # ----- END BASE METHODS -----



    # ----- ARM METHODS -----

    def fingers_position(self):
        _, _, _, _, pos_left, _ = p.getLinkState(self.robot, self.LEFT_GRIPPER)
        _, _, _, _, pos_right, _ = p.getLinkState(self.robot, self.RIGHT_GRIPPER)

        return (np.array(pos_left) + np.array(pos_right))/2

    def execute_grasp_direct(self, pos, wrist_rot=0.0):
        new_pos = np.array([pos[0], pos[1], 0.1])
        self.open_gripper(steps=0)
        self.move_ee(new_pos, wrist_rot, steps=70, max_velocity=8.0)
        new_pos[2] = 0
        self.move_ee(new_pos, wrist_rot, steps=40, max_velocity=8.0)
        self.close_gripper(steps=30)
        new_pos[2] = 0.1
        self.move_ee(new_pos, wrist_rot, steps=60, max_velocity=1.0)
        
    def execute_place_direct(self, pos, wrist_rot=0.0):
        new_pos = np.array([pos[0], pos[1], 0.1])
        
        self.move_ee(new_pos, wrist_rot, steps=60, max_velocity=12.0)
        new_pos[2] = 0
        self.move_ee(new_pos, wrist_rot, steps=30, max_velocity=12.0)
        self.open_gripper(steps=30)
        new_pos[2] = 0.1
        self.move_ee(new_pos, wrist_rot, steps=60, max_velocity=1.0)
        self.open_gripper(steps=0)

    def execute_grasp(self, pos, wrist_rot=0.0):
        """ Do a predetermined single grasp action by doing the following:
            1. Move end-effector to the pregrasp_pos.
            2. Offset the effector x,y postion by the given pos.
            3. Move end-effector downwards and grasp.
            4. Move end-effector upwards
        Args:
            pos: (x,y) location of the grasp with origin at the pregrasp_pos, local to the robot
            wrist_rot: wrist rotation in radians
        """
        new_pos = self.params["pregrasp_pos"].copy()
        self.open_gripper(steps=0)
        self.move_ee(new_pos, wrist_rot, steps=20)

        new_pos[:2] += np.array(pos)
        self.move_ee(new_pos, wrist_rot, steps=20)

        new_pos[2] = 0
        self.move_ee(new_pos, wrist_rot, steps=30, max_velocity=5.0)
        self.close_gripper()

        new_pos[2] = 0.185
        self.move_ee(new_pos, wrist_rot, steps=60, max_velocity=1.0)

    def move_ee(self, pos, wrist_rot=0, steps=30, max_velocity=float("inf"), ik_steps=256):
        """ Move the end-effector (tip of gripper) to the given pos, pointing down.
        Args:
            pos: (3,) vector local coordinate for the desired end effector position.
            wrist_rotate: rotation of the wrist in radians. 0 is the gripper closing from the sides.
            steps: how many simulation steps to do.
            max_velocity: the maximum velocity of the joints..
            ik_steps: how many IK steps to calculate the final joint values.
        """
        # print("MOVE_EE", pos)
        pos = (pos[0], pos[1], pos[2] - 0.05626175403 + self.GRIPPER_LENGTH_FROM_WRIST)
        print("POS AFTER CORRECTIOn", pos)
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        pos, ori = p.multiplyTransforms(base_pos, base_ori, pos, self.params["down_quat"])
        # ori = p.getQuaternionFromEuler([0,1,0])
        print("??????", pos, p.getEulerFromQuaternion(ori))

        # print("FINAL POS", pos, ori)
        # print("ANS", p.calculateInverseKinematics(self.robot, self.WRIST_JOINT, pos, ori, maxNumIterations=ik_steps))
        jointStates = p.calculateInverseKinematics(self.robot, self.WRIST_JOINT, pos, ori, maxNumIterations=int(1e6), residualThreshold=1e-10)[2:6]

        self.move_arm(jointStates, wrist_rot=wrist_rot, steps=steps, max_velocity=max_velocity)

        self.do_steps(steps)

    def move_ee_absolute(self, pos, wrist_rot=0, steps=30, max_velocity=float("inf"), ik_steps=256):
        """ Move the end-effector (tip of gripper) to the given pos, pointing down.
        Args:
            pos: (3,) vector local coordinate for the desired end effector position.
            wrist_rotate: rotation of the wrist in radians. 0 is the gripper closing from the sides.
            steps: how many simulation steps to do.
            max_velocity: the maximum velocity of the joints..
            ik_steps: how many IK steps to calculate the final joint values.
        """
        print("MOVE_EE_ABSOLUTE", pos)
        ori = p.getQuaternionFromEuler([0, 1.57, 0])
        
        # print("FINAL POS", pos, ori)
        # print("ANS", p.calculateInverseKinematics(self.robot, self.WRIST_JOINT, pos, ori, maxNumIterations=ik_steps))
        jointStates = p.calculateInverseKinematics(self.robot, self.WRIST_JOINT, pos, ori, maxNumIterations=int(1e5), residualThreshold=1e-8)[2:6]
        print(jointStates)
        self.move_arm(jointStates, wrist_rot=wrist_rot, steps=steps, max_velocity=max_velocity)
        for joint in self.ARM_JOINTS:
            print(p.getJointState(self.robot, joint))
        self.do_steps(steps)

        _, _, _, _, ee_pos, ee_ori = p.getLinkState(self.robot, self.WRIST_JOINT)
        print("AFTER", ee_pos)

    def move_arm(self, arm_joint_values, wrist_rot=None, steps=60, max_velocity=float("inf")):
        """ Move the arms joints to the given joints values.
        Args:
            pos: (4,) vector of the 4 arm joint pos
            wrist_rot: If not None, rotate wrist to wrist rot.
            steps: how many simulation steps to do
            max_velocity: the maximum velocity of the joints
        """
        for joint, value in zip(self.ARM_JOINTS, arm_joint_values):
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, value, maxVelocity=max_velocity)
        
        if wrist_rot is not None:
            p.setJointMotorControl2(self.robot, self.WRIST_JOINT, p.POSITION_CONTROL, wrist_rot, maxVelocity=max_velocity)
        
        #self.do_steps(steps) #TODO 


    def move_arm_to_start(self, wrist_rot=None, steps=60, max_velocity=float("inf")):
        """ Move the arms joints to the start_joints position
        Args:
            steps: how many simulation steps to do
        """
        self.move_arm(self.params["start_arm_joints"], wrist_rot=wrist_rot, steps=steps, max_velocity=max_velocity)

    def rotate_wrist(self, wrist_rot, steps=30, max_velocity=float("inf")):
        p.setJointMotorControl2(self.robot, self.WRIST_JOINT, p.POSITION_CONTROL, wrist_rot, maxVelocity=max_velocity)
        self.do_steps(steps)

    # ONLY UNTIL PI
    def rotate_wrist_ccw(self):
        curr_wrist_angle, _ = self.get_wrist_state()
        self.rotate_wrist(curr_wrist_angle + 0.25)

    # ONLY UNTIL -PI
    def rotate_wrist_cw(self):
        curr_wrist_angle, _ = self.get_wrist_state()
        self.rotate_wrist(curr_wrist_angle - 0.25)


    def open_gripper(self, steps=30):
        """ Open the gripper in steps simulation steps. """
        p.setJointMotorControl2(self.robot, self.LEFT_GRIPPER, p.POSITION_CONTROL, .02)
        p.setJointMotorControl2(self.robot, self.RIGHT_GRIPPER, p.POSITION_CONTROL, -.02)

        self.do_steps(steps)

    def close_gripper(self, steps=30):
        """ Close the gripper in steps simulation steps. """
        maxForce = 10
        p.setJointMotorControl2(self.robot, self.LEFT_GRIPPER, p.POSITION_CONTROL, -0.001, force=maxForce)
        p.setJointMotorControl2(self.robot, self.RIGHT_GRIPPER, p.POSITION_CONTROL, 0.001, force=maxForce)

        self.do_steps(steps)
    
    def move_joint_to_pos(self, joint, pos, steps=30, max_velocity=float("inf")):
        """ Move an arbitrary joint to the desired pos. """
        p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, pos)
        self.do_steps(steps)

    def set_joint_velocity(self, joint, velocity):
        """ Set an arbitrary joint to the desired velocity. """
        p.setJointMotorControl2(self.robot, joint, p.VELOCITY_CONTROL, targetVelocity=velocity)

    def get_wrist_state(self):
        """ Returns wrist rotation and how open gripper is.
        """
        curr_wrist_angle, _, _, _ = p.getJointState(self.robot, self.WRIST_JOINT)
        curr_open, _, _, _ = p.getJointState(self.robot, self.LEFT_GRIPPER)
        return curr_wrist_angle, curr_open
        
    def get_ee_global(self):
        """ Returns ee position and orientation in world coordinates. """
        _, _, _, _, ee_pos, ee_ori = p.getLinkState(self.robot, self.WRIST_JOINT)
        return ee_pos, ee_ori
    
    def get_ee_local(self):
        """ Returns ee position and orientation relative to the robot's base. """
        _, _, _, _, ee_pos, ee_ori = p.getLinkState(self.robot, self.WRIST_JOINT)
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        base_pos, base_ori = p.invertTransform(base_pos, base_ori)
        local_ee_pos, local_ee_ori = p.multiplyTransforms(base_pos, base_ori, ee_pos, ee_ori)
        local_ee_pos = [local_ee_pos[0], local_ee_pos[1], local_ee_pos[2] - self.GRIPPER_LENGTH_FROM_WRIST]
        return local_ee_pos, local_ee_ori
    
    def apply_continuous_action(self, action, max_velocity=float("inf")):
        """Args:
          action: 5-vector parameterizing XYZ offset, vertical angle offset
          (radians), and grasp (value between 0.001 (close) and 0.2 (open)."""
        curr_ee, curr_ori = self.get_ee_global()
        new_ee = np.array(curr_ee) + action[:3]
        jointStates = p.calculateInverseKinematics(self.robot, self.WRIST_JOINT, new_ee, curr_ori, maxNumIterations=150)[2:6]
        curr_wrist_angle, gripper_opening = self.get_wrist_state()
        new_wrist_angle = curr_wrist_angle + action[3]
        self.move_arm(jointStates, wrist_rot=new_wrist_angle, steps=70, max_velocity=max_velocity)

        p.setJointMotorControl2(self.robot, self.LEFT_GRIPPER, p.POSITION_CONTROL, -1*action[4])
        p.setJointMotorControl2(self.robot, self.RIGHT_GRIPPER, p.POSITION_CONTROL, action[4])
        # self.do_steps(30)
        return

    def gripper_center_pos(self):
        left_pos, _, _, _, _, _ = p.getLinkState(self.robot, self.LEFT_GRIPPER)
        right_pos, _, _, _, _, _ = p.getLinkState(self.robot, self.RIGHT_GRIPPER)

        mid_pos = 0.5 * (np.array(left_pos) + np.array(right_pos))
        return mid_pos

    def grab(self, grabbin_point): # should also provide rotation angle
        """
        Assumes robot is well oriented, i.e. the object is in front of the robot. Which means y-axis is already ok
        """

        # get gripper pos

        pos = self.gripper_center_pos()
        dx, dy, dz = grabbin_point - pos

        vx = 1
        vz = 1

        # FIRST rotate gripper

        action = [vx, 0, vz, 0, -0.03]
        self.apply_continuous_action(action)
        self.steps(200)

        action = [0, 0, 0, 0, 0.03]
        self.apply_continuous_action(action)
        self.steps(200)

        return


    def move_arm_up(self):
        _, _, _, _, ee_pos, ee_ori = p.getLinkState(self.robot, self.WRIST_JOINT)
        ee_pos = np.array(ee_pos)
        ee_pos[2] += 0.02
        self.move_ee_absolute(ee_pos, steps = 100)
        # x, y, theta = self.get_base_pos_and_yaw()

        # x0, y0 = ee_pos[:2]
        # dx = x0 - x
        # dy = y0 - y

        # look_x = math.cos(theta)
        # look_y = math.sin(theta)

        # # angle between observation vector and wrist (TODO check sign)
        # alfa = math.atan2(dy, dx) - math.atan2(look_y, look_x) 
        # alfa *= -1

        # # rotate (dx, dy) -theta rad so that its equivalent to the standard reference system
        # dx, dy = dx * math.cos(-theta) - dy * math.sin(-theta), dx * math.sin(-theta) + dy * math.cos(-theta)

        # # project to each axis
        # xp = dx * math.cos(alfa)
        # yp = 0 # should always be 0. Don't propagate compounding errors. It should theoretuically be: dy * math.sin(alfa)

        # pos = [xp, yp, ee_pos[2] - self.GRIPPER_LENGTH_FROM_WRIST + 0.02]
        # self.move_ee(pos)
        

    def move_arm_down(self):
        _, _, _, _, ee_pos, ee_ori = p.getLinkState(self.robot, self.WRIST_JOINT)
        ee_pos = np.array(ee_pos)
        ee_pos[2] -= 0.02
        self.move_ee_absolute(ee_pos, steps = 100)
        # x, y, theta = self.get_base_pos_and_yaw()

        # x0, y0 = ee_pos[:2]
        # dx = x0 - x
        # dy = y0 - y

        # look_x = math.cos(theta)
        # look_y = math.sin(theta)

        # # angle between observation vector and wrist (TODO check sign)
        # alfa = math.atan2(dy, dx) - math.atan2(look_y, look_x) 
        # alfa *= -1

        # # rotate (dx, dy) -theta rad so that its equivalent to the standard reference system
        # dx, dy = dx * math.cos(-theta) - dy * math.sin(-theta), dx * math.sin(-theta) + dy * math.cos(-theta)

        # # project to each axis
        # xp = dx * math.cos(alfa)
        # yp = 0 # should always be 0. Don't propagate compounding errors. It should theoretuically be: dy * math.sin(alfa)

        # pos = [xp, yp, ee_pos[2] - self.GRIPPER_LENGTH_FROM_WRIST - 0.02]
        # # pos = [xp, yp, ee_pos[2] - self.GRIPPER_LENGTH_FROM_WRIST]
        # self.move_ee(pos)
        

    def move_arm_forward(self):
        dx = 0.02
        _, _, _, _, ee_pos, ee_ori = p.getLinkState(self.robot, self.WRIST_JOINT)
        ee_pos = np.array(ee_pos)
        x, y, theta = self.get_base_pos_and_yaw()

        x0, y0 = ee_pos[:2]
        # dx = x0 - x
        # dy = y0 - y
        
        look = np.array([math.cos(theta), math.sin(theta), 0])
        look = dx * look / np.linalg.norm(look)
        
        self.move_ee_absolute(ee_pos + look, steps = 100)
        # # angle between observation vector and wrist (TODO check sign)
        # alfa = math.atan2(dy, dx) - math.atan2(look_y, look_x) 
        # alfa *= -1

        # # rotate (dx, dy) -theta rad so that its equivalent to the standard reference system
        # dx, dy = dx * math.cos(-theta) - dy * math.sin(-theta), dx * math.sin(-theta) + dy * math.cos(-theta)

        # # project to each axis
        # xp = dx * math.cos(alfa)
        # yp = 0 # should always be 0. Don't propagate compounding errors. It should theoretuically be: dy * math.sin(alfa)

        # pos = [xp + 0.02, yp, ee_pos[2]  - self.GRIPPER_LENGTH_FROM_WRIST]
        # self.move_ee(pos)
        
        
    def move_arm_backwards(self):
        dx = 0.02 
        _, _, _, _, ee_pos, ee_ori = p.getLinkState(self.robot, self.WRIST_JOINT)
        ee_pos = np.array(ee_pos)
        x, y, theta = self.get_base_pos_and_yaw()

        x0, y0 = ee_pos[:2]
        # dx = x0 - x
        # dy = y0 - y

        look = np.array([math.cos(theta), math.sin(theta), 0])
        look = dx * look / np.linalg.norm(look)

        print("BACK", ee_pos, look)
        
        self.move_ee_absolute(ee_pos - look, steps = 100)

        # # angle between observation vector and wrist (TODO check sign)
        # alfa = math.atan2(dy, dx) - math.atan2(look_y, look_x) 
        # alfa *= -1

        # # rotate (dx, dy) -theta rad so that its equivalent to the standard reference system
        # dx, dy = dx * math.cos(-theta) - dy * math.sin(-theta), dx * math.sin(-theta) + dy * math.cos(-theta)

        # # project to each axis
        # xp = dx * math.cos(alfa)
        # yp = 0 # should always be 0. Don't propagate compounding errors. It should theoretuically be: dy * math.sin(alfa)
        
        # pos = [xp - 0.02, yp, ee_pos[2]  - self.GRIPPER_LENGTH_FROM_WRIST]
        # self.move_ee(pos)
        
    # ----- END ARM METHODS -----

    def reset_joints(self):
        p.setJointMotorControl2(self.robot, self.LEFT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)
        p.setJointMotorControl2(self.robot, self.RIGHT_WHEEL, p.VELOCITY_CONTROL, targetVelocity=0, force=1e4)


    def action(self, action):
        """"
        All actions are expeted to be around the range [-1, 1]. They will then be corrected to the ideal range.

        action[0] - move forward. Ideal range: [-1, 1]
        action[1] - rotate anti-clockwise. Ideal range: [-2, 2]
        action[2] - move arm along x-axis. Ideal range [-0.1, 0.1]
        action[3] - move arm along a werid axis (like diagonally between x & y). Ideal range: [-0.5, 0.5]
        action[4] - move arm along z-axis. Ideal range [-0.1, 0.1]
        action[5] - rotate gripper hand. Ideal range [-0.5, 0.5]
        action[6] - close open gripper. Ideal range [-0.03, 0.03] (actually [0, 0.03] but it doesn't matter)
        """
        
        action = np.clip(np.array(action).astype(float), -1, 1)

        # 100 steps
        # action[0] *= 0.2
        # action[1] *= 0.5
        # action[2] *= 0.01
        # action[3] *= 0.05
        # action[4] *= 0.01
        # action[5] *= 0.05
        # action[6] *= 0.04


        action[0] *= 1
        action[1] *= 2
        action[2] *= 0.1
        action[3] *= 0.5
        action[4] *= 0.1
        action[5] *= 0.5
        action[6] *= 0.3

        self.differential_drive(action[:2])
        self.apply_continuous_action(action[2:])

        if (abs(action[1]) > 0.1):
            self.do_steps(200)
        elif (abs(action[0]) > 0.1):
            self.do_steps(200)

        self.do_steps(200)

        self.reset_base()

        self.reset_joints()


    # ----- MISC METHODS -----

    def step(self):
        """ Do a single simulation step. If in GUI mode, then this takes step_duration seconds. """
        if self.renders:
            time.sleep(self.step_duration)
        p.stepSimulation()
        self.total_sim_steps += 1
        
    def do_steps(self, num_steps):
        """ Do num_steps simulation steps. If in GUI mode, then this takes num_steps * step_duration seconds. """
        for _ in range(num_steps):
            self.step()

    def add_frame(self, frame):
        self.frames.append(frame)

    def clear_frames(self):
        self.frames = []

    def save_frames(self, folder, name):
        if len(self.frames) <= 1:
            return
        try:
            save_path = os.path.join(folder, f"{name}_{self.video_index}.webm")
            
            clip = mpy.ImageSequenceClip(self.frames, fps=self.video_fps)
            clip.write_videofile(save_path, fps=self.video_fps, audio=False)
        except Exception as e:
            print("Error", e, "occurred while trying to save frames")

        self.video_index += 1
        self.clear_frames()

    def render_camera(self, use_aux=False, save_frame=False, **kwargs):
        """ Renders the scene
        Args:
            use_aux: determines whether this renders using the main camera or the auxilary camera.
        Returns:
            (height, width, channel) uint8 array
        """
        if use_aux:
            camera = self.aux_camera
            camera_look_pos = self.params["aux_camera_look_pos"]
            camera_link = self.AUX_CAMERA_LINK
        else:
            camera = self.camera
            camera_look_pos = self.params["camera_look_pos"]
            camera_link = self.CAMERA_LINK

        camera_pos, camera_ori, _, _, _, _ = p.getLinkState(self.robot, camera_link)
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        look_pos, look_ori = p.multiplyTransforms(base_pos, base_ori, camera_look_pos, self.default_ori)
        camera.update(camera_pos, look_pos)
            
        img = camera.get_image(width=self.w, height=self.h, **kwargs)
        return img