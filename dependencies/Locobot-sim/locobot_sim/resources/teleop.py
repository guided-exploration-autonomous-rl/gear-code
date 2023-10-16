import logging
import os
import platform
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import pybullet as p

ARROWS = {
    0: "up_arrow",
    1: "down_arrow",
    2: "left_arrow",
    3: "right_arrow",
    65295: "left_arrow",
    65296: "right_arrow",
    65297: "up_arrow",
    65298: "down_arrow",
}

class KeyboardController:
    """
    Simple class for controlling iGibson robots using keyboard commands
    """

    def __init__(self):
        """
        :param robot: BaseRobot, robot to control
        """
        # Store relevant info from robot

        self.last_grip = 0
        self.action_dim = 7
        self.controller_info = OrderedDict()
        
        # Other persistent variables we need to keep track of
        self.joint_control_idx = None  # Indices of joints being directly controlled via joint control
        self.current_joint = -1  # Active joint being controlled for joint control
        self.gripper_direction = 1.0  # Flips between -1 and 1
        self.persistent_gripper_action = None  # Whether gripper actions should persist between commands,
        # i.e.: if using binary gripper control and when no keypress is active, the gripper action should still the last executed gripper action
        self.last_keypress = None  # Last detected keypress
        self.keypress_mapping = None
        self.populate_keypress_mapping()
        self.time_last_keyboard_input = time.time()

    def populate_keypress_mapping(self):
        """
        Populates the mapping @self.keypress_mapping, which maps keypresses to action info:

            keypress:
                idx: <int>
                val: <float>
        """
        self.keypress_mapping = {}
        

        self.keypress_mapping["h"] = {"idx": 0, "val": 1}
        self.keypress_mapping["n"] = {"idx": 0, "val": -1}
        self.keypress_mapping["b"] = {"idx": 1, "val": 1}
        self.keypress_mapping["m"] = {"idx": 1, "val": -1}
        self.keypress_mapping["up_arrow"] = {"idx": 2, "val": -1}
        self.keypress_mapping["down_arrow"] = {"idx": 2, "val": 1}
        self.keypress_mapping["right_arrow"] = {"idx": 4, "val": -1}
        self.keypress_mapping["left_arrow"] = {"idx": 4, "val": 1}
        self.keypress_mapping["x"] = {"idx": 3, "val": 1}
        self.keypress_mapping["z"] = {"idx": 3, "val": -1}
        self.keypress_mapping["e"] = {"idx": 5, "val": 1}
        self.keypress_mapping["r"] = {"idx": 5, "val": -1}
        self.keypress_mapping[","] = {"idx": 6, "val": 1}
        self.keypress_mapping["."] = {"idx": 6, "val": -1}

    def get_random_action(self):
        """
        :return Array: Generated random action vector (normalized)
        """
        return np.random.uniform(-1, 1, self.action_dim)

    def get_teleop_action(self):
        """
        :return Array: Generated action vector based on received user inputs from the keyboard
        """

        action = np.zeros(self.action_dim)
        action[-1] = self.last_grip
        keypress = self.get_keyboard_input()

        if keypress is not None:
            # If the keypress is a number, the user is trying to select a specific joint to control
            if keypress.isnumeric():
                if int(keypress) in self.joint_control_idx:
                    self.current_joint = int(keypress)

            elif keypress in self.keypress_mapping:
                action_info = self.keypress_mapping[keypress]
                idx, val = action_info["idx"], action_info["val"]

                # Non-null gripper
                if val is not None:
                    # If the keypress is a spacebar, this is a gripper action
                    if keypress == " ":
                        # We toggle the gripper direction if the last keypress is DIFFERENT from this keypress AND
                        # we're past the gripper time threshold, to avoid high frequency toggling
                        # i.e.: holding down the spacebar shouldn't result in rapid toggling of the gripper
                        if keypress != self.last_keypress:
                            self.gripper_direction *= -1.0

                        # Modify the gripper value
                        val *= self.gripper_direction
                        if self.persistent_gripper_action is not None:
                            self.persistent_gripper_action = val

                    # If there is no index, the user is controlling a joint with "[" and "]". Set the idx to self.current_joint
                    if idx is None and self.current_joint != -1:
                        idx = self.current_joint

                    if idx is not None:
                        action[idx] = val

        self.last_grip = action[-1]

        sys.stdout.write("\033[K")
        print("Pressed {}. Action: {}".format(keypress, action))
        sys.stdout.write("\033[F")

        # Update last keypress
        self.last_keypress = keypress

        # Possibly set the persistent gripper action
        if self.persistent_gripper_action is not None and self.keypress_mapping[" "]["val"] is not None:
            action[self.keypress_mapping[" "]["idx"]] = self.persistent_gripper_action

        # Return action
        return action

    def get_keyboard_input(self):
        """
        Checks for newly received user inputs and returns the first received input, if any
        :return None or str: User input in string form. Note that only the characters mentioned in
        @self.print_keyboard_teleop_info are explicitly supported
        """
        # Getting current time
        current_time = time.time()
        
        kbe = p.getKeyboardEvents()
        # Record the first keypress if any was detected
        keypress = -1 if len(kbe.keys()) == 0 else list(kbe.keys())[0]
        
        # Updating the time of the last check
        self.time_last_keyboard_input = current_time

        if keypress in ARROWS:
            # Handle special case of arrow keys, which are mapped differently between pybullet and cv2
            keypress = ARROWS[keypress]
        else:
            # Handle general case where a key was actually pressed (value > -1)
            keypress = chr(keypress) if keypress > -1 else None

        return keypress

    @staticmethod
    def print_keyboard_teleop_info():
        """
        Prints out relevant information for teleop controlling a robot
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print()
        print("*" * 30)
        print("Controlling the Robot Using the Keyboard")
        print("*" * 30)
        print()
        print("Joint Control")
        print_command("0-9", "specify the joint to control")
        print_command("[, ]", "move the joint backwards, forwards, respectively")
        print()
        print("Differential Drive Control")
        print_command("i, k", "turn left, right")
        print_command("l, j", "move forward, backwards")
        print()
        print("Inverse Kinematics Control")
        print_command("\u2190, \u2192", "translate arm eef along x-axis")
        print_command("\u2191, \u2193", "translate arm eef along y-axis")
        print_command("p, ;", "translate arm eef along z-axis")
        print_command("n, b", "rotate arm eef about x-axis")
        print_command("o, u", "rotate arm eef about y-axis")
        print_command("v, c", "rotate arm eef about z-axis")
        print()
        print("Boolean Gripper Control")
        print_command("space", "toggle gripper (open/close)")
        print()
        print("*" * 30)
        print()
