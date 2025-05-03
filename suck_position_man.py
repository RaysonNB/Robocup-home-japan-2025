#!/usr/bin/env python3
from dynamixel_control import DynamixelController
from robotic_arm_control import RoboticController
id_list = [11, 13, 15, 14, 12, 1, 2]

Dy = DynamixelController()
Ro = RoboticController()
Ro.open_robotic_arm("/dev/arm", id_list, Dy)
Ro.go_to_real_xyz_alpha(id_list, [0, 100, 200], -15, 0, 90, 0, Dy)
Ro.go_to_real_xyz_alpha(id_list, [0, 100, 200], -15, 0, -8, 0, Dy)
