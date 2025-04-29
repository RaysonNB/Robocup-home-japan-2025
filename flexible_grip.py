import time

from dynamixel_control import DynamixelController
from robotic_arm_control import RoboticController

id_list = [11, 13, 15, 14, 12, 1, 2]

Dy = DynamixelController()
Ro = RoboticController()
Ro.open_robotic_arm("COM4", id_list, Dy)
print("Arm init")

grip_id = id_list[-1]
target_angle = angle = 90
dt = 0.1

print("Opening")
Dy.‎goal_absolute_direction(grip_id, target_angle)
time.sleep(3)

print("Start")
while target_angle > 5:

    dangle = abs(angle - Dy.‎present_position(grip_id))
    time.sleep(dt)
    angle = Dy.‎present_position(grip_id)
    angle_speed = dangle / dt
    print(angle_speed)

    target_angle = angle - 5
    Dy.‎goal_absolute_direction(grip_id, target_angle)

    if angle_speed < 8.0:
        print(f"Stop at {Dy.‎present_position(grip_id)}")
        break

Dy.‎goal_absolute_direction(grip_id, 90)
