from dynamixel_control import DynamixelController
from robotic_arm_control import RoboticController

arm_id_list = [11, 13, 15, 14, 12, 1, 2]

Dy = DynamixelController()
Ro = RoboticController()
Ro.open_robotic_arm("COM4", id_list, Dy)

grip_id = arm_id_list[-1]
target_angle = 30
dt = 0.1
Dy.‎goal_absolute_direction(grip_id, angle)
while True:
    dangle = abs(angle - Dy.‎present_position(grip_id))
    time.sleep(dt)
    angle = Dy.‎present_position(grip_id)
    angle_speed = dangle / dt
    print(angle_speed)

    if abs(angle - target_angle) < 3.5:
        print("Target Reached!")
        

Dy.‎goal_absolute_direction(grip_id, 90)
