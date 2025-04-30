import time

from dynamixel_control import DynamixelController
from robotic_arm_control import RoboticController

id_list = [11, 13, 15, 14, 12, 1, 2]

Dy = DynamixelController()
Ro = RoboticController()
Ro.open_robotic_arm("/dev/arm", id_list, Dy)

print("Arm init")

grip_id = id_list[-1]
target_angle = 90
last_angle = Dy.present_position(grip_id)
dt = 0.2
final_angle = 0
i = 0

print("Opening")
Ro.go_to_real_xyz_alpha(id_list, [0, 300, 150], 0, 0, target_angle, 0, Dy)
Dy.profile_velocity(grip_id, 20)

print("Start")
while target_angle > 0:

    i += 1
    angle = Dy.present_position(grip_id)
    dangle = abs( last_angle - Dy.present_position(grip_id) )
    time.sleep(dt)
    last_angle = angle

    angle = Dy.present_position(grip_id)
    angle_speed = dangle / dt
    print(angle, last_angle, angle_speed, target_angle, i)

    target_angle = angle - 10
    Dy.goal_absolute_direction(grip_id, max(target_angle, -5))

    if angle_speed <= 20.0 and i > 3:
        print(f"Stop at {Dy.present_position(grip_id)}")
        final_angle = Dy.present_position(grip_id) + 4
        Dy.goal_absolute_direction(grip_id, final_angle)
        break

time.sleep(1)
Ro.go_to_real_xyz_alpha(id_list, [0, 250, 150], -25, 0, final_angle, 0, Dy)

