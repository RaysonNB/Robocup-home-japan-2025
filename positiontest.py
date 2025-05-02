#!/usr/bin/env python3
from RobotChassis import RobotChassis
from std_srvs.srv import Empty
import rospy
import os
import time


def speak(g):
    os.system(f'espeak "{g}"')
    # rospy.loginfo(g)
    print(g)


def check_item(name):
    corrected = "starting point"
    cnt = 0
    if name in locations:
        corrected = name
    else:
        corrected = corrected.replace("_", " ")
    return corrected


clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)


def walk_to(name):
    if "none" not in name or "unknow" in name:
        name = name.lower()
        real_name = check_item(name)
        speak("going to " + str(name))
        if real_name in locations:
            num1, num2, num3 = locations[real_name]
            chassis.move_to(num1, num2, num3)
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
                if code == 4:
                    break
            speak("arrived")
            time.sleep(1)
            clear_costmaps


locations = {
    # Furniture and objects
    "counter": [-3.324, 1.2, 1.57],
    "left tray": [-3.874, 1.49, 1.57],
    "right tray": [-3.478, 1.49, 1.57],
    "pen holder": [-2.031, 1.49, 1.57],
    "container": [-2.814, 1.49, 1.57],
    "left kachaka shelf": [-2.134, 1.145, 1.57],
    "right kachaka shelf": [-1.662, 1.114, 1.57],
    "low table": [-2.61,1.166, -1.57],
    "left chair": [-1.623, -1.582, -1.57],  #
    "right chair": [-1.940, -1.642, -1.57],  #
    "trash bin": [-4.442, 1.206, 1.57],  #
    "tall table": [-1.432,1.192, -1.57],
    "left kachaka station": [-3.124, -2.026, -2.617],
    "right kachaka station": [-2.976, -1.794, 3.14],
    "shelf": [-2.884, -1.256, -1.57],
    # bed
    "bed": [-0.410,-0.640,-0.663],
    # dining room
    "dining table": [-0.491, 1.404, 0],
    "couch": [1.813, 0.339, 2.180],

    # Locations and special points
    "exit": [1.596, 1.729, 0],
    "final": [2.888,-1.048, 0],
    "entrance": [1.677, -1.070, 0],
    "instruction point": [-3.093,-1.571,-1.638],
    "dining room": [-0.921, 1.349, 0],
    "living room": [-2.927, 1.279, 0],
    "bedroom": [-0.014, -0.719, -0.185],
    "study room": [-2.666, -1.412, -0.494]
}
# front 0 back 3.14 left 90 1.5 right 90 -1.5
cout_location = {
    "living room": [-0.068, 1.220, 3.14],
    "bedroom": [-1.951, -1.584, 0],
    "dining room": [-2.348, 1.273, 0],
    "study room": [-0.137, -1.579, 3.14]
}

dining_room_dif = {
    "din1": [-0.934, 0.314, 1.568],
    "din2": [1.916, 2.449, -1.510]
}


# name
# qestion list
# answer
def walk_to1(name):
    if "none" not in name or "unknow" in name:

        name = name.lower()
        real_name = check_item(name)
        if real_name in cout_location:
            speak("going to " + str(name))
            num1, num2, num3 = cout_location[real_name]
            chassis.move_to(num1, num2, num3)
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
                if code == 4:
                    break
            speak("arrived")
            time.sleep(1)
            clear_costmaps


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    # open things
    chassis = RobotChassis()

    for x in locations.keys():
        walk_to(x)
    for x in cout_location.keys():
        speak("count position")
        walk_to1(x)
    speak("going to dining room 1")
    num1, num2, num3 = dining_room_dif["din1"]
    chassis.move_to(num1, num2, num3)
    while not rospy.is_shutdown():
        # 4. Get the chassis status.
        code = chassis.status_code
        text = chassis.status_text
        if code == 3:
            break
        if code == 4:
            break
    time.sleep(1)
    clear_costmaps
    speak("going to dining room 2")
    num1, num2, num3 = dining_room_dif["din2"]
    chassis.move_to(num1, num2, num3)
    while not rospy.is_shutdown():
        # 4. Get the chassis status.
        code = chassis.status_code
        text = chassis.status_text
        if code == 3:
            break
        if code == 4:
            break
    time.sleep(1)
    clear_costmaps
