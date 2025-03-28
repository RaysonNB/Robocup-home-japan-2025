#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8, HumanPoseEstimation
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
import math
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from gtts import gTTS
from playsound import playsound
import requests
import json
import os

def callback_voice(msg):
    global s
    s = msg.text


def say(g):
    os.system(f'espeak "{g}"')
    rospy.loginfo(g)


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    #open things
    location1 = {"wall long":(0,0,0),"wall short":(0,0,0),
                 "long table a":(0,0,0),"long table b":(0,0,0),
                 "tall table":(0,0,0),"shelf":(0,0,0),
                 "chair a":(0,0,0),"chair b":(0,0,0),
                 "tray a":(0,0,0),"tray b":(0,0,0),
                 "container":(0,0,0),"pen holder":(0,0,0),
                 "trash bin a":(0,0,0),"trash bin b":(0,0,0),
                 "storage box a":(0,0,0),"storage box b":(0,0,0)}

    location2 = {"starting point":(0,0,0),"exit":(0,0,0),"host":(0,0,0),
                 "dinning room":(0,0,0),"living room":(0,0,0),
                 "hallway":(0,0,0), "dinning_room":(0,0,0),"living_room":(0,0,0)}
    for i in range(3):
        s1 = input("The sentence is: ")
        #post question
        api_url = "http://192.168.50.147:8888/Fambot"
        my_todo = {"Question1": "None",
                   "Question2": "None",
                   "Question3": "None",
                   "Steps": 0,
                   "Voice": s1}
        response = requests.post(api_url, json=my_todo, timeout=2.5)
        result = response.json()
        print("post", result)
        #get gemini answer
        while True:
            r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
            response_data = r.text
            dictt = json.loads(response_data)
            if dictt["Steps"] == 1:
                break
            pass
            time.sleep(2)
        Q1 = dictt["Question1"]
        Q2 = dictt["Question2"]
        Q3 = dictt["Question3"]
        print(Q1)
        print(Q2)
        print(Q3)
        #say how the robot understand
        say(Q3[0])
        #divide
        command_type=Q1[0]
        command_type=command_type.lower()

        #Manipulation1
        if "Manipulation1" in command_type or ("Mani" in command_type and "1" in command_type):
            pass
        #Manipulation2
        elif "Manipulation2" in command_type or ("Mani" in command_type and "2" in command_type):
            pass
        #Vision (Enumeration)1
        elif "Vision (Enumeration)1" in command_type or ("Vision" in command_type and "1" in command_type):
            #Move
            liyt=Q2.json
            num1,num2,num3=location1[liyt["$PLACE"]]
            chassis.move_to(num1,num2,num3)
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            clear_costmaps
            #save frame
            output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
            cv2.imwrite(output_dir + "test.jpg", frame)
            #ask gemini
            url = "http://192.168.50.147:8888/upload_image"
            file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/test.jpg"
            with open(file_path, 'rb') as f:
                files = {'image': (file_path.split('/')[-1], f)}
                response = requests.post(url, files=files)

            print("Upload Status Code:", response.status_code)
            upload_result = response.json()
            print("sent image")
            #get answer from gemini
            while True:
                r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
                response_data = r.text
                dictt = json.loads(response_data)
                if dictt["Steps"] == 7:
                    break
                pass
                time.sleep(2)
            Q1 = dictt["Question1"]
            #back
            num1, num2, num3 = location2["starting point"]
            chassis.move_to(num1, num2, num3)
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            time.sleep(1)
            clear_costmaps
        #Vision (Enumeration)2
        elif "Vision (Enumeration)2" in command_type or ("Vision" in command_type and "1" in command_type):
            pass
        #Vision (Description)1
        elif "Vision (Description)1" in command_type or ("Vision" in command_type and "1" in command_type):
            pass
        #Vision (Description)2
        elif "Vision (Description)2" in command_type or ("Mani" in command_type and "1" in command_type):
            pass
        #Navigation1
        elif "Navigation1" in command_type or ("Mani" in command_type and "1" in command_type):
            pass
        #Navigation2
        elif "Navigation2" in command_type or ("Mani" in command_type and "1" in command_type):
            pass
        #Speech1
        elif "Speech1" in command_type or ("Mani" in command_type and "1" in command_type):
            pass
        #Speech2
        elif "Speech2" in command_type or ("Mani" in command_type and "1" in command_type):
            pass
