#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, os
from pcms.pytorch_models import *
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
import math
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
from rospkg import RosPack
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Imu
from typing import Tuple, List
from RobotChassis import RobotChassis
import datetime
from std_srvs.srv import Empty
from gtts import gTTS
from playsound import playsound
import requests
import speech_recognition as sr
import json
import os


# gemini2
def callback_image2(msg):
    global _frame2
    _frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth2(msg):
    global _depth2
    _depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def get_real_xyz(dp, x, y, num):
    a1 = 49.5
    b1 = 60.0
    if num == 2:
        a1 = 55.0
        b1 = 86.0
    a = a1 * np.pi / 180
    b = b1 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y - k][j]
                    if d > 0:
                        break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x + k]
                    if d > 0:
                        break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y + k][j]
                    if d > 0:
                        break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x - k]
                    if d > 0:
                        break
            if d > 0:
                break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d


def callback_voice(msg):
    global s
    s = msg.text


def speak(g):
    os.system(f'espeak "{g}"')
    # rospy.loginfo(g)
    print(g)


class FollowMe(object):
    def __init__(self) -> None:
        self.pre_x, self.pre_z = 0.0, 0.0

    def get_pose_target(self, pose, num):
        p = []
        for i in [num]:
            if pose[i][2] > 0:
                p.append(pose[i])

        if len(p) == 0:
            return -1, -1, -1
        return int(p[0][0]), int(p[0][1]), 1

    def get_real_xyz(self, depth, x: int, y: int) -> Tuple[float, float, float]:
        if x < 0 or y < 0:
            return 0, 0, 0
        a1 = 55.0
        b1 = 86.0
        a = a1 * np.pi / 180
        b = b1 * np.pi / 180

        d = depth[y][x]
        h, w = depth.shape[:2]
        if d == 0:
            for k in range(1, 15, 1):
                if d == 0 and y - k >= 0:
                    for j in range(x - k, x + k, 1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y - k][j]
                        if d > 0:
                            break
                if d == 0 and x + k < w:
                    for i in range(y - k, y + k, 1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x + k]
                        if d > 0:
                            break
                if d == 0 and y + k < h:
                    for j in range(x + k, x - k, -1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y + k][j]
                        if d > 0:
                            break
                if d == 0 and x - k >= 0:
                    for i in range(y + k, y - k, -1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x - k]
                        if d > 0:
                            break
                if d > 0:
                    break
        x = x - w // 2
        y = y - h // 2
        real_y = y * 2 * d * np.tan(a / 2) / h
        real_x = x * 2 * d * np.tan(b / 2) / w
        return real_x, real_y, d

    def calc_linear_x(self, cd: float, td: float) -> float:
        if cd <= 0:
            return 0
        e = cd - td
        p = 0.0005
        x = p * e
        if x > 0:
            x = min(x, 0.15)
        if x < 0:
            x = max(x, -0.15)
        return x

    def calc_angular_z(self, cx: float, tx: float) -> float:
        if cx < 0:
            return 0
        e = tx - cx
        p = 0.0025
        z = p * e
        if z > 0:
            z = min(z, 0.4)
        if z < 0:
            z = max(z, -0.4)
        return z

    def calc_cmd_vel(self, image, depth, cx, cy) -> Tuple[float, float]:
        image = image.copy()
        depth = depth.copy()

        frame = image
        if cx == 2000:
            cur_x, cur_z = 0, 0
            return cur_x, cur_z, frame, "no"

        print(cx, cy)
        _, _, d = self.get_real_xyz(depth, cx, cy)

        cur_x = self.calc_linear_x(d, 850)
        cur_z = self.calc_angular_z(cx, 350)

        dx = cur_x - self.pre_x
        if dx > 0:
            dx = min(dx, 0.15)
        if dx < 0:
            dx = max(dx, -0.15)

        dz = cur_z - self.pre_z
        if dz > 0:
            dz = min(dz, 0.4)
        if dz < 0:
            dz = max(dz, -0.4)

        cur_x = self.pre_x + dx
        cur_z = self.pre_z + dz

        self.pre_x = cur_x
        self.pre_z = cur_z

        return cur_x, cur_z, frame, "yes"


def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def find_walk_in_front(image, depth):
    # finding
    pass


def answer_question():
    pass


def post_message_request(step, s1,question):
    api_url = "http://192.168.50.147:8888/Fambot"
    my_todo = {"Question1": "None",
               "Question2": "None",
               "Question3": "None",
               "Steps": step,
               "Voice": s1,
               "Questionasking":question,
               "answer":"None"}
    response = requests.post(api_url, json=my_todo, timeout=2.5)
    result = response.json()
    return result


def callback_voice(msg):
    global s
    s = msg.text



def recognize_speech(duration):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = recognizer.record(source, duration=duration)
        try:
            return recognizer.recognize_google(audio_data).lower()
        except sr.UnknownValueError:
            return None


def confirm_recording(original_text):
    while True:  # Keep asking until we get a "yes"
        speak(f"Did you say: {original_text}?")
        speak("Please answer hello fambot 'yes yes yes' or 'no no no' after the sound")
        time.sleep(1)
        playsound("nigga2.mp3")
        response = recognize_speech(5)

        if response is None:
            speak("Sorry, I didn't catch that. Let's try again.")
            continue

        print(f"You responded: {response}")

        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            speak("I didn't understand your answer. Please say 'yes' or 'no'.")


def get_user_input():
    while True:
        # Voice introduction
        speak("Hi, I am fambot. How can I help you? Speak you command after the")
        playsound("nigga2.mp3")  # Changed sound file name
        speak("sound")

        # Countdown
        time.sleep(0.5)
        playsound("/home/pcms/catkin_ws/nigga2.mp3")

        user_text = recognize_speech(10)

        if not user_text:
            error_msg = "Sorry, I couldn't understand. Please try again."
            print(error_msg)
            speak(error_msg)
            continue

        print(f"You said: {user_text}")
        #speak(f"You said: {user_text}")
        user_text=user_text.replace("facebook","fambot")
        if confirm_recording(user_text):
            return user_text
        else:
            speak("Let's try recording again.")

    return None
def ask_question(question):
    while True:
        # Voice introduction
        speak("Dear Guest")
        speak(question)
        speak("please speak the entire sentence, for example my name is Fambot")
        speak("speak after the")
        playsound("nigga2.mp3")  # Changed sound file name
        speak("sound")

        # Countdown
        time.sleep(1)
        playsound("/home/pcms/catkin_ws/nigga2.mp3")

        user_text = recognize_speech(5)

        if not user_text:
            error_msg = "Sorry, I couldn't understand. Please try again."
            print(error_msg)
            speak(error_msg)
            continue

        print(f"You said: {user_text}")
        #speak(f"You said: {user_text}")
        user_text=user_text.replace("facebook","fambot")
        if confirm_recording(user_text):
            return user_text
        else:
            speak("Let's try recording again.")

    return None
def find_person(frame,pose):
    detections = dnn_yolo.forward(frame)[0]["det"]
    for i, detection in enumerate(detections):
        # print(detection)
        x1, y1, x2, y2, score, class_id = map(int, detection)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        score = detection[4]
        if class_id == 0 and score >= 0.55:
            # print(cx,cy,"position bottle")
            cx = min(cx, 640)
            cy = min(cy, 480 - 1)
            k2, kk1, kkkz = get_real_xyz(frame, cx, cy)
            # print(float(score), class_id)
            hhh = str(class_id) + " " + str(k2) + " " + str(kk1) + " " + str(kkkz)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            #capture
            #gemini response


    return frame
if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    # open things
    location1 = {"wall long": (0, 0, 0), "wall short": (0, 0, 0),
                 "long table a": (0, 0, 0), "long table b": (0, 0, 0),
                 "tall table": (0, 0, 0), "shelf": (0, 0, 0),
                 "chair a": (0, 0, 0), "chair b": (0, 0, 0),
                 "tray a": (0, 0, 0), "tray b": (0, 0, 0),
                 "container": (0, 0, 0),"containers": (0, 0, 0), "pen holder": (0, 0, 0),
                 "trash bin a": (0, 0, 0), "trash bin b": (0, 0, 0),
                 "storage box a": (0, 0, 0), "storage box b": (0, 0, 0)}

    location2 = {"starting point": (0, 0, 0), "exit": (0, 0, 0), "host": (0, 0, 0),
                 "dining room": (0, 0, 0), "living room": (0, 0, 0),
                 "hallway": (0, 0, 0), "dining_room": (0, 0, 0), "living_room": (0, 0, 0)}

    #chassis = RobotChassis()
    #clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    net_pose = HumanPoseEstimation(device_name="GPU")
    _fw = FollowMe()
    print("gemini2 rgb")
    _frame2 = None
    _sub_down_cam_image = rospy.Subscriber("/cam2/color/image_raw", Image, callback_image2)
    print("gemini2 depth")
    _depth2 = None
    _sub_down_cam_depth = rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth2)
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # step_action
    # add action for all code
    # Step 0 first send
    # Step 1 first get
    # Step 9 send image response text
    # step 10 get the image response
    s = ""
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    for i in range(3):
        user_input = get_user_input()
        if user_input:
            print(f"Final recognized input: {user_input},    {i}")
        else:
            speak("I couldn't understand your input after several attempts. Please try again later.")
        # post question
        gg = post_message_request("first", user_input,"")  # step
        print("post", gg)
        # get gemini answer
        nigga = 1
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
        # say how the robot understand
        speak(Q3[0])
        # divide
        command_type = Q1[0]
        command_type = command_type.lower()
        step_action = 0
        #continue
        while not rospy.is_shutdown():
            # voice check
            # break
            code_image = _frame2.copy()
            code_depth = _depth2.copy()
            rospy.Rate(10).sleep()
            cv2.imshow("frame", code_image)
            key = cv2.waitKey(1)
            if key in [ord('q'), 27]:
                break
            #DIVIDE

            '''
            #Manipulation1
            if "manipulation1" in command_type or ("mani" in command_type and "1" in command_type):
                pass
            #Manipulation2
            elif "manipulation2" in command_type or ("mani" in command_type and "2" in command_type):
                pass
            #Vision'''
            if ("vision (enumeration)1" in command_type or ("vision" in command_type and "1" in command_type and "enume" in command_type)) or ("vision (enumeration)2" in command_type or ("vision" in command_type and "2" in command_type and "enume" in command_type)):
                #Move
                if  step_action==0:
                    liyt=Q2
                    if ("2" in command_type):
                        #num1, num2, num3 = location2[liyt["$ROOM"]]
                        speak("going to"+liyt["$ROOM"])
                    else:
                        #num1, num2, num3 = location1[liyt["$PLACE"]]
                        speak("going to"+liyt["$PLACE"])
                    '''
                    chassis.move_to(num1,num2,num3)
                    while not rospy.is_shutdown():
                        # 4. Get the chassis status.
                        code = chassis.status_code
                        text = chassis.status_text
                        if code == 3:
                            break
                    time.sleep(1)
                    clear_costmaps'''
                    step_action = 1
                if  step_action==1:
                    time.sleep(2)
                    print("take picture")
                    #save frame
                    output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                    cv2.imwrite(output_dir + "test.jpg", code_image)
                    #ask gemini
                    url = "http://192.168.50.147:8888/upload_image"
                    file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/test.jpg"
                    with open(file_path, 'rb') as f:
                        files = {'image': (file_path.split('/')[-1], f)}
                        response = requests.post(url, files=files)
                        # remember to add the text question on the computer code
                    print("Upload Status Code:", response.status_code)
                    upload_result = response.json()
                    print("sent image")
                    gg = post_message_request("file", user_input,"")
                    print(gg)
                    #get answer from gemini
                    while True:
                        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == 10:
                            break
                        pass
                        time.sleep(2)
                    step_action = 2
                    print(dictt["Voice"])
                if  step_action==2:
                    #back
                    speak("walking to"+" the starting point")
            else:
                speak("next")
            
                    
