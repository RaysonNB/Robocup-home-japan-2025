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


def post_message_request(step, s1, question):
    api_url = "http://192.168.50.147:8888/Fambot"
    my_todo = {"Question1": "None",
               "Question2": "None",
               "Question3": "None",
               "Steps": step,
               "Voice": s1,
               "Questionasking": question,
               "answer": "None"}
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
        # speak(f"You said: {user_text}")
        user_text = user_text.replace("facebook", "fambot")
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
        # speak(f"You said: {user_text}")
        user_text = user_text.replace("facebook", "fambot")
        if confirm_recording(user_text):
            return user_text
        else:
            speak("Let's try recording again.")

    return None


def find_person(frame, pose):
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
            # capture
            # gemini response

    return frame
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
            speak("I didn't understand your answer.")


locations = {
    # Furniture and objects
    "wall long": [0, 0, 0],
    "wall short": [0, 0, 0],
    "long table a": [0, 0, 0],
    "long table b": [0, 0, 0],
    "tall table": [0, 0, 0],
    "shelf": [0, 0, 0],
    "chair a": [0, 0, 0],
    "chair b": [0, 0, 0],
    "tray a": [0, 0, 0],
    "tray b": [0, 0, 0],
    "container": [0, 0, 0],
    "pen holder": [0, 0, 0],
    "trash bin a": [0, 0, 0],
    "trash bin b": [0, 0, 0],
    "storage box a": [0, 0, 0],
    "storage box b": [0, 0, 0],

    # Locations and special points
    "starting point": [0, 0, 0],
    "exit": [0, 0, 0],
    "host": [0, 0, 0],
    "dining room": [0, 0, 0],
    "living room": [0, 0, 0],
    "hallway": [0, 0, 0]
}
def check_item(name):
    corrected="starting point"
    cnt=0
    if name in locations:
        corrected=name
    else:
        corrected=corrected.replace("_"," ")
    return corrected



chassis = RobotChassis()
clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
def walk_to(name):
    name=name.lower()
    real_name = check_item(name)
    num1, num2, num3 = locations[real_name]
    chassis.move_to(num1, num2, num3)
    while not rospy.is_shutdown():
        # 4. Get the chassis status.
        code = chassis.status_code
        text = chassis.status_text
        if code == 3:
            break
    time.sleep(1)
    clear_costmaps
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
            z = min(z, 0.2)
        if z < 0:
            z = max(z, -0.2)
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

        cur_x = self.calc_linear_x(d, 800)
        cur_z = self.calc_angular_z(cx, 320)

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

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    # open things


    # chassis = RobotChassis()
    # clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
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
    follow_cnt=0
    action=0
    _fw=FollowMe()
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)

    # step_action
    # add action for all code
    # Step 0 first send
    # Step 1 first get
    # Step 9 send image response text
    # step 10 get the image response
    s = ""
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    while True:
        if "start" in s:
            speak("going to" + "starting point")
            walk_to(locations["starting point"])
            break
    step="none"
    confirm_command = 0
    for i in range(3):
        qr_code_detector = cv2.QRCodeDetector()
        data=0
        speak("please scan your qr code in front of my camera")
        while True:
            print("step1")
            if _frame2 is None: continue
            code_image = _frame2.copy()
            data, bbox, _ = qr_code_detector.detectAndDecode(code_image)

            if data:
                print("QR Code detected:", data)
                break

            cv2.imshow("QR Code Scanner", code_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        #confirm
        if confirm_command==0:
            speak("Is your command "+str(data))
            speak("robot yes or robot no after the sound")
            confirm_command=1
        while True:
            if "yes" in s:
                speak("ok, I will go now")
                break   
        user_input = data
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
        # continue
        liyt = Q2
        print("yolov8")
        Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
        dnn_yolo1 = Yolov8("yolov8n", device_name="GPU")
        pre_s=""
        name_cnt="none"
        ageList = ['1', '5', '10', '17', '27', '41', '50', '67']
        # Initialize video capture
        #video = cv2.VideoCapture(args.video if args.video else 0)
        padding = 20
        while not rospy.is_shutdown():
            # voice check
            # break
            confirm_command=0
            if s != "" and s != pre_s:
                print(s)
                pre_s = s
            code_image = _frame2.copy()
            code_depth = _depth2.copy()
            rospy.Rate(10).sleep()
            cv2.imshow("frame", code_image)
            key = cv2.waitKey(1)
            if key in [ord('q'), 27]:
                break
            # DIVIDE
            #Manipulation1 just walk
            if "manipulation1" in command_type or ("mani" in command_type and "1" in command_type):
                if step_action==0:
                    name_position="$ROOM1"
                    if "$ROOM1" in liyt:
                        name_position="ROOM1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action=1
                if step_action==1:
                    name_position = "$PLACE1"
                    if "$PLACE1" in liyt:
                        name_position = "PLACE1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action=2
                if step_action==2:
                    name_position = "$PLACE2"
                    if "$PLACE2" in liyt:
                        name_position = "PLACE2"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action=3
                if step_action==3:
                    speak("going to" + liyt["starting point"])
                    walk_to(locations["starting point"])
                    step_action=4
                    break
            #Manipulation2 just walk
            elif "manipulation2" in command_type or ("mani" in command_type and "2" in command_type):
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" in liyt:
                        name_position = "ROOM1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action = 1
                if step_action == 1:
                    name_position = "$PLACE1"
                    if "$PLACE1" in liyt:
                        name_position = "PLACE1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action = 2
                if step_action == 2:
                    name_position = "$PLACE1"
                    if "$PLACE1" in liyt:
                        name_position = "PLACE1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action = 4
                    break
            #Vision'''
            elif ("vision (enumeration)1" in command_type or (
                    "vision" in command_type and "1" in command_type and "enume" in command_type)) or (
                    "vision (enumeration)2" in command_type or (
                    "vision" in command_type and "2" in command_type and "enume" in command_type)):
                # Move
                if step_action == 0:
                    liyt = Q2
                    if ("2" in command_type):
                        name_position = "$ROOM1"
                        if "$ROOM1" in liyt:
                            name_position = "ROOM1"
                        speak("going to" + liyt[name_position])
                        walk_to(locations[liyt[name_position]])
                    else:
                        name_position = "$PLACE1"
                        if "$PLACE1" in liyt:
                            name_position = "PLACE1"
                        speak("going to" + liyt[name_position])
                        walk_to(locations[liyt[name_position]])
                    step_action = 1
                if step_action == 1:
                    time.sleep(2)
                    print("take picture")
                    # save frame
                    output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                    cv2.imwrite(output_dir + "GSPR.jpg", code_image)
                    # ask gemini
                    url = "http://192.168.50.147:8888/upload_image"
                    file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR.jpg"
                    with open(file_path, 'rb') as f:
                        files = {'image': (file_path.split('/')[-1], f)}
                        response = requests.post(url, files=files)
                        # remember to add the text question on the computer code
                    print("Upload Status Code:", response.status_code)
                    upload_result = response.json()
                    print("sent image")
                    gg = post_message_request("file", user_input, "")
                    print(gg)
                    # get answer from gemini
                    while True:
                        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == 10:
                            break
                        pass
                        time.sleep(2)
                    step_action = 2
                    speak(dictt["Voice"])
                if step_action == 2:
                    # back
                    speak("going to" + liyt["starting point"])
                    walk_to(locations["starting point"])
                    break
            elif (("vision (descridption)1" in command_type or ("vision" in command_type and "1" in command_type and "descri" in command_type))):
                if step_action == 0:
                    liyt = Q2
                    name_position = "$PLACE1"
                    if "$PLACE1" in liyt:
                        name_position = "PLACE1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action = 1
                if step_action == 1:
                    time.sleep(2)
                    print("take picture")
                    # save frame
                    output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                    cv2.imwrite(output_dir + "GSPR.jpg", code_image)
                    # ask gemini
                    url = "http://192.168.50.147:8888/upload_image"
                    file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR.jpg"
                    with open(file_path, 'rb') as f:
                        files = {'image': (file_path.split('/')[-1], f)}
                        response = requests.post(url, files=files)
                        # remember to add the text question on the computer code
                    print("Upload Status Code:", response.status_code)
                    upload_result = response.json()
                    print("sent image")
                    gg = post_message_request("file", user_input, "")
                    print(gg)
                    # get answer from gemini
                    while True:
                        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == 10:
                            break
                        pass
                        time.sleep(2)
                    step_action = 2
                    speak(dictt["Voice"])
                if step_action == 2:
                    # back
                    speak("going to" + liyt["starting point"])
                    walk_to(locations["starting point"])
                    break
            elif ("vision (descridption)2" in command_type or ("vision" in command_type and "2" in command_type and "descri" in command_type)):
                if step_action == 0:
                    liyt = Q2
                    name_position = "$PLACE1"
                    if "$PLACE1" in liyt:
                        name_position = "PLACE1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action = 1
                if step_action == 1:
                    if "height" in user_input or "tall" in user_input:
                        poses = net_pose.forward(code_image)
                        if len(poses) > 0:
                            YN = -1
                            a_num = 5
                            for issack in range(len(poses)):
                                yu = 0
                                if poses[issack][5][2] > 0:
                                    YN = 0
                                    a_num, b_num = 5, 5
                                    A = list(map(int, poses[issack][a_num][:2]))
                                    if (640 >= A[0] >= 0 and 320 >= A[1] >= 0):
                                        ax, ay, az = get_real_xyz(code_depth, A[0], A[1], 2)
                                        print(ax, ay)
                                        if az <= 2500 and az != 0:
                                            yu += 1
                                if yu >= 1:
                                    break
                        if len(A) != 0 and yu >= 1:
                            cv2.circle(code_image, (A[0], A[1]), 3, (0, 255, 0), -1)
                            target_y = ay
                            print("your height is", (1000 - target_y + 330) / 10.0)
                            final_height = (1000 - target_y + 330) / 10.0
                            step_action = 2
                    if "age" in user_input or "old" in user_input:
                        resultImg, faceBoxes = highlightFace(faceNet, code_image)

                        if not faceBoxes:
                            print("No face detected")
                            # continue
                        for faceBox in faceBoxes:
                            face = code_image[max(0, faceBox[1] - padding):
                                            min(faceBox[3] + padding, code_image.shape[0] - 1),
                                   max(0, faceBox[0] - padding):
                                   min(faceBox[2] + padding, code_image.shape[1] - 1)]
                            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                         (78.4263377603, 87.7689143744, 114.895847746),
                                                         swapRB=False)

                            ageNet.setInput(blob)
                            agePreds = ageNet.forward()
                            age = ageList[agePreds[0].argmax()]
                            print(age)
                            final_age = age
                            cv2.putText(resultImg, f'Age: {age}', (faceBox[0], faceBox[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            if "face" not in age and "not" not in age:
                                step_action = 2
                    elif "color" in user_input or "shirt" in user_input:
                        detections = dnn_yolo1.forward(code_image)[0]["det"]
                        # clothes_yolo
                        # nearest people
                        nx = 2000
                        cx_n, cy_n = 0, 0
                        CX_ER = 99999
                        need_position = 0
                        for i, detection in enumerate(detections):
                            # print(detection)
                            x1, y1, x2, y2, score, class_id = map(int, detection)
                            score = detection[4]
                            cx = (x2 - x1) // 2 + x1
                            cy = (y2 - y1) // 2 + y1
                            # depth=find_depsth
                            _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                            # cv2.rectangle(up_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            if score > 0.65 and class_id == 0 and d <= nx and d != 0 and d < CX_ER:
                                need_position = [x1, y1, x2, y2, cx, cy]
                                # ask gemini
                                cv2.rectangle(code_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.circle(code_image, (cx, cy), 5, (0, 255, 0), -1)
                                print("people distance", d)
                                CX_ER = d
                        if action1 == 0:
                            output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                            x1, y1, x2, y2 = need_position[0], need_position[1], need_position[2], need_position[3]
                            face_box = [x1, y1, x2, y2]
                            box_roi = _frame2[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                            fh, fw = abs(x1 - x2), abs(y1 - y2)
                            cv2.imwrite(output_dir + "GSPR_color.jpg", box_roi)
                            print("writed")
                            file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR_color.jpg"
                            with open(file_path, 'rb') as f:
                                files = {'image': (file_path.split('/')[-1], f)}
                                url = "http://192.168.50.147:8888/upload_image"
                                response = requests.post(url, files=files)
                                # remember to add the text question on the computer code
                            print("Upload Status Code:", response.status_code)
                            upload_result = response.json()
                            print("sent image")
                            who_help = 0
                            feature = 0
                            gg = post_message_request("color", feature, who_help)
                            print(gg)
                            # get answer from gemini
                            while True:
                                r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                                response_data = r.text
                                dictt = json.loads(response_data)
                                if dictt["Steps"] == 12:
                                    break
                                pass
                                time.sleep(2)
                            aaa = dictt["Voice"].lower()
                            speak("answer:", aaa)
                            gg = post_message_request("-1", feature, who_help)
                            action1 = 1
                            step_action = 2
                    elif "name" in user_input:
                        # jack, check, track
                        # aaron, ellen, evan
                        # angel
                        # adam, ada, aiden
                        # Vanessa, lisa, Felicia
                        # chris
                        # william
                        # max, mix
                        # hunter
                        # olivia
                        if step_speak == 0:
                            speak("hello nigga can u speak your name to me")
                            speak("speak it in complete sentence, for example, my name is fambot")
                            speak("speak after the")
                            playsound("nigga2.mp3")
                            speak("sound")
                            time.sleep(0.5)
                            playsound("nigga2.mp3")
                            step_speak = 1
                        if step_speak == 1:
                            if "check" in s or "track" in s or "jack" in s: name_cnt = "jack"
                            if "aaron" in s or "ellen" in s or "evan" in s: name_cnt = "aaron"
                            if "angel" in s: name_cnt = "angel"
                            if "adam" in s or "ada" in s or "aiden" in s: name_cnt = "adam"
                            if "vanessa" in s or "lisa" in s or "felicia" in s: name_cnt = "vanessa"
                            if "chris" in s: name_cnt = "chris"
                            if "william" in s: name_cnt = "william"
                            if "max" in s or "mix" in s: name_cnt = "max"
                            if "hunter" in s: name_cnt = "hunter"
                            if "olivia" in s: name_cnt = "olivia"

                            if name_cnt != "none":
                                speak("hello " + name_cnt + " I gonna go now.")
                                step_action=2
                if step_action == 2:
                    # back
                    speak("going to" + liyt["starting point"])
                    walk_to(locations["starting point"])
                    break
            elif "navigation1" in command_type or ("navi" in command_type and "1" in command_type):
                # follow
                liyt = Q2.json
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" in liyt:
                        name_position = "ROOM1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[liyt[name_position]])
                    step_action=1
                    step="turn"
                    action = "find"
                if step_action==1:
                    # walk in front of the guy
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" in liyt:
                        name_position = "POSE/GESTURE"
                    feature=liyt[name_position]
                    if step == "turn":
                        move(0, -0.2)
                    if step == "confirm":
                        print("imwrited")
                        file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR_people.jpg"
                        with open(file_path, 'rb') as f:
                            files = {'image': (file_path.split('/')[-1], f)}
                            url = "http://192.168.50.147:8888/upload_image"
                            response = requests.post(url, files=files)
                            # remember to add the text question on the computer code
                        print("Upload Status Code:", response.status_code)
                        upload_result = response.json()
                        print("sent image")
                        who_help = "Is the guy " + pose
                        gg = post_message_request("checkpeople", feature, who_help)
                        print(gg)
                        # get answer from gemini
                        while True:
                            r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            pass
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        if "yes" in aaa or "ys" in aaa:
                            speak("found you the guying rising hand")
                            action = "front"
                            step = "none"
                        else:
                            action = "find"
                            step = "turn"
                        gg = post_message_request(-1, feature, who_help)

                    if action == "find":
                        detections = dnn_yolo1.forward(code_image)[0]["det"]
                        # clothes_yolo
                        # nearest people
                        nx = 2000
                        cx_n, cy_n = 0, 0
                        CX_ER = 99999
                        need_position = 0
                        for i, detection in enumerate(detections):
                            # print(detection)
                            x1, y1, x2, y2, score, class_id = map(int, detection)
                            score = detection[4]
                            cx = (x2 - x1) // 2 + x1
                            cy = (y2 - y1) // 2 + y1
                            # depth=find_depsth
                            _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                            # cv2.rectangle(up_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            if score > 0.65 and class_id == 0 and d <= nx and d != 0 and (320 - cx) < CX_ER:
                                need_position = [x1, y1, x2, y2, cx, cy]
                                # ask gemini
                                cv2.rectangle(code_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.circle(code_image, (cx, cy), 5, (0, 255, 0), -1)
                                print("people distance", d)
                                CX_ER = 320 - cx
                        if need_position != 0:
                            h, w, c = code_image.shape
                            x1, y1, x2, y2, cx2, cy2 = map(int, need_position)
                            e = w // 2 - cx2
                            v = 0.001 * e
                            if v > 0:
                                v = min(v, 0.3)
                            if v < 0:
                                v = max(v, -0.3)
                            move(0, v)
                            print(e)
                            output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                            face_box = [x1, y1, x2, y2]
                            box_roi = _frame2[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                            fh, fw = abs(x1 - x2), abs(y1 - y2)
                            cv2.imwrite(output_dir + "GSPR_people.jpg", box_roi)
                            if abs(e) <= 5:
                                # speak("walk")
                                action = "none"
                                step = "confirm"
                                print("turned")
                                move(0, 0)

                    if action == "front":
                        speed = 0.2
                        h, w, c = code_image.shape
                        cx, cy = w // 2, h // 2
                        for i in range(cy + 1, h):
                            if _depth2[cy][cx] == 0 or 0 < _depth2[i][cx] < _depth2[cy][cx]:
                                cy = i
                        _, _, d = get_real_xyz(_depth2, cx, cy, 2)
                        print("depth", d)
                        if d != 0 and d <= 700:
                            action = "speak"
                            move(0, 0)
                        else:
                            move(0.2, 0)
                    if action == "speak":
                        speak("hi nigga can u stand behind me and I will follow u now")
                        speak("please say robot stop when you arrived and I will go back")
                        time.sleep(4)
                        speak("dear guest please walk")
                        action=1
                        step="none"
                        step_action=2
                # follow me
                if action == 1:
                    s = s.lower()
                    print("listening", s)
                    if "thank" in s or "you" in s or "stop" in s:
                        action = 0
                        step_action=3
                if step_action == 2:

                    msg = Twist()

                    poses = net_pose.forward(code_image)
                    min_d = 9999
                    t_idx = -1
                    for i, pose in enumerate(poses):
                        if pose[5][2] == 0 or pose[6][2] == 0:
                            continue
                        p5 = list(map(int, pose[5][:2]))
                        p6 = list(map(int, pose[6][:2]))

                        cx = (p5[0] + p6[0]) // 2
                        cy = (p5[1] + p6[1]) // 2
                        cv2.circle(code_image, p5, 5, (0, 0, 255), -1)
                        cv2.circle(code_image, p6, 5, (0, 0, 255), -1)
                        cv2.circle(code_image, (cx, cy), 5, (0, 255, 0), -1)
                        _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                        if d >= 1800 or d == 0: continue
                        if (d != 0 and d < min_d):
                            t_idx = i
                            min_d = d

                    x, z = 0, 0
                    if t_idx != -1:
                        p5 = list(map(int, poses[t_idx][5][:2]))
                        p6 = list(map(int, poses[t_idx][6][:2]))
                        cx = (p5[0] + p6[0]) // 2
                        cy = (p5[1] + p6[1]) // 2
                        _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                        cv2.circle(code_image, (cx, cy), 5, (0, 255, 255), -1)

                        print("people_d", d)
                        if d >= 1800 or d == 0: continue

                        x, z, code_image, yn = _fw.calc_cmd_vel(code_image, code_depth, cx, cy)
                        print("turn_x_z:", x, z)
                    move(x, z)
                if step_action == 3:
                    speak("going to" + liyt["starting point"])
                    walk_to(locations["starting point"])
                    break
            # Navigation2
            elif "navigation2" in command_type or ("navi" in command_type and "2" in command_type):
                liyt = Q2.json
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" in liyt:
                        name_position = "ROOM1"
                    speak("going to" + liyt[name_position])
                    walk_to(liyt[name_position])
                    step_action = 1
                    step = "turn"
                    action = "find"
                if step_action == 1:
                    # walk in front of the guy
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" in liyt:
                        name_position = "POSE/GESTURE"
                    speak("going to" + liyt[name_position])
                    feature = liyt[name_position]
                    if step == "turn":
                        move(0, -0.2)
                    if step == "confirm":
                        print("imwrited")
                        file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR_people.jpg"
                        with open(file_path, 'rb') as f:
                            files = {'image': (file_path.split('/')[-1], f)}
                            url = "http://192.168.50.147:8888/upload_image"
                            response = requests.post(url, files=files)
                            # remember to add the text question on the computer code
                        print("Upload Status Code:", response.status_code)
                        upload_result = response.json()
                        print("sent image")
                        who_help = "Is the guy " + pose
                        gg = post_message_request("checkpeople", feature, who_help)
                        print(gg)
                        # get answer from gemini
                        while True:
                            r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            pass
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        if "yes" in aaa or "ys" in aaa:
                            speak("found you the guying rising hand")
                            action = "front"
                            step = "none"
                        else:
                            action = "find"
                            step = "turn"
                        gg = post_message_request(-1, feature, who_help)

                    if action == "find":
                        detections = dnn_yolo1.forward(code_image)[0]["det"]
                        # clothes_yolo
                        # nearest people
                        nx = 2000
                        cx_n, cy_n = 0, 0
                        CX_ER = 99999
                        need_position = 0
                        for i, detection in enumerate(detections):
                            # print(detection)
                            x1, y1, x2, y2, score, class_id = map(int, detection)
                            score = detection[4]
                            cx = (x2 - x1) // 2 + x1
                            cy = (y2 - y1) // 2 + y1
                            # depth=find_depsth
                            _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                            # cv2.rectangle(up_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            if score > 0.65 and class_id == 0 and d <= nx and d != 0 and (320 - cx) < CX_ER:
                                need_position = [x1, y1, x2, y2, cx, cy]
                                # ask gemini
                                cv2.rectangle(code_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.circle(code_image, (cx, cy), 5, (0, 255, 0), -1)
                                print("people distance", d)
                                CX_ER = 320 - cx
                        if need_position != 0:
                            h, w, c = code_image.shape
                            x1, y1, x2, y2, cx2, cy2 = map(int, need_position)
                            e = w // 2 - cx2
                            v = 0.001 * e
                            if v > 0:
                                v = min(v, 0.3)
                            if v < 0:
                                v = max(v, -0.3)
                            move(0, v)
                            print(e)
                            output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                            face_box = [x1, y1, x2, y2]
                            box_roi = _frame2[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                            fh, fw = abs(x1 - x2), abs(y1 - y2)
                            cv2.imwrite(output_dir + "GSPR_people.jpg", box_roi)
                            if abs(e) <= 5:
                                # speak("walk")
                                action = "none"
                                step = "confirm"
                                print("turned")
                                move(0, 0)

                    if action == "front":
                        speed = 0.2
                        h, w, c = code_image.shape
                        cx, cy = w // 2, h // 2
                        for i in range(cy + 1, h):
                            if _depth2[cy][cx] == 0 or 0 < _depth2[i][cx] < _depth2[cy][cx]:
                                cy = i
                        _, _, d = get_real_xyz(_depth2, cx, cy, 2)
                        print("depth", d)
                        if d != 0 and d <= 700:
                            action = "speak"
                            move(0, 0)
                        else:
                            move(0.2, 0)
                    if action == "speak":
                        speak("hi nigga can u stand in front of me and I will guild u now")
                        action = 1
                        step = "none"
                        step_action = 3
                if step_action == 3:
                    name_position = "$ROOM2"
                    if "$ROOM2" in liyt:
                        name_position = "ROOM2"
                    speak("going to " + liyt[name_position])
                    walk_to(liyt[name_position])
                    speak("dear guest here is "+liyt[name_position]+" and I will go back now")
                    step_action = 3
                if step_action == 4:
                    speak("going to " + liyt["starting point"])
                    walk_to(liyt["starting point"])
                    break
            #Speech1
            elif "speech1" in command_type or ("spee" in command_type and "1" in command_type):
                liyt = Q2.json
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" in liyt:
                        name_position = "ROOM1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[name_position])
                if step_action == 1:
                    name_position = "$PLACE1"
                    if "$PLACE1" in liyt:
                        name_position = "PLACE1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[name_position])
                    if action == "speak":
                        speak("hi nigga can u stand in front of me")
                        action = 1
                        step = "none"
                        step_action = 2
                if step_action == 2: #get text
                    #question detect
                    s=s.lower()
                    now = datetime.now()
                    s = s.lower()
                    current_time = now.strftime("%H:%M:%S")
                    current_month = now.strftime("%B")  # Full month name
                    current_day_name = now.strftime("%A")  # Full weekday name
                    day_of_month = now.strftime("%d")
                    answer="none"
                    none_cnt=0
                    speak("dear guest please speak complete sentence after the")
                    playsound("nigga2.mp3")
                    speak("sound")
                    time.sleep(0.5)
                    speak("for example hi robot what day is it today")
                    time.sleep(0.5)
                    playsound("nigga2.mp3")
                    while True:
                        if "what" in s:
                            if "today" in s:
                                answer = f"It is {current_month} {day_of_month}"
                            elif "team" in s and "name" in s:
                                answer = "My team's name is FAMBOT"
                            elif "tomorrow" in s:
                                answer = f"It is {current_month} {day_of_month + 1}"  # Note: you might need to handle month boundaries
                            elif "your" in s and "name" in s:
                                answer = "My name is FAMBOT robot"
                            elif "time" in s:
                                answer = f"It is {current_time}"
                            elif "capital" in s or "shiga" in s or "cap" in s:
                                answer = "Ōtsu is the capital of Shiga Prefecture, Japan"
                            elif "venue" in s in s or "2025" in 2 or "open" in s:
                                answer = "The name of the venue is Shiga Daihatsu Arena"
                            elif "week" in s:
                                answer = f"It is {current_day_name}"
                            elif "month" in s:
                                answer = f"It is {day_of_month}"
                            elif "plus" in s or "half":
                                answer = "It is 1.5"
                            elif "back" in s or "bird" in s:
                                answer = "It is the hummingbird"
                            elif "mammal" in s or "fly" in s:
                                answer = "It is the bat"
                            elif "broken" in s:
                                answer = "An egg"
                            elif "tell" in s and "about" in s:
                                answer = "I am a home robot called fambot and I am 4 years old in 2025"
                            else:
                                answer = "none"

                        elif "where" in s:
                            if "from" in s:
                                answer = "I am from Macau Puiching middle school, Macau China"
                            else:
                                answer = "none"

                        elif "how" in s:
                            if "member" in s or "team" in s:
                                answer = "We have 4 members in our team"
                            elif "day" in s or "week" in s:  # Duplicate from 'what' questions
                                answer = "There are seven days in a week"
                            else:
                                answer = "none"

                        elif "who" in s:
                            if "leader" in s:
                                answer = "Our Team leader is Wu Iat Long"
                            else:
                                answer = "none"

                        else:
                            answer = "none"
                        none_cnt+=1
                        if answer == "none" and none_cnt>=30 and s!=pre_s:
                            speak("can u please speak it again")
                            none_cnt=0
                        else:
                            speak(answer)
                            break
                    step_action = 3
                if step_action == 3:
                    post_message_request("answer_list", "", question)
                    while True:
                        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == "answer1":
                            break
                        pass
                        time.sleep(2)

                    post_message_request(-1, "", question)
                    speak(dictt["answer"])
                    time.sleep(1)
                    speak("I will go back now bye bye")
                    step_action = 4
                if step_action == 4:
                    speak("going to " + "starting point")
                    walk_to(locations["starting point"])
                    break
            # Speech2
            elif "speech2" in command_type or ("spee" in command_type and "2" in command_type):
                liyt = Q2.json
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" in liyt:
                        name_position = "ROOM1"
                    speak("going to" + liyt[name_position])
                    walk_to(locations[name_position])
                if step_action == 1:
                    # walk in front of the guy
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" in liyt:
                        name_position = "POSE/GESTURE"
                    feature = liyt[name_position]
                    if step == "turn":
                        move(0, -0.2)
                    if step == "confirm":
                        print("imwrited")
                        file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR_people.jpg"
                        with open(file_path, 'rb') as f:
                            files = {'image': (file_path.split('/')[-1], f)}
                            url = "http://192.168.50.147:8888/upload_image"
                            response = requests.post(url, files=files)
                            # remember to add the text question on the computer code
                        print("Upload Status Code:", response.status_code)
                        upload_result = response.json()
                        print("sent image")
                        who_help = "Is the guy " + pose
                        gg = post_message_request("checkpeople", feature, who_help)
                        print(gg)
                        # get answer from gemini
                        while True:
                            r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            pass
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        if "yes" in aaa or "ys" in aaa:
                            speak("found you the guying rising hand")
                            action = "front"
                            step = "none"
                        else:
                            action = "find"
                            step = "turn"
                        gg = post_message_request(-1, feature, who_help)
                    if action == "find":
                        detections = dnn_yolo1.forward(code_image)[0]["det"]
                        # clothes_yolo
                        # nearest people
                        nx = 2000
                        cx_n, cy_n = 0, 0
                        CX_ER = 99999
                        need_position = 0
                        for i, detection in enumerate(detections):
                            # print(detection)
                            x1, y1, x2, y2, score, class_id = map(int, detection)
                            score = detection[4]
                            cx = (x2 - x1) // 2 + x1
                            cy = (y2 - y1) // 2 + y1
                            # depth=find_depsth
                            _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                            # cv2.rectangle(up_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            if score > 0.65 and class_id == 0 and d <= nx and d != 0 and (320 - cx) < CX_ER:
                                need_position = [x1, y1, x2, y2, cx, cy]
                                # ask gemini
                                cv2.rectangle(code_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.circle(code_image, (cx, cy), 5, (0, 255, 0), -1)
                                print("people distance", d)
                                CX_ER = 320 - cx
                        if need_position != 0:
                            h, w, c = code_image.shape
                            x1, y1, x2, y2, cx2, cy2 = map(int, need_position)
                            e = w // 2 - cx2
                            v = 0.001 * e
                            if v > 0:
                                v = min(v, 0.3)
                            if v < 0:
                                v = max(v, -0.3)
                            move(0, v)
                            print(e)
                            output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                            face_box = [x1, y1, x2, y2]
                            box_roi = _frame2[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                            fh, fw = abs(x1 - x2), abs(y1 - y2)
                            cv2.imwrite(output_dir + "GSPR_people.jpg", box_roi)
                            if abs(e) <= 5:
                                # speak("walk")
                                action = "none"
                                step = "confirm"
                                print("turned")
                                move(0, 0)
                    if action == "front":
                        speed = 0.2
                        h, w, c = code_image.shape
                        cx, cy = w // 2, h // 2
                        for i in range(cy + 1, h):
                            if _depth2[cy][cx] == 0 or 0 < _depth2[i][cx] < _depth2[cy][cx]:
                                cy = i
                        _, _, d = get_real_xyz(_depth2, cx, cy, 2)
                        print("depth", d)
                        if d != 0 and d <= 700:
                            action = "speak"
                            move(0, 0)
                        else:
                            move(0.2, 0)
                    if action == "speak":
                        action = 1
                        step = "none"
                        step_action = 2
                if step_action==2:

                    name_position = "$TELL_LIST"
                    if "$TELL_LIST" in liyt:
                        name_position = "TELL_LIST"
                    question = "My question is "+liyt[name_position]
                    post_message_request("talk_list","",question)
                    while True:
                        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == "answer2":
                            break
                        pass
                        time.sleep(2)

                    post_message_request(-1, "", question)
                    speak(dictt["answer"])
                    time.sleep(1)
                    speak("I will go back now bye bye")
                    step_action=3
                if step_action == 3:
                    speak("going to " + "starting point")
                    walk_to(locations["starting point"])
                    break
            else:
                speak("I can't do it")
                break
