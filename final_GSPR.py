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
import re
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
from datetime import datetime

# gemini2
def callback_image2(msg):
    global _frame2
    _frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth2(msg):
    global _depth2
    _depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
def callback_image1(msg):
    global _frame1
    _frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")

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
        speak("going to " + str(name))
        name = name.lower()
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


locations = {
    # Furniture and objects
    "wall long": [0, 0, 0],
    "wall short": [0, 0, 0],
    "long table a": [3.154, 2.870, 1.53],
    "long table b": [3.350, 3.111, -1.5],
    "tall table": [2.507, 3.287, -1.607],
    "shelf": [-0.715, -0.193, 1.569],
    "chair a": [-0.261, -0.067, 0],
    "chair b": [-0.265, 0.633, 0],
    "tray a": [2.490, 3.353, 1.53],
    "tray b": [3.238, 3.351, 1.53],
    "container": [3.829, 3.092, 1.55],
    "pen holder": [3.031, 3.436, 1.53],
    "trash bin a": [-1.182, 3.298, 3.12],
    "trash bin b": [5.080, 3.032, 1.54],
    "storage box a": [-1.058, 4.001, 3.11],
    "storage box b": [5.661, 3.102, 1.54],

    # Locations and special points
    "starting point": [3.809, 2.981, 3.053],
    "exit": [6.796, 3.083, 0],
    "host": [-0.967, -0.013, -1.709],
    "dining room": [-0.397, 0.297, 0],
    "living room": [3.364, 2.991, 1.436],
    "hallway": [0.028, 3.514, 3.139]
}
# front 0 back 3.14 left 90 1.5 right 90 -1.5
cout_location = {
    "living room": [1.153, 3.338, 0],
    "hallway": [1.153, 3.338, 3.14],
    "dining room": [-1.581, -0.345, 0.15]
}


def walk_to1(name):
    if "none" not in name or "unknow" in name:
        speak("going to " + str(name))
        name = name.lower()
        real_name = check_item(name)
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
        time.sleep(1)
        clear_costmaps


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    # open things
    chassis = RobotChassis()

    print("cmd_vel")
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
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
    _frame1 = None
    _sub_down_cam_image1 = rospy.Subscriber("/cam1/color/image_raw", Image, callback_image1)
    print("gemini2 depth")
    _depth1 = None
    _sub_down_cam_depth1 = rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth1)
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    follow_cnt = 0
    action = 0
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    step_action = 0
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    print("yolov8")
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo1 = Yolov8("yolov8n", device_name="GPU")
    s = ""
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    # step_action
    # add action for all code
    # Step 0 first send
    # Step 1 first get
    # Step 9 send image response text
    # step 10 get the image response
    walk_to("starting point")

    speak("please say start, then I will go to the host point")


    while True:
        if "start" in s or "stop" in s:
            break
    step = "none"
    confirm_command = 0
    walk_to("host")
    command_list=[
                  "Say hello to the person wearing a white clothes in the hallway and say where RoboCup is held this year",
                  "Go to the hallway then meet Angel and answer a quiz",
                  "Find a sitting person in the hallway and take them to the tray B",
                  "Find a standing person in the dining room and follow them to the hallway",
                  "Tell me what is the largest object on the chair B",
                  "Tell me how many kitchen items there are on the chair B",
                  "Tell me how many people in the dining room are wearing black sweaters",
                  "Tell me the height of the person at the chair B",
                  "Tell me the name of the person at the chair B",
                  "Get a lunch box from the chair A and put it on the long table A",
                  "Give me a coffee from the long table A"]
    for i in range(0,10):

        qr_code_detector = cv2.QRCodeDetector()
        data = ""
        speak("dear host please scan your qr code in front of my camera on top")
        data = command_list[i]
        '''
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
        cv2.destroyAllWindows()'''

        # confirm
        if confirm_command == 0:
            speak("dear host your command is")
            time.sleep(0.3)
            print("Yout command is **********************")
            print(data)
            speak(str(data))
            print("********************")
            time.sleep(0.3)
            speak("to confirm your command plase answer robot yes yes yes or robot no no no,  thank you")
            confirm_command = 1
        while True:
            if "yes" in s:
                speak("ok")
                break
        user_input = data
        # post question
        gg = post_message_request("first", user_input, "")  # step
        print("post", gg)
        # get gemini answer
        nigga = 1
        while True:
            r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
            response_data = r.text
            dictt = json.loads(response_data)
            if dictt["Steps"] == 1:
                break
            time.sleep(2)
        Q1 = dictt["Question1"]
        Q2 = dictt["Question2"]
        Q3 = dictt["Question3"]
        print(Q1)
        print(Q2)
        Q3=str(Q3)
        Q3=Q3.replace("['","")
        Q3=Q3.replace("']","")
        
        Q3="I should "+Q3
        Q3=Q3.replace("me","you")
        speak(Q3)
        
        # say how the robot understand
        # speak(Q3[0])
        # divide
        command_type = str(Q1[0])
        command_type = command_type.lower()
        step_action = 0
        # continue
        liyt = Q2
        gg = post_message_request("-1", "", "")
        pre_s = ""
        name_cnt = "none"
        ageList = ['1', '5', '10', '17', '27', '41', '50', '67']
        # Initialize video capture
        # video = cv2.VideoCapture(args.video if args.video else 0)
        padding = 20
        confirm_command = 0
        s = ""
        action1 = 0
        step_speak = 0
        age_cnt = 0
        failed_cnt = 0
        final_speak_to_guest = ""
        feature = "none"
        skip_cnt_vd=0
        nav1_skip_cnt=0
        output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
        while not rospy.is_shutdown():
            # voice check
            # break
            now1 = datetime.now()
            current_time = now1.strftime("%H:%M:%S")
            rospy.Rate(10).sleep()
            if step_action == 100 or step_action == 101:
                break
            confirm_command = 0
            if s != "" and s != pre_s:
                print(s)
                pre_s = s
            if _frame2 is None:
                print("no frame")
            if _depth2 is None:
                print("no depth")
            code_image = _frame2.copy()
            code_depth = _depth2.copy()
            catch_image = _frame1.copy()
            cv2.imshow("frame", code_image)
            key = cv2.waitKey(1)
            if key in [ord('q'), 27]:
                break
            # DIVIDE
            # Manipulation1 just walk
            if "manipulation1" in command_type or ("mani" in command_type and "1" in command_type):
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step_action = 1
                if step_action == 1:
                    name_position = "$PLACE1"
                    if "$PLACE1" not in liyt:
                        name_position = "PLACE1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    time.sleep(2)
                    speak("robot arm is in error")
                    step_action = 2
                if step_action == 2:
                    name_position = "$PLACE2"
                    if "$PLACE2" not in liyt:
                        name_position = "PLACE2"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step_action = 100
                    final_speak_to_guest = ""
            # Manipulation2 just walk
            elif "manipulation2" in command_type or ("mani" in command_type and "2" in command_type):
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step_action = 1
                if step_action == 1:
                    name_position = "$PLACE1"
                    if "$PLACE1" not in liyt:
                        name_position = "PLACE1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    time.sleep(2)
                    speak("robot arm is in error")
                    step_action = 100
                    final_speak_to_guest = "here you are"
            # Vision E 1,2
            elif ("vision (enumeration)1" in command_type or (
                    "vision" in command_type and "1" in command_type and "enume" in command_type)) or (
                    "vision (enumeration)2" in command_type or (
                    "vision" in command_type and "2" in command_type and "enume" in command_type)):
                # Move
                if step_action == 0:
                    liyt = Q2
                    if ("2" in command_type):
                        name_position = "$ROOM1"
                        if "$ROOM1" not in liyt:
                            name_position = "ROOM1"
                        if name_position in liyt:
                            walk_to1(liyt[name_position])
                    else:
                        name_position = "$PLACE1"
                        if "$PLACE1" not in liyt:
                            name_position = "PLACE1"
                        if name_position in liyt:
                            walk_to(liyt[name_position])
                    step_action = 1
                if step_action == 1:
                    time.sleep(2)
                    print("take picture")
                    # save frame
                    output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                    if ("2" in command_type):
                        cv2.imshow("capture_vision_(enumeration)2_img", _frame2)
                        cv2.imwrite(output_dir + "GSPR.jpg", _frame2)
                    else:
                        image_flip = cv2.flip(_frame1, 0) 
                        cv2.imshow("capture_vision_(enumeration)1_img", image_flip)
                        cv2.imwrite(output_dir + "GSPR.jpg", image_flip)
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
                    gg = post_message_request("Enumeration", user_input, "")
                    print(gg)
                    # get answer from gemini
                    while True:
                        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == 10:
                            break
                        time.sleep(2)
                    step_action = 100
                    final_speak_to_guest = dictt["Voice"]
                    gg = post_message_request("-1", "", "")
                    current_file_name = output_dir + "GSPR" + str(current_time) + ".jpg"
                    new_file_name = output_dir + "GSPR.jpg"
                    try:
                        os.rename(new_file_name, current_file_name)
                        print("File renamed successfully.")
                    except FileNotFoundError:
                        print("File renamed failed")
                    except PermissionError:
                        print("File renamed failed")
            # vision D1
            elif (("vision (descridption)1" in command_type or (
                    "vision" in command_type and "1" in command_type and "descri" in command_type))):
                if step_action == 0:
                    liyt = Q2
                    name_position = "$PLACE1"
                    if "$PLACE1" not in liyt:
                        name_position = "PLACE1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step_action = 1
                if step_action == 1:
                    time.sleep(2)
                    print("take picture")
                    # save frame
                    image_flip = cv2.flip(_frame1, 0) 
                    output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                    cv2.imshow("capture_vision_(descridption)1_img", image_flip)
                    cv2.imwrite(output_dir + "GSPR.jpg", image_flip)
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
                    gg = post_message_request("Description", user_input, "")
                    print(gg)
                    # get answer from gemini
                    while True:
                        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == 10:
                            break
                        time.sleep(2)
                    step_action = 100
                    final_speak_to_guest = dictt["Voice"]
                    gg = post_message_request("-1", "", "")
                    current_file_name = output_dir + "GSPR" + str(current_time) + ".jpg"
                    new_file_name = output_dir + "GSPR.jpg"
                    try:
                        os.rename(new_file_name, current_file_name)
                        print("File renamed successfully.")
                    except FileNotFoundError:
                        print("File renamed failed")
                    except PermissionError:
                        print("File renamed failed")
            # vision D2
            elif ("vision (descridption)2" in command_type or (
                    "vision" in command_type and "2" in command_type and "descri" in command_type)):
                if step_action == 0:
                    liyt = Q2
                    name_position = "$PLACE1"
                    if "$PLACE1" not in liyt:
                        name_position = "PLACE1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step_action = 1
                    skip_cnt_vd=0
                if step_action == 1:
                    if "height" in user_input or "tall" in user_input:
                        code_image = _frame2.copy()
                        poses = net_pose.forward(code_image)
                        yu = 0
                        ay = 0
                        A = []
                        skip_cnt_vd+=1
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
                        if skip_cnt_vd>=500:
                            step_action = 2
                            speak("I can't see the guy I gonna go now")
                        if len(A) != 0 and yu >= 1:
                            cv2.circle(code_image, (A[0], A[1]), 3, (0, 255, 0), -1)
                            target_y = ay
                            print("your height is", (1000 - target_y + 330) / 10.0)
                            final_height = (1000 - target_y + 330) / 10.0
                            step_action = 2
                            final_speak_to_guest = "the guys height is " + str(final_height)
                    if "age" in user_input or "old" in user_input:
                        code_image = _frame2.copy()
                        resultImg, faceBoxes = highlightFace(faceNet, code_image)
                        age_cnt += 1
                        final_age=20
                        if not faceBoxes:
                            print("No face detected")
                            # continue
                        for faceBox in faceBoxes:
                            face = _frame2[max(0, faceBox[1] - padding):
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
                            if "face" not in age and "not" not in age or age_cnt >= 100:
                                step_action = 2
                                final_speak_to_guest = "the guys is " + str(final_age) + " years old"
                    elif "color" in user_input or "shirt" in user_input:
                        code_image = _frame2.copy()
                        detections = dnn_yolo1.forward(code_image)[0]["det"]
                        # clothes_yolo
                        # nearest people
                        nx = 2000
                        cx_n, cy_n = 0, 0
                        CX_ER = 99999
                        need_position = 0
                        skip_cnt_vd+=1
                        if skip_cnt_vd>=500:
                            step_action = 2
                            speak("I can't see the guy I gonna go now")
                        for i, detection in enumerate(detections):
                            # print(detection)
                            x1, y1, x2, y2, _, class_id = map(int, detection)
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
                                time.sleep(2)
                            final_speak_to_guest = dictt["Voice"]
                            gg = post_message_request("-1", "", "")
                            current_file_name = output_dir + "GSPR_color" + str(current_time) + ".jpg"
                            new_file_name = output_dir + "GSPR_color.jpg"
                            try:
                                os.rename(new_file_name, current_file_name)
                                print("File renamed successfully.")
                            except FileNotFoundError:
                                print("File renamed failed")
                            except PermissionError:
                                print("File renamed failed")
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
                            speak("hello guest can u speak your name to me")
                            speak("speak it in complete sentence, for example, my name is fambot")
                            speak("speak after the")
                            playsound("nigga2.mp3")
                            speak("sound")
                            time.sleep(0.5)
                            playsound("nigga2.mp3")
                            step_speak = 1
                        if step_speak == 1:
                            skip_cnt_vd += 1
                            s=s.lower()
                            if skip_cnt_vd >= 500:
                                step_action = 2
                                speak("I can't hear you I gonna go now")
                                print(skip_cnt_vd)
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
                                final_speak_to_guest = "the guys name is " + name_cnt
                                step_action = 2
                if step_action == 2:
                    step_action = 100
            # navigation 1 ***
            elif "navigation1" in command_type or ("navi" in command_type and "1" in command_type):
                # follow
                liyt = Q2
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step_action = 1
                    step = "turn"
                    action = "find"
                    nav1_skip_cnt=0
                if step_action == 1:
                    # walk in front of the guy
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" not in liyt:
                        name_position = "POSE/GESTURE"
                    if name_position in liyt:
                        feature = liyt[name_position]
                    if step == "turn":
                        move(0, -0.2)
                        nav1_skip_cnt+=1
                        if nav1_skip_cnt>=30:
                            step = "none"
                            action = "none"
                            step_action = 3
                            speak("I can't find you I gonna go back to the host")
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
                        who_help = "Is the guy " + feature
                        gg = post_message_request("checkpeople", feature, who_help)
                        print(gg)
                        # get answer from gemini
                        while True:
                            r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        current_file_name = output_dir + "GSPR_people" + str(current_time) + ".jpg"
                        new_file_name = output_dir + "GSPR_people.jpg"
                        try:
                            os.rename(new_file_name, current_file_name)
                            print("File renamed successfully.")
                        except FileNotFoundError:
                            print("File renamed failed")
                        except PermissionError:
                            print("File renamed failed")
                        if "yes" in aaa or "ys" in aaa:
                            speak("found you the guest rising hand")
                            action = "front"
                            step = "none"
                        else:
                            action = "find"
                            step = "turn"
                        gg = post_message_request("-1", feature, who_help)

                    if action == "find":
                        code_image = _frame2.copy()
                        detections = dnn_yolo1.forward(code_image)[0]["det"]
                        # clothes_yolo
                        # nearest people
                        nx = 2000
                        cx_n, cy_n = 0, 0
                        CX_ER = 99999
                        need_position = 0
                        for i, detection in enumerate(detections):
                            # print(detection)
                            x1, y1, x2, y2, _, class_id = map(int, detection)
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
                        speak("hello dear guest can u stand behind me and I will follow u now")
                        time.sleep(7)
                        speak("dear guest please say robot you can stop")
                        time.sleep(0.5)
                        speak("when you arrived and I will go back")
                        time.sleep(0.5)
                        speak("hello dear guest please walk but don't walk too fast, and remember to say robot stop when you arrived thank you")
                        action = 1
                        step = "none"
                        step_action = 2
                # follow me
                if action == 1:
                    s = s.lower()
                    print("listening", s)
                    if "thank" in s or "you" in s or "stop" in s or "arrive" in s or "robot" in s:
                        action = 0
                        step_action = 3
                if step_action == 2:
                    msg = Twist()
                    code_image = _frame2.copy()
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
                    step_action = 100
            # Navigation2
            elif "navigation2" in command_type or ("navi" in command_type and "2" in command_type):
                liyt = Q2
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step_action = 1
                    step = "turn"
                    action = "find"
                    nav2_skip_cnt=0
                if step_action == 1:
                    # walk in front of the guy
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" not in liyt:
                        name_position = "POSE/GESTURE"
                    if name_position in liyt:
                        feature = liyt[name_position]
                    if step == "turn":
                        move(0, -0.2)
                        nav2_skip_cnt =0
                        if nav2_skip_cnt>=30:
                            step = "none"
                            action = "none"
                            step_action = 3
                            speak("I can't find you I gonna go back to the host")
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
                        who_help = "Is the guy " + feature
                        gg = post_message_request("checkpeople", feature, who_help)
                        print(gg)
                        # get answer from gemini
                        while True:
                            r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        current_file_name = output_dir + "GSPR_people" + str(current_time) + ".jpg"
                        new_file_name = output_dir + "GSPR_people.jpg"
                        try:
                            os.rename(new_file_name, current_file_name)
                            print("File renamed successfully.")
                        except FileNotFoundError:
                            print("File renamed failed")
                        except PermissionError:
                            print("File renamed failed")
                        if "yes" in aaa or "ys" in aaa:
                            speak("found you the guying rising hand")
                            action = "front"
                            step = "none"
                        else:
                            action = "find"
                            step = "turn"
                        gg = post_message_request("-1", feature, who_help)

                    if action == "find":
                        code_image = _frame2.copy()
                        detections = dnn_yolo1.forward(code_image)[0]["det"]
                        # clothes_yolo
                        # nearest people
                        nx = 2000
                        cx_n, cy_n = 0, 0
                        CX_ER = 99999
                        need_position = 0
                        for i, detection in enumerate(detections):
                            # print(detection)
                            x1, y1, x2, y2, _, class_id = map(int, detection)
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
                        speak("hello dear guest can u stand in front of me and I will guild u now")
                        action = 1
                        step = "none"
                        step_action = 3
                if step_action == 3:
                    name_position = "$ROOM2"
                    if "$ROOM2" not in liyt:
                        name_position = "ROOM2"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    speak("dear guest here is " + liyt[name_position] + " and I will go back now")
                    step_action = 100
            # Speech1
            elif "speech1" in command_type or ("spee" in command_type and "1" in command_type):
                liyt = Q2
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                if step_action == 1:
                    name_position = "$PLACE1"
                    if "$PLACE1" not in liyt:
                        name_position = "PLACE1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    if action == "speak":
                        speak("hello dear guest can u stand in front of me")
                        action = 1
                        step = "none"
                        step_action = 2
                if step_action == 2:  # get text
                    # question detect
                    answer = "none"
                    none_cnt = 0
                    speak("dear guest please speak your question in complete sentence after the")
                    playsound("nigga2.mp3")
                    speak("sound")
                    time.sleep(0.5)
                    speak("for example hi robot what day is it today")
                    time.sleep(0.5)
                    playsound("nigga2.mp3")
                    step_action = 3
                if step_action == 3:
                    now1 = datetime.now()
                    s = s.lower()
                    current_time = now1.strftime("%H:%M:%S")
                    current_month = now1.strftime("%B")  # Full month name
                    current_day_name = now1.strftime("%A")  # Full weekday name
                    day_of_month = now1.strftime("%d")
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
                            number_words = {
                                "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                                "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
                                "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
                                "eighteen": "18", "nineteen": "19", "twenty": "20"
                            }
                            for word, digit in number_words.items():
                                s = s.replace(word, digit)
                            numbers = list(map(int, re.findall(r'\d+', s)))
                            operation = re.search(r'(plus|minus|times|divided by)', s).group(1)
                            result = "Unknown talk list I can't answer it"
                            if operation and len(numbers) >= 2:
                                a, b = numbers[0], numbers[1]
                                if operation == 'plus':
                                    result = a + b
                                elif operation == 'minus':
                                    result = a - b
                                elif operation == 'times':
                                    result = a * b
                                elif operation == 'divided by':
                                    result = a / b
                            speak(result)
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
                    none_cnt += 1
                    if failed_cnt > 5:
                        speak("I can't get your question, I gonna go back now")
                        step_action = 4
                    if answer == "none" and none_cnt >= 30 and s != pre_s:
                        speak("can u please speak it again")
                        none_cnt = 0
                        failed_cnt += 1
                    else:
                        speak(answer)
                        step_action = 4
                if step_action == 4:
                    '''
                    question=s
                    post_message_request("answer_list", "", question)
                    while True:
                        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == "answer1":
                            break
                        pass
                        time.sleep(2)
                    speak(dictt["answer"])
                    time.sleep(1)
                    post_message_request("-1", "", "")'''
                    speak("I will go back now bye bye")
                    step_action = 100
            # Speech2
            elif "speech2" in command_type or ("spee" in command_type and "2" in command_type):
                liyt = Q2
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step = "turn"
                    action = "find"
                    speech2_turn_skip=0
                if step_action == 1:
                    # walk in front of the guy
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" not in liyt:
                        name_position = "POSE/GESTURE"
                    if name_position in liyt:
                        feature = liyt[name_position]
                    if step == "turn":
                        move(0, -0.2)
                        speech2_turn_skip+=1
                        if speech2_turn_skip>=30:
                            speech2_turn_skip=0
                            step = "none"
                            action = "none"
                            speak("I can't find you I gonna go back to the host")
                            step_action = 2
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
                        who_help = "Is the guy " + feature
                        gg = post_message_request("checkpeople", feature, who_help)
                        print(gg)
                        # get answer from gemini
                        while True:
                            r = requests.get("http://192.168.50.147:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        current_file_name = output_dir + "GSPR_people" + str(current_time) + ".jpg"
                        new_file_name = output_dir + "GSPR_people.jpg"
                        try:
                            os.rename(new_file_name, current_file_name)
                            print("File renamed successfully.")
                        except FileNotFoundError:
                            print("File renamed failed")
                        except PermissionError:
                            print("File renamed failed")
                        if "yes" in aaa or "ys" in aaa:
                            speak("found you the guy " + str(feature))
                            action = "front"
                            step = "none"
                        else:
                            action = "find"
                            step = "turn"
                        gg = post_message_request("-1", feature, who_help)
                    if action == "find":
                        code_image = _frame2.copy()
                        detections = dnn_yolo1.forward(code_image)[0]["det"]
                        # clothes_yolo
                        # nearest people
                        nx = 2000
                        cx_n, cy_n = 0, 0
                        CX_ER = 99999
                        need_position = 0
                        for i, detection in enumerate(detections):
                            # print(detection)
                            x1, y1, x2, y2, _, class_id = map(int, detection)
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
                        action = "none"
                        step_action = 2
                if step_action == 2:
                    now = datetime.now()
                    name_position = "$TELL_LIST"
                    if "$TELL_LIST" not in liyt:
                        name_position = "TELL_LIST"
                    current_time = now.strftime("%H:%M:%S")
                    question = "My question is " + liyt[name_position]
                    speak("dear guest")
                    time.sleep(1)
                    if "something about yourself" in command_type or ("something" in command_type and "yourself" in command_type):
                        speak("We are Fambot from Macau Puiching Middle School, and I was made in 2024")
                    elif "what day today is" in command_type or ("today" in command_type and "day" in command_type):
                        speak("today is 25 th April in 2025")
                    elif "what day tomorrow is" in command_type or ("tomorrow" in command_type and "say" in command_type):
                        speak("today is 26 th April in 2025")
                    elif "where robocup is held this year" in command_type or ("where" in command_type and "robocup" in command_type and "year" in command_type):
                        speak("the robocup 2025 is held in Brazil,Salvador")
                    elif "your team's name" in command_type or ("name" in command_type and "team" in command_type):
                        speak("my team name is Fambot")
                    elif "where you come from" in command_type or ("where" in command_type and "come" in command_type and "from" in command_type):
                        speak("We are Fambot from Macau Puiching Middle School")
                    elif "what the weather is like today" in command_type or ("weather" in command_type and "today" in command_type and "what" in command_type):
                        speak("today weather is Raining")
                    elif "what the time is" in command_type or ("what" in command_type and "time" in command_type):
                        speak("the current time is" + current_time)
                    else:
                        numbers = list(map(int, re.findall(r'\d+', command_type)))
                        operation = re.search(r'(plus|minus|times|divided by)', command_type).group(1)
                        result = "Unknown talk list I can't answer it"
                        if operation and len(numbers)>=2:
                            a, b = numbers[0], numbers[1]
                            if operation == 'plus':
                                result = a + b
                            elif operation == 'minus':
                                result = a - b
                            elif operation == 'times':
                                result = a * b
                            elif operation == 'divided by':
                                result = a / b
                        speak("hello dear guest"+str(result))
                    step_action = 3
                if step_action == 3:
                    time.sleep(1)
                    speak("I will go back now bye bye")
                    step_action = 100
            else:
                speak("I can't do it take the next command please")
                break
        walk_to("host")
        speak(final_speak_to_guest)
        time.sleep(2)
    walk_to("exit")
    speak("end")
