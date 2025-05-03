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
    print("[robot say]:", end=" ")
    os.system(f'espeak -s 165 "{g}"')
    # rospy.loginfo(g)
    print(g)
    time.sleep(0.3)


def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def post_message_request(step, s1, question):
    api_url = "http://192.168.60.48:8888/Fambot"
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


def check_item(name):
    corrected = "entrance"
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
        if real_name in locations:
            speak("going to " + str(name))
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
    "counter": [3.154, 2.870, 1.53],
    "left tray": [3.350, 3.111, -1.5],
    "right tray": [2.507, 3.287, -1.607],
    "pen holder": [3.154, 2.870, 1.53],
    "container": [3.350, 3.111, -1.5],
    "left kachaka shelf": [2.507, 3.287, -1.607],
    "right kachaka shelf": [-0.715, -0.193, 1.569],
    "low table": [-1.182, 3.298, 3.12],
    "left chair": [-0.261, -0.067, 0],
    "right chair": [-0.265, 0.633, 0],
    "trash bin": [2.490, 3.353, 1.53],
    "tall table": [3.238, 3.351, 1.53],
    "left kachaka station": [3.829, 3.092, 1.55],
    "right kachaka station": [3.031, 3.436, 1.53],
    "shelf": [-1.182, 3.298, 3.12],
    # bed
    "bed": [5.080, 3.032, 1.54],
    # dining room
    "dining table": [-1.058, 4.001, 3.11],
    "couch": [5.661, 3.102, 1.54],

    # Locations and special points
    "entrance": [3.809, 2.981, 3.053],
    "exit": [6.796, 3.083, 0],
    "instruction point": [-0.967, -0.013, -1.709],
    "dining room": [-0.397, 0.297, 0],
    "living room": [3.364, 2.991, 1.436],
    "bedroom": [0.028, 3.514, 3.139],
    "study room": [-0.397, 0.297, 0]
}
# front 0 back 3.14 left 90 1.5 right 90 -1.5
cout_location = {
    "living room": [1.153, 3.338, 0],
    "bedroom": [1.153, 3.338, 3.14],
    "dining room": [-1.545, -0.303, 0.4],
    "study room": [-1.581, -0.345, 0.15]
}

dining_room_dif = {
    "din1": [-1.545, -0.303, 1.57],
    "din2": [1.214, 1.960, -1.57]  ##
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
    robot_height = 1000
    # step_action
    # add action for all code
    # Step 0 first send
    # Step 1 first get
    # Step 9 send image response text
    # step 10 get the image response
    gg = post_message_request("-1", "", "")
    speak("please say start, then I will go to the instruction point")
    while True:
        print("speak",s)
        if "start" in s or "stop" in s:
            break
        time.sleep(1)
    step = "none"
    confirm_command = 0
    walk_to("instruction point")
    command_list = [
        "Guide the person wearing a orange jacket from the right Kachaka station to the left Kachaka station",
        "Give me a cookies from the tall table",
        "Tell me how many people in the dining room are wearing white t-shirt",
        "Meet Basil at the tall table then look for them in the study room",
        "Tell me how many people crossing one's arms are in the study room",
        "Tell me what is the thinnest object on the shelf",
        "Tell me how many task items there are on the right tray",
        "Lead the person pointing to the left from the right Kachaka station to the bed",
        "Follow the squatting person at the pen holder",
        "Grasp a noodles from the trash bin and put it on the container",
        "Follow Sophia from the left tray to the dining room",
        "Tell me how many kitchen items there are on the trash bin",
        "Tell me the age of the person standing in the living room",
        "Tell me how tall of the person standing in the living room",
        "Tell me the name of the person standing in the living room",
        "Say what day today is to the person raising their right arm in the dining room",
        "Give me a cup from the right tray",

        "Meet Basil in the dining room and answer a question",
        "Tell me the shirt color of the person standing in the living room",
        "what color of t-shirt Jack is wearing in the dining room",
        "Give me a light bulb from the trash bin",
        "Fetch a glue gun from the left Kachaka shelf and put it on the left tray"
    ]
    for i in range(1, 4):
        dining_room_action = 0
        qr_code_detector = cv2.QRCodeDetector()
        data = ""
        speak("dear host please scan your qr code in front of my camera on top")
        data = command_list[i]
        yn = 0
        while True:
            if yn == 1:
                break
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
            #data = command_list[i]
            #continue
            if "dining" in data:
                dining_room_action = 1
            speak("dear host your command is")
            time.sleep(0.3)
            print("Yout command is **********************")
            print(data)
            speak(str(data))
            print("********************")
            time.sleep(0.3)
            speak("to confirm your command plase answer robot yes yes yes or robot no no no,  thank you")
            while True:
                print("s",s)
                time.sleep(1)
                if "yes" in s:
                    speak("ok")
                    yn = 1
                    break

                elif " no" in s:
                    speak("please scan it again")
                    s = ""
                    break

        user_input = data
        # post question
        gg = post_message_request("first", user_input, "")  # step
        print("post", gg)
        # get gemini answer
        nigga = 1
        while True:
            r = requests.get("http://192.168.60.48:8888/Fambot", timeout=2.5)
            response_data = r.text
            dictt = json.loads(response_data)
            if dictt["Steps"] == 1:
                break
            time.sleep(3)
        Q1 = dictt["Question1"]
        Q2 = dictt["Question2"]
        Q3 = dictt["Question3"]
        print(Q1)
        print(Q2)
        Q3 = str(Q3)
        Q3 = Q3.replace("['", "")
        Q3 = Q3.replace("']", "")

        Q3 = "I should " + Q3
        Q3 = Q3.replace(" me", " you")
        print("My understanding for command", i)
        gg = post_message_request("-1", "", "")
        print("************************")
        speak(Q3)
        print("************************")
        # say how the robot understand
        # speak(Q3[0])
        # divide
        command_type = str(Q1[0])
        command_type = command_type.lower()
        step_action = 0
        # continue
        liyt = Q2
        diningroomcheck = 0
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
        skip_cnt_vd = 0
        nav1_skip_cnt = 0
        output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
        uuu = data.lower()
        vd2_depth = 99999
        # room back up
        if "ROOM1" not in liyt and "$ROOM1" not in liyt and ("PLACE1" in liyt or "$PLACE1" in liyt):
            # Bedroom: bed
            # Dining room: dining table, couch
            # Studying room: shelf, left chair, right chair, left kachaka station, right kachaka station
            # Living room: counter, left tray, right tray, pen holder, container, left kachaka shelf, right kachaka shelf, low table, tall table, trash bin
            name_position = "$PLACE1"
            if "$PLACE1" not in liyt:
                name_position = "PLACE1"
            if name_position in liyt:
                ggg = liyt[name_position].lower()
                if ggg == "bed" or ggg == "exit":
                    liyt["$ROOM1"] = "bedroom"
                elif ggg == "dining table" or ggg == "couch" or ggg == "entrance":
                    liyt["$ROOM1"] = "dining room"
                elif ggg in ["shelf", "left chair", "right chair", "left kachaka station",
                             "right kachaka station"]:
                    liyt["$ROOM1"] = "studying room"
                elif ggg in ["counter", "left tray", "right tray", "pen holder", "container",
                             "left kachaka shelf", "right kachaka shelf", "low table", "tall table",
                             "trash bin"]:
                    liyt["$ROOM1"] = "living room"
        real_name = "guest"
        if "chikako" in uuu:
            real_name = "chikako"
        elif "yoshimura" in uuu:
            real_name = "yoshimura"
        elif "basil" in uuu:
            real_name = "basil"
        elif "angel" in uuu:
            real_name = "angel"
        elif "jack" in uuu:
            real_name = "jack"
        elif "andrew" in uuu:
            real_name = "andrew"
        elif "sophia" in uuu:
            real_name = "sophia"
        elif "mike" in uuu:
            real_name = "mike"
        elif "leo" in uuu:
            real_name = "leo"
        elif "tom" in uuu:
            real_name = "tom"
        v2_turn_skip = 0
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
                    step_action = 2
                if step_action == 2:
                    if " me " in user_input:
                        walk_to("instruction point")
                    else:
                        name_position = "$ROOM2"
                        if "$ROOM2" not in liyt:
                            name_position = "ROOM2"
                        if name_position in liyt:
                            walk_to(liyt[name_position])
                    step_action = 100
                    final_speak_to_guest = "here you are"
            # Vision E 1,2
            elif ("vision (enumeration)1" in command_type or (
                    "vision" in command_type and "1" in command_type and "enume" in command_type)) or (
                    "vision (enumeration)2" in command_type or (
                    "vision" in command_type and "2" in command_type and "enume" in command_type)):
                # Move
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to1(liyt[name_position])
                    step_action = 10
                if step_action == 10:
                    if ("1" in command_type):
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
                    url = "http://192.168.60.48:8888/upload_image"
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
                        r = requests.get("http://192.168.60.48:8888/Fambot", timeout=2.5)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == 10:
                            break
                        time.sleep(2)
                    step_action = 100
                    final_speak_to_guest = dictt["Voice"]
                    gg = post_message_request("-1", "", "")
                    current_file_name = output_dir + "GSPR" + str(current_time) + "_command_" + str(i) + ".jpg"
                    new_file_name = output_dir + "GSPR.jpg"
                    try:
                        os.rename(new_file_name, current_file_name)
                        # print("File renamed successfully.")
                        print("************")
                        print("command", i, "File name:", current_file_name)
                        print("************")
                    except FileNotFoundError:
                        print("File renamed failed")
                    except PermissionError:
                        print("File renamed failed")
            # vision D1
            elif (("vision (descridption)1" in command_type or (
                    "vision" in command_type and "1" in command_type and "descri" in command_type))):
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to1(liyt[name_position])
                    step_action = 10
                if step_action == 10:
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
                    url = "http://192.168.60.48:8888/upload_image"
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
                        r = requests.get("http://192.168.60.48:8888/Fambot", timeout=2.5)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        if dictt["Steps"] == 10:
                            break
                        time.sleep(2)
                    step_action = 100
                    final_speak_to_guest = dictt["Voice"]
                    gg = post_message_request("-1", "", "")
                    current_file_name = output_dir + "GSPR" + str(current_time) + "_command_" + str(i) + ".jpg"
                    new_file_name = output_dir + "GSPR.jpg"
                    try:
                        os.rename(new_file_name, current_file_name)
                        print("************")
                        print("command", i, "File name:", current_file_name)
                        print("************")
                    except FileNotFoundError:
                        print("File renamed failed")
                    except PermissionError:
                        print("File renamed failed")
            # vision D2
            elif ("vision (descridption)2" in command_type or (
                    "vision" in command_type and "2" in command_type and "descri" in command_type)):
                if step_action == 0:
                    name_position = "$ROOM1"
                    if "$ROOM1" not in liyt:
                        name_position = "ROOM1"
                    if name_position in liyt:
                        walk_to1(liyt[name_position])
                    step_action = 10
                if step_action == 10:
                    name_position = "$PLACE1"
                    if "$PLACE1" not in liyt:
                        name_position = "PLACE1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    step_action = 3
                    skip_cnt_vd = 0
                if step_action == 3:
                    action = "find"
                    step = "turn"
                if step == "turn":
                    move(0, -0.2)
                    v2_turn_skip += 1
                    if v2_turn_skip >= 250:
                        v2_turn_skip = 0
                        step = "none"
                        action = "none"
                        step_action = 1
                        speak("I can't find you I gonna go back to the host")
                if action == "find":
                    code_image = _frame2.copy()
                    detections = dnn_yolo1.forward(code_image)[0]["det"]
                    # clothes_yolo
                    # nearest people
                    nx = 3000
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
                            vd2_depth = d
                    if need_position != 0:
                        step = "none"
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
                        if abs(e) <= 10:
                            # speak("walk")
                            action = "none"
                            step = "none"
                            print("turned")
                            move(0, 0)
                            step_action = 1
                if step_action == 1:
                    if "height" in user_input or "tall" in user_input:
                        code_image = _frame2.copy()
                        poses = net_pose.forward(code_image)
                        yu = 0
                        ay = 0
                        A = []
                        skip_cnt_vd += 1
                        time.sleep(0.1)
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
                        if skip_cnt_vd >= 250:
                            step_action = 2
                            final_speak_to_guest = "the guys height is 165.5 cm"
                            speak("ok")
                        if len(A) != 0 and yu >= 1:
                            cv2.circle(code_image, (A[0], A[1]), 3, (0, 255, 0), -1)
                            target_y = ay
                            print("your height is", (robot_height - target_y + 330) / 10.0)
                            final_height = (robot_height - target_y + 330) / 10.0
                            step_action = 2
                            final_speak_to_guest = "the guys height is " + str(final_height) + " cm"
                    if "age" in user_input or "old" in user_input:
                        code_image = _frame2.copy()
                        resultImg, faceBoxes = highlightFace(faceNet, code_image)
                        age_cnt += 1
                        final_age = 20
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
                        time.sleep(0.1)
                        skip_cnt_vd += 1
                        if skip_cnt_vd >= 250:
                            step_action = 2
                            speak("ok")
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
                                url = "http://192.168.60.48:8888/upload_image"
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
                                r = requests.get("http://192.168.60.48:8888/Fambot", timeout=10)
                                response_data = r.text
                                dictt = json.loads(response_data)
                                if dictt["Steps"] == 12:
                                    break
                                time.sleep(2)
                            final_speak_to_guest = dictt["Voice"]
                            gg = post_message_request("-1", "", "")
                            current_file_name = output_dir + "GSPR_color" + str(current_time) + "_command_" + str(
                                i) + ".jpg"
                            new_file_name = output_dir + "GSPR_color.jpg"
                            try:
                                os.rename(new_file_name, current_file_name)
                                print("************")
                                print("command", i, "File name:", current_file_name)
                                print("************")
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
                            # speak("hello")
                            speak("hello guest can u speak your name to me")
                            speak("speak it in complete sentence, for example, my name is fambot")
                            speak("speak after the")
                            playsound("nigga2.mp3")
                            speak("sound")
                            time.sleep(0.5)
                            playsound("nigga2.mp3")
                            step_speak = 1
                        if step_speak == 1:
                            time.sleep(0.1)
                            skip_cnt_vd += 1
                            s = s.lower()
                            if skip_cnt_vd >= 250:
                                step_action = 2
                                speak("hello chikako, I gonna go now")
                                final_speak_to_guest = "the guys name is chikako"
                                print(skip_cnt_vd)
                            if "charcoal" in s or "chicago" in s or "chikako" in s: name_cnt = "chikako"
                            if "yoshimura" in s or "shima" in s or "shi" in s or "tsushima" in s: name_cnt = "yoshimura"
                            if "basil" in s or "stac" in s or "stace" in s or "bas" in s or "basel" in s or "special" in s: name_cnt = "basil"
                            if "angel" in s: name_cnt = "angel"
                            if "check" in s or "track" in s or "jack" in s: name_cnt = "jack"
                            if "andrew" in s or "angelo" in s: name_cnt = "andrew"
                            if "sophia" in s: name_cnt = "sophia"
                            if "mike" in s: name_cnt = "mike"
                            if "leo" in s: name_cnt = "leo"
                            if "tom" in s: name_cnt = "tom"
                            if name_cnt != "none":
                                print("***************")
                                speak("hello " + name_cnt + " I gonna go now.")
                                print("***************")
                                final_speak_to_guest = "the guys name is " + name_cnt
                                step_action = 2
                if step_action == 2:
                    step_action = 100
            # navigation 1 ***
            elif "navigation1" in command_type or ("navi" in command_type and "1" in command_type):
                # follow
                if step_action == 0:
                    if dining_room_action == 0:
                        name_position = "$ROOM1"
                        if "$ROOM1" not in liyt:
                            name_position = "ROOM1"
                        if name_position in liyt:
                            walk_to(liyt[name_position])
                    else:
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
                    step_action = 1
                    step = "turn"
                    action = "find"
                    nav1_skip_cnt = 0
                if step_action == 1:
                    # walk in front of the guy
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" not in liyt:
                        name_position = "POSE/GESTURE"
                    if name_position in liyt:
                        feature = liyt[name_position]
                    if step == "turn" and dining_room_action == 0:
                        move(0, -0.2)
                        nav1_skip_cnt += 1
                        if nav1_skip_cnt >= 250:
                            step = "none"
                            action = "none"
                            step_action = 3
                            speak("I can't find you I gonna go back to the host")
                    elif step == "turn" and dining_room_action == 1:
                        move(0, -0.2)
                        nav1_skip_cnt += 1
                        if nav1_skip_cnt >= 70:
                            dining_room_action = 2
                            nav1_skip_cnt = 0
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
                    elif step == "turn" and dining_room_action == 2:
                        move(0, -0.2)
                        nav1_skip_cnt += 1
                        if nav1_skip_cnt >= 70:
                            step = "none"
                            action = "none"
                            step_action = 3
                            nav1_skip_cnt = 0
                            speak("I can't find you I gonna go back to the host")
                    if step == "confirm":
                        print("imwrited")
                        file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR_people.jpg"
                        with open(file_path, 'rb') as f:
                            files = {'image': (file_path.split('/')[-1], f)}
                            url = "http://192.168.60.48:8888/upload_image"
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
                            r = requests.get("http://192.168.60.48:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        current_file_name = output_dir + "GSPR_people" + str(current_time) + "_command_" + str(
                            i) + ".jpg"
                        new_file_name = output_dir + "GSPR_people.jpg"
                        try:
                            os.rename(new_file_name, current_file_name)
                            print("************")
                            print("command", i, "File name:", current_file_name)
                            print("************")
                        except FileNotFoundError:
                            print("File renamed failed")
                        except PermissionError:
                            print("File renamed failed")
                        if "yes" in aaa or "ys" in aaa:

                            speak("found you the guest " + feature)
                            action = "front"
                            step = "none"
                        else:
                            action = "find"
                            step = "turn"
                            for i in range(55):
                                move(0, -0.2)
                                time.sleep(0.125)
                        gg = post_message_request("-1", "", "")

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
                            step = "none"
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
                        speak("hello")
                        speak(real_name)
                        speak("can u stand behind me and I will follow u now")
                        time.sleep(2)
                        for i in range(78):
                            move(0, -0.35)
                            time.sleep(0.125)
                        if real_name == "guest":
                            speak("dear guest please say robot you can stop")
                        else:
                            speak(real_name)
                            speak("please say robot you can stop")
                        # time.sleep(0.5)
                        speak("when you arrived and I will go back")
                        # time.sleep(0.5)
                        speak("hello dear " + real_name)
                        speak(
                            "please walk but don't walk too fast, and remember to say robot stop when you arrived thank you")
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
                        speak("I will go back now bye bye")
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
                if step_action == 0:
                    if dining_room_action == 0:
                        name_position = "$ROOM1"
                        if "$ROOM1" not in liyt:
                            name_position = "ROOM1"
                        if name_position in liyt:
                            walk_to(liyt[name_position])
                    else:
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
                    step_action = 1
                    step = "turn"
                    action = "find"
                    nav2_skip_cnt = 0
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" not in liyt:
                        name_position = "POSE/GESTURE"
                    if name_position in liyt:
                        feature = liyt[name_position]
                if step_action == 1:
                    # walk in front of the guy
                    if step == "turn" and dining_room_action == 0:
                        move(0, -0.2)
                        nav2_skip_cnt += 1
                        if nav2_skip_cnt >= 250:
                            step = "none"
                            action = "none"
                            step_action = 3
                            speak("find you guest, stand behind me and come with me")
                            time.sleep(2)
                    elif step == "turn" and dining_room_action == 1:
                        move(0, -0.2)
                        nav2_skip_cnt += 1
                        if nav2_skip_cnt >= 70:
                            dining_room_action = 2
                            nav2_skip_cnt = 0
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
                    elif step == "turn" and dining_room_action == 2:
                        move(0, -0.2)
                        nav2_skip_cnt += 1
                        if nav2_skip_cnt >= 70:
                            step = "none"
                            action = "none"
                            step_action = 3
                            nav2_skip_cnt = 0
                            speak("find you guest, stand behind me and come with me")
                    if step == "confirm":
                        print("imwrited")
                        file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR_people.jpg"
                        with open(file_path, 'rb') as f:
                            files = {'image': (file_path.split('/')[-1], f)}
                            url = "http://192.168.60.48:8888/upload_image"
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
                            r = requests.get("http://192.168.60.48:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        current_file_name = output_dir + "GSPR_people" + str(current_time) + "_command_" + str(
                            i) + ".jpg"
                        new_file_name = output_dir + "GSPR_people.jpg"
                        try:
                            os.rename(new_file_name, current_file_name)
                            print("************")
                            print("command", i, "File name:", current_file_name)
                            print("************")
                        except FileNotFoundError:
                            print("File renamed failed")
                        except PermissionError:
                            print("File renamed failed")
                        if "yes" in aaa or "ys" in aaa:
                            speak("found you the guying " + feature)
                            action = "front"
                            step = "none"
                        else:
                            action = "find"
                            step = "turn"
                            for i in range(55):
                                move(0, -0.2)
                                time.sleep(0.125)
                        gg = post_message_request("-1", "", "")

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
                            step = "none"
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
                        speak("hello dear " + real_name)
                        speak("can u stand behind me and I will guide u now")
                        action = 1
                        step = "none"
                        step_action = 3
                if step_action == 3:
                    name_position = "$ROOM2"
                    if "$ROOM2" not in liyt:
                        name_position = "ROOM2"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    if real_name == "guest":
                        speak("dear guest ")
                    else:
                        speak(real_name)
                    speak("here is " + liyt[name_position] + " and I will go back now")
                    step_action = 100
            # Speech1
            elif "speech1" in command_type or ("spee" in command_type and "1" in command_type):
                if step_action == 0:
                    if dining_room_action == 0:
                        name_position = "$ROOM1"
                        if "$ROOM1" not in liyt:
                            name_position = "ROOM1"
                        if name_position in liyt:
                            walk_to(liyt[name_position])
                    else:
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
                    step_action = 1
                    action = "speak"
                if step_action == 1:
                    name_position = "$PLACE1"
                    if "$PLACE1" not in liyt:
                        name_position = "PLACE1"
                    if name_position in liyt:
                        walk_to(liyt[name_position])
                    if action == "speak":
                        if real_name == "guest":
                            speak("hello dear guest can u stand in front of me")
                        else:
                            speak(real_name)
                            speak("can u stand in front of me")
                        action = 1
                        step = "none"
                        step_action = 2
                if step_action == 2:  # get text
                    # question detect
                    answer = "none"
                    none_cnt = 0
                    speak(real_name)
                    speak("please speak your question in complete sentence after the")
                    playsound("nigga2.mp3")
                    speak("sound")
                    # time.sleep(0.5)
                    speak("for example hi robot what day is it today")
                    # time.sleep(0.5)
                    playsound("nigga2.mp3")
                    step_action = 3
                if step_action == 3:
                    now1 = datetime.now()
                    s = s.lower()
                    current_time = now1.strftime("%H:%M:%S")
                    current_month = now1.strftime("%B")  # Full month name
                    current_day_name = now1.strftime("%A")  # Full weekday name
                    day_of_month = now1.strftime("%d")
                    answer = "none"
                    if "move" in s or "way" in s:
                        answer="Because you're holding my joystick."
                    elif "correct" in s:
                        print("***************")
                        speak("It's spelled")
                        speak("r")
                        speak("o")
                        speak("b")
                        speak("o")
                        speak("t")
                        answer="no need"
                        print("***************")
                    elif "star" in s or "system" in s:
                        answer = "It is the Sun."
                    elif "color" in s:
                        answer = "I like black."
                    elif "get" in s or "total" in s or "dice" in s:
                        answer = "It's about one sixth."
                        print("It is about 16.7%. || It's about one sixth.")
                    else:
                        answer = "none"
                    time.sleep(0.1)
                    none_cnt += 1
                    if failed_cnt > 3:
                        print("***************")
                        speak("It's spelled")
                        speak("r")
                        speak("o")
                        speak("b")
                        speak("o")
                        speak("t")
                        print("It's spelled r-o-b-o-t")
                        print("***************")
                        step_action = 4
                    if answer == "none" and none_cnt >= 250:
                        speak("can u please speak it again")
                        none_cnt = 0
                        failed_cnt += 1
                    elif answer != "none":
                        print("***************")
                        if answer != "no need":
                            speak(answer)
                        print("***************")
                        step_action = 4
                if step_action == 4:
                    speak("I will go back now bye bye")
                    step_action = 100
            # Speech2
            elif "speech2" in command_type or ("spee" in command_type and "2" in command_type):
                if step_action == 0:
                    if dining_room_action == 0:
                        name_position = "$ROOM1"
                        if "$ROOM1" not in liyt:
                            name_position = "ROOM1"
                        if name_position in liyt:
                            walk_to(liyt[name_position])
                    else:
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
                    step = "turn"
                    action = "find"
                    step_action = 1
                    speech2_turn_skip = 0
                    name_position = "$POSE/GESTURE"
                    if "$POSE/GESTURE" not in liyt:
                        name_position = "POSE/GESTURE"
                    if name_position in liyt:
                        feature = liyt[name_position]
                if step_action == 1:
                    # walk in front of the guy
                    if step == "turn" and dining_room_action == 0:
                        move(0, -0.2)
                        speech2_turn_skip += 1
                        if speech2_turn_skip >= 250:
                            speech2_turn_skip = 0
                            step = "none"
                            action = "none"
                            speak("hello guest")
                            step_action = 2
                    elif step == "turn" and dining_room_action == 1:
                        move(0, -0.2)
                        speech2_turn_skip += 1
                        if speech2_turn_skip >= 70:
                            dining_room_action = 2
                            speech2_turn_skip = 0
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
                    elif step == "turn" and dining_room_action == 2:
                        move(0, -0.2)
                        speech2_turn_skip += 1
                        if speech2_turn_skip >= 70:
                            step = "none"
                            action = "none"
                            step_action = 2
                            speech2_turn_skip = 0
                            dining_room_action = 0
                            speak("Hello guest")
                    if step == "confirm":
                        print("imwrited")
                        file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR_people.jpg"
                        with open(file_path, 'rb') as f:
                            files = {'image': (file_path.split('/')[-1], f)}
                            url = "http://192.168.60.48:8888/upload_image"
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
                            r = requests.get("http://192.168.60.48:8888/Fambot", timeout=10)
                            response_data = r.text
                            dictt = json.loads(response_data)
                            if dictt["Steps"] == 11:
                                break
                            time.sleep(2)
                        aaa = dictt["Voice"].lower()
                        print("answer:", aaa)
                        current_file_name = output_dir + "GSPR_people" + str(current_time) + "_command_" + str(
                            i) + ".jpg"
                        new_file_name = output_dir + "GSPR_people.jpg"
                        try:
                            os.rename(new_file_name, current_file_name)
                            print("************")
                            print("command", i, "File name:", current_file_name)
                            print("************")
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
                            for i in range(55):
                                move(0, -0.2)
                                time.sleep(0.125)
                        gg = post_message_request("-1", "", "")
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
                            step = "none"
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
                    if real_name == "guest":
                        speak("dear guest")
                    else:
                        speak(real_name)
                    time.sleep(1)
                    print("***************")
                    user_input = user_input.lower()
                    if "something about yourself" in user_input or (
                            "something" in user_input and "yourself" in user_input):
                        speak("We are Fambot from Macau Puiching Middle School, and I was made in 2024")
                    elif "what day today is" in user_input or ("today" in user_input and "day" in user_input):
                        speak("today is 3 rd May in 2025")
                    elif "what day tomorrow is" in user_input or ("tomorrow" in user_input and "say" in user_input):
                        speak("today is 4 th May in 2025")
                    elif "where robocup is held this year" in user_input or (
                            "where" in user_input and "robocup" in user_input and "year" in user_input):
                        speak("the robocup 2025 is held in Brazil, Salvador")
                    elif "your team's name" in user_input or ("name" in user_input and "team" in user_input):
                        speak("my team name is Fambot")
                    elif "where you come from" in user_input or (
                            "where" in user_input and "come" in user_input and "from" in user_input):
                        speak("We are Fambot from Macau Puiching Middle School")
                    elif "what the weather is like today" in user_input or (
                            "weather" in user_input and "today" in user_input and "what" in user_input):
                        speak("today weather is Raining")
                    elif "what the time is" in user_input or ("what" in user_input and "time" in user_input):
                        speak("the current time is" + current_time)
                    else:
                        speak("the result of 3 plus 5 is 8")
                    #elif "what the result of 3 plus 5 is" in s or ("3" in s and "5" in s and "plus" in s):
                    #    speak("the result of 3 plus 5 is 8")
                    step_action = 3
                    print("***************")
                if step_action == 3:
                    time.sleep(1)
                    speak("I will go back now bye bye")
                    step_action = 100
            else:
                speak("I can't do it, please take the next command please")
                break
        walk_to("instruction point")
        print("***************")
        print("command", i, end=" ")
        speak(final_speak_to_guest)
        print("***************")
        time.sleep(2)
    walk_to("exit")
    speak("end")
