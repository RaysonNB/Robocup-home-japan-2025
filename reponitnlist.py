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
    api_url = "http://172.20.10.5:8888/Fambot"
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
    step = "fp"

    confirm_command = 0
    for i in range(2):
        check_cnt=0
        walk_to("instruction point")
        while not rospy.is_shutdown():
            if step=="fp":
                code_image = _frame2.copy()
                detections = dnn_yolo1.forward(code_image)[0]["det"]
                for i, detection in enumerate(detections):
                    # print(detection)
                    x1, y1, x2, y2, _, class_id = map(int, detection)
                    score = detection[4]
                    cx = (x2 - x1) // 2 + x1
                    cy = (y2 - y1) // 2 + y1
                    # depth=find_depsth
                    _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                    if score > 0.65 and class_id == 0 and d!=0 and d<=1200:
                        check_cnt+=1
                if check_cnt>=5:
                    step="name"
            if step=="name":
                #name, favorite drink, and a interest
                name_cnt="none"
                s = s.lower()
                speak("hello what is your name")
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
                if name_cnt !="none":
                    step="drink"
            if step=="drink":
                name_cnt = "none"
                s = s.lower()
                speak("hello what is your favourite drink")
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
                    step="interest"
            if step=="interest":
                name_cnt = "none"
                s = s.lower()
                speak("hello what is your interest")
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
                    step = "drinktable"
            if step == "drinktable":
                walk_to("drinktable")
                output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                face_box = [x1, y1, x2, y2]
                box_roi = _frame2[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                fh, fw = abs(x1 - x2), abs(y1 - y2)
                cv2.imwrite(4output_dir + "checkdrink.jpg", box_roi)
                file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/checkdrink.jpg"
                with open(file_path, 'rb') as f:
                    files = {'image': (file_path.split('/')[-1], f)}
                    url = "http://172.20.10.5:8888/upload_image"
                    response = requests.post(url, files=files)
                    # remember to add the text question on the computer code
                print("Upload Status Code:", response.status_code)
                upload_result = response.json()
                print("sent image")
                who_help = "Is the guy " + feature
                gg = post_message_request("checkdrink", feature, who_help)
                print(gg)
                # get answer from gemini
                while True:
                    r = requests.get("http://172.20.10.5:8888/Fambot", timeout=10)
                    response_data = r.text
                    dictt = json.loads(response_data)
                    if dictt["Steps"] == 11:
                        break
                    time.sleep(2)
                aaa = dictt["Voice"].lower()
                print("answer:", aaa)
                speak(aaa)
                step="walk"
            if step=="walk":
                #
