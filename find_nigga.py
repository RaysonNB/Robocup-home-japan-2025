#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
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
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Imu
from typing import Tuple, List
from RobotChassis import RobotChassis
import datetime
from tf.transformations import euler_from_quaternion
import os
import requests
import json


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


def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def get_pose_target2(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0:
        return -1, -1, -1
    return int(p[0][0]), int(p[0][1]), 1


def get_pose_target(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0:
        return -1, -1
    return int(p[0][0]), int(p[0][1])


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


# astrapro
def callback_image1(msg):
    global _image1
    _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


# gemini2
def callback_image2(msg):
    global _frame2
    _frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth2(msg):
    global _depth2
    _depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def callback_imu(msg):
    global _imu
    _imu = msg


def speak(g):
    os.system(f'espeak "{g}"')
    # rospy.loginfo(g)
    print(g)


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


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    print("gemini2 rgb")
    _frame2 = None
    _sub_down_cam_image = rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)
    print("gemini2 depth")
    _depth2 = None
    _sub_down_cam_depth = rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    s = ""
    print("cmd_vel")
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    print("arm")
    t = 3.0
    # change model
    print("yolov8")
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo1 = Yolov8("yolov8n", device_name="GPU")
    # dnn_yolo1.classes = ['obj']
    # two yolo
    print("pose")
    net_pose = HumanPoseEstimation(device_name="GPU")
    print("chassis")
    chassis = RobotChassis()
    _imu = None
    topic_imu = "/imu/data"
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)
    _fw = FollowMe()
    action = "walk"
    mode = 0
    find_people = "none"
    step = "none"
    pre_s = 0
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    speak("start the code")
    feature = "rising right hand"
    question_text = "go to the dining room and find the guy who is rasing his hand"
    pose = "rasing his hand"
    detection_list = []
    while not rospy.is_shutdown():
        # voice check
        # break
        if s != "" and s != pre_s:
            print(s)
            pre_s = s
        rospy.Rate(10).sleep()

        if _frame2 is None: print("down rgb none")
        if _depth2 is None: print("down depth none")

        if _depth2 is None or _frame2 is None: continue

        # var needs in while
        cx1, cx2, cy1, cy2 = 0, 0, 0, 0
        up_image = _frame2.copy()
        up_depth = _depth2.copy()
        # 到咗
        if action == "walk":
            # navigation
            # position=(-7.204, -6.026, -0.007)
            clear_costmaps
            chassis.move_to(-7.204, -6.026, -0.007)
            # checking
            while not rospy.is_shutdown():
                # 4. Get the chassis status.
                code = chassis.status_code
                text = chassis.status_text
                if code == 3:
                    break
            if mode == 0 and mode != 1:
                time.sleep(1)
                q = [
                    _imu.orientation.x,
                    _imu.orientation.y,
                    _imu.orientation.z,
                    _imu.orientation.w
                ]
                roll1, pitch1, yaw1 = euler_from_quaternion(q)
                mode = 1
            action = "find"
            step = "turn"
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
            detections = dnn_yolo1.forward(up_image)[0]["det"]
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
                _, _, d = get_real_xyz(up_depth, cx, cy, 2)
                # cv2.rectangle(up_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if score > 0.65 and class_id == 0 and d <= nx and d != 0 and (320 - cx) < CX_ER:
                    need_position = [x1, y1, x2, y2, cx, cy]
                    # ask gemini
                    cv2.rectangle(up_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(up_image, (cx, cy), 5, (0, 255, 0), -1)
                    print("people distance", d)
                    CX_ER = 320 - cx
            if need_position != 0:
                h, w, c = up_image.shape
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
                box_roi = up_image[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
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
            h, w, c = up_image.shape
            cx, cy = w // 2, h // 2
            for i in range(cy + 1, h):
                if _depth2[cy][cx] == 0 or 0 < _depth2[i][cx] < _depth2[cy][cx]:
                    cy = i
            _, _, d = get_real_xyz(_depth2, cx, cy, 2)
            print("depth", d)
            if d != 0 and d <= 500:
                action = "speak"
                move(0, 0)
            else:
                move(0.2, 0)
        if action == "speak":
            speak("hi nigga how can I help you")
            break
        cv2.imshow("frame", up_image)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
