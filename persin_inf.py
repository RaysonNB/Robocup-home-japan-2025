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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file')
parser.add_argument('--video', help='Path to video file', default=0)
args = parser.parse_args()


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
def speak(g):
    os.system(f'espeak "{g}"')
    # rospy.loginfo(g)
    print(g)

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
# gemini2
def callback_image2(msg):
    global _frame2
    _frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth2(msg):
    global _depth2
    _depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def callback_voice(msg):
    global s
    s = msg.text
    
if __name__ == "__main__":
    rospy.init_node("demo")
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)

    # Age categories
    ageList = ['1', '5', '10', '17', '27', '41', '50', '67']

    # Initialize video capture
    video = cv2.VideoCapture(args.video if args.video else 0)
    padding = 20
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
    #dnn_yolo1.classes = ['obj']
    # two yolo
    print("pose")
    net_pose = HumanPoseEstimation(device_name="GPU")
    print("chassis")
    
    ss="age height"
    final_height=0
    final_age=0
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
        if final_height !=0 and final_age!=0:
            speak(str(final_height)+" cm"+"  "+str(final_age)+" years old")
            break
        # var needs in while
        cx1, cx2, cy1, cy2 = 0, 0, 0, 0
        up_image = _frame2.copy()
        up_depth = _depth2.copy()
        
        s=s.lower()
        if "height" in ss:
            poses = net_pose.forward(up_image)
            if len(poses) > 0:
                YN = -1
                a_num = 5
                for issack in range(len(poses)):
                    yu = 0
                    if poses[issack][5][2] > 0:
                        YN = 0
                        a_num, b_num = 5,5
                        A = list(map(int, poses[issack][a_num][:2]))
                        if (640 >= A[0] >= 0 and 320 >= A[1] >= 0):
                            ax, ay, az = get_real_xyz(up_depth, A[0], A[1], 2)
                            print(ax,ay)
                            if az <= 2500 and az != 0:
                                yu += 1
                    if yu >= 1:
                        break
            if len(A) != 0 and yu >= 1:
                cv2.circle(up_image, (A[0], A[1]), 3, (0, 255, 0), -1)
                target_y = ay
            print("your height is",(1000-target_y+330)/10.0)
            final_height=(1000-target_y+330)/10.0
        if "age" in ss:
            frame=up_image.copy()
            resultImg, faceBoxes = highlightFace(faceNet, frame)

            if not faceBoxes:
                print("No face detected")
                #continue
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - padding):
                             min(faceBox[3] + padding, frame.shape[0] - 1),
                       max(0, faceBox[0] - padding):
                       min(faceBox[2] + padding, frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                print(age)
                final_age=age
                cv2.putText(resultImg, f'Age: {age}', (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        elif "color" in ss:
            
        elif "name" in ss:
            #jack, check, track
            #aaron, ellen, evan
            #angel
            #adam, ada, aiden
            #Vanessa, lisa, Felicia
            #chris
            #william
            #max, mix
            #hunter
            #olivia
            
            speak("hello nigga can u speak your name to me")
            speak("speak it in complete sentence, for example, my name is fambot")
            speak("speak after the")
            playsound("nigga2.mp3")
            speak("sound")
            time.sleep(0.5)
            playsound("nigga2.mp3")
            break
        frmaeshow=frame.copy()
        cv2.imshow("frame", resultImg)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break

