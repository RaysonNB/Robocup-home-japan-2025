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
    api_url = "http://192.168.60.20:8888/Fambot"
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
def turn_to(angle: float, speed: float):
    global _imu
    max_speed = 0.05
    limit_time = 5
    start_time = rospy.get_time()
    while True:
        q = [
            _imu.orientation.x,
            _imu.orientation.z,
            _imu.orientation.y,
            _imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        e = angle - yaw
        print(yaw, e)
        if yaw < 0 and angle > 0:
            cw = np.pi + yaw + np.pi - angle
            aw = -yaw + angle
            if cw < aw:
                e = -cw
        elif yaw > 0 and angle < 0:
            cw = yaw - angle
            aw = np.pi - yaw + np.pi + angle
            if aw < cw:
                e = aw
        if abs(e) < 0.01 or rospy.get_time() - start_time > limit_time:
            break
        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()
    move(0.0, 0.0)
def turn(angle: float):
    global _imu
    q = [
        _imu.orientation.x,
        _imu.orientation.y,
        _imu.orientation.z,
        _imu.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(q)
    target = yaw + angle
    if target > np.pi:
        target = target - np.pi * 2
    elif target < -np.pi:
        target = target + np.pi * 2
    turn_to(target, 0.1)
locations = {
    # Furniture and objects
    "guest": [3.154, 2.870, 1.53],
    "seats": [3.350, 3.111, -1.5],
    "drinktable": [2.507, 3.287, -1.607],
}
def hand_turn_left():
    say("robot arm turn left")
def seat_turn(num12):
    check_num=str(num12)
    angle1 = -75
    angle2 = -30
    angle3 = 0
    angle4 = 30
    angle5 = 75
    if "1" in check_num:
        turn(angle1)
    elif "2" in check_num:
        turn(angle2)
    elif "3" in check_num:
        turn(angle3)
    elif "4" in check_num:
        turn(angle4)
    elif "5" in check_num:
        turn(angle5)
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
    gg = post_message_request("-1", "", "")
    step = "fp"
    pre_s=""
    confirm_command = 0
    for nigga_i in [1,2]:
        check_cnt=0
        walk_to("guest")
        time.sleep(1)
        if nigga_i == 1:
            speak("hello dear guest, can u stand one meter in front of me, i will take you a picture")
            speak("please walk backward until the camera can see your face and shoulder")
        else:
            speak("hello dear guest, can u stand in front of me")
        speak("My name is Fambot, please answer my following questions with louder voice, thank you")
        if nigga_i == 2:
            step="name"
        while not rospy.is_shutdown():
            now1 = datetime.now()
            current_time = now1.strftime("%H:%M:%S")
            rospy.Rate(10).sleep()
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
            cv2.imshow("frame", code_image)
            if step=="fp":
                robot_height=1300
                code_image = _frame2.copy()
                poses = net_pose.forward(code_image)
                yu = 0
                ay = 0
                A = []
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
                                if az <= 1500 and az != 0:
                                    yu += 1
                        if yu >= 1:
                            break
                if len(A) != 0 and yu >= 1:
                    cv2.circle(code_image, (A[0], A[1]), 3, (0, 255, 0), -1)
                    target_y = ay
                    print("your height is", (robot_height - target_y + 330) / 10.0)
                    final_height = (robot_height - target_y + 330) / 10.0
                    step_action = 2
                    final_speak_to_guest = "the guest height is " + str(final_height) + " cm.  "
                    step="fp1"
            if step == "fp1":
                code_image = _frame2.copy()
                mx1, my1, mx2, my2 = 0,0,0,0
                detections = dnn_yolo1.forward(_frame2)[0]["det"]
                yn=0
                for i, detection in enumerate(detections):
                    # print(detection)
                    x1, y1, x2, y2, _, class_id = map(int, detection)
                    score = detection[4]
                    cx = (x2 - x1) // 2 + x1
                    cy = (y2 - y1) // 2 + y1
                    _, _, d = get_real_xyz(code_depth, cx, cy, 2)
                    if score > 0.65 and class_id == 0 and d!=0 and 1000<=d<=1500:
                        check_cnt+=1
                        mx1, my1, mx2, my2 = x1, y1, x2, y2
                        yn=1
                if yn == 1:
                    face_box = [mx1, my1, mx2, my2]
                    box_roi = _frame2[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                file_name="guest"+str(nigga_i)

                promt_guest_feature= '''
                question1: how old is the guy, give me a range
                question2: he is male or female?
                question3: what color of colthes he is wearing
                
                answer the question in complete sentence
                entire_answer = question1 answer + question2 answer + question3 answer
                just need one sentence
                answer format: ******[entire_answer]******)
                '''
                #male, color, height, old
                cv2.imwrite(output_dir + file_name, box_roi)
                file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/guest1.jpg"
                with open(file_path, 'rb') as f:
                    files = {'image': (file_path.split('/')[-1], f)}
                    url = "http://192.168.60.20:8888/upload_image"
                    response = requests.post(url, files=files)
                    # remember to add the text question on the computer code
                print("Upload Status Code:", response.status_code)
                upload_result = response.json()
                print("sent image")
                gg = post_message_request("guest1", feature, "")
                step="name"
            if step=="name":
                #name, favorite drink, and a interest
                name_cnt="none"
                s = s.lower()
                speak("hello dear guest what is your name")
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
                    name=name_cnt
            if step=="drink":
                name_cnt = "none"
                s = s.lower()
                speak("hello dear guest what is your favourite drink")
                if "coffee" in s: name_cnt = "Coffee"
                if "tea" in s: name_cnt = "Tea"
                if "orangle" in s: name_cnt = "Orangle Juice"
                if "apple" in s: name_cnt = "Apple Juice"
                if "Cola" in s: name_cnt = "Cola"
                if "chocolate" in s: name_cnt = "chocolate"
                if "water" in s: name_cnt = "water"
                if "wine" in s: name_cnt = "wine"
                if "beer" in s: name_cnt = "beer"
                if name_cnt != "none":
                    drink_name=name_cnt
                    step="interest"
            if step=="interest":
                interest_name = "none"
                s = s.lower()
                speak("hello dear guest what is your interest")
                if "art" in s: interest_name = "art"
                if "act" in s: interest_name = "act"
                if "animal" in s: interest_name = "animals"
                if "basketball" in s: interest_name = "basketball"
                if "football" in s: interest_name = "football"
                if "eat" in s: interest_name = "eat"
                if interest_name != "none":
                    step = "drinktable"
                    say("your name is " + name + " your favourite drink is " + drink_name + " your interest is "+interest_name)
            if step == "drinktable":
                walk_to("drinktable")
                step="gemini_drinks"
            if step=="gemini_drinks":
                output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                box_roi = _frame2.copy()
                cv2.imwrite(output_dir + "checkdrink.jpg", box_roi)
                file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/checkdrink.jpg"
                with open(file_path, 'rb') as f:
                    files = {'image': (file_path.split('/')[-1], f)}
                    url = "http://192.168.60.20:8888/upload_image"
                    response = requests.post(url, files=files)
                    # remember to add the text question on the computer code
                print("Upload Status Code:", response.status_code)
                upload_result = response.json()
                print("sent image")
                who_help = '''
                now, you are going to find my friends favourite drink on this table(the image).
                
                tell me where is my friends favourite drink with definite position on the table.
                
                you may answer your favourite drink isn't on the table or your favourite drink is in the ... position of the table
                
                answer format: ******[your speech]******
                
                '''
                favhh="my friends favourite drink is "+ drink_name
                gg = post_message_request("task3", feature, who_help+favhh)
                print(gg)
                step="waitdrink"
                # get answer from gemini
            if step=="waitdrink":
                r = requests.get("http://192.168.60.20:8888/Fambot", timeout=10)
                response_data = r.text
                dictt = json.loads(response_data)
                time.sleep(2)
                if dictt["Steps"] == 101:
                    gg = post_message_request("-1", "", "")
                    aaa = dictt["Voice"].lower()
                    print("answer:", aaa)
                    speak(aaa)
                    time.sleep(1)
                    speaking_text="dear guest, please follow me to the seat"
                    say(speaking_text)
                    step="walk"
            if step=="walk":
                walk_to("seats")
                speak("dear guest, can u stand on my left left left side, thank you")
                time.sleep(1)
                check_empty_img=_frame2.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_color = (255, 255, 255)
                font_thickness = 5
                # correct the numbers**********************
                positions = {
                    1: (int(width * 0.08), int(height * 0.4)),
                    2: (int(width * 0.35), int(height * 0.4)),
                    3: (int(width * 0.5), int(height * 0.4)),
                    4: (int(width * 0.68), int(height * 0.4)),
                    5: (int(width * 0.85), int(height * 0.4))
                }
                for number, pos in positions.items():
                    text_size = cv2.getTextSize(str(number), font, font_scale, font_thickness)[0]
                    rectangle_start = (pos[0] - 10, pos[1] + 10)  # Adjust margins
                    rectangle_end = (pos[0] + text_size[0] + 10, pos[1] - text_size[1] - 10)
                    cv2.rectangle(check_empty_img, rectangle_start, rectangle_end, (0, 0, 0),
                                  cv2.FILLED)
                    cv2.putText(check_empty_img, str(number), pos, font, font_scale, font_color, font_thickness,
                                cv2.LINE_AA)
                output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
                cv2.imwrite(output_dir + "emptyseat.jpg", check_empty_img)
                step="askempty"
            if step=="askempty":
                file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/emptyseat.jpg"
                with open(file_path, 'rb') as f:
                    files = {'image': (file_path.split('/')[-1], f)}
                    url = "http://192.168.60.20:8888/upload_image"
                    response = requests.post(url, files=files)
                    # remember to add the text question on the computer code
                print("Upload Status Code:", response.status_code)
                upload_result = response.json()
                print("sent image")
                if  nigga_i == 1:
                    number=4
                    who_help = "where have empty seat, just give me number in [1,2,3,4,5], there should be " +str(number)+ " numbers, answer format: ******[numbers]******" #correct the numbers**********************
                    gg = post_message_request("seat1", feature, who_help)
                    print(gg)
                elif nigga_i==2:
                    gg = post_message_request("seat2", feature, "")
                    print(gg)
                step = "waitempty"
            if step=="waitempty":
                r = requests.get("http://192.168.60.20:8888/Fambot", timeout=10)
                response_data = r.text
                dictt = json.loads(response_data)
                time.sleep(2)
                if dictt["Steps"] == 101:
                    gg = post_message_request("-1", "", "")
                    aaa = dictt["Voice"].lower()
                    if nigga_i == 1:
                        host_seat = "0"
                        if "1" not in str(gg): host_seat = "1"
                        if "2" not in str(gg): host_seat = "2"
                        if "3" not in str(gg): host_seat = "3"
                        if "4" not in str(gg): host_seat = "4"
                        if "5" not in str(gg): host_seat = "5"
                        print("answer:", aaa)
                    else:
                        check=""
                        if "1" not in str(gg): check+="1"
                        if "2" not in str(gg): check+="2"
                        if "3" not in str(gg): check+="3"
                        if "4" not in str(gg): check+="4"
                        if "5" not in str(gg): check+="5"
                        guest1_seat=aaa #get by gemini
                        hosts_seat=check.replace(guest1_seat,"")
                    #speak(aaa)
                    confirm_seat=aaa
                    step = "tell_hosts"
            if step=="tell_hosts":
                seat_turn(host_seat)
                hand_turn_left()
                host_name,host_drink_name,host_interest_name="john","milk","basketball"
                if nigga_i == 1:
                    speak("dear "+host_name+" this is the first guest "+name+" favourite drink is "+drink_name+" interest is "+interest_name)
                    turn(-90) # the chassis left
                    speak("dear "+name+" this is the host "+host_name+" favourite drink is "+host_drink_name+" interest is "+host_interest_name)
                else:
                    speak("dear " + host_name + " this is the second guest " + name + " favourite drink is " + drink_name + " interest is " + interest_name)
                    turn(-90)  # the chassis left
                    speak("dear " + name + " this is the host" + host_name + " favourite drink is " + host_drink_name + " interest is " + host_interest_name)
                step="tell1"
            if step=="tell1":
                seat_turn(guest1_seat)
                if nigga_i==2:
                    say("dear "+pre_name +" this is the second guest " + name + " favourite drink is " + drink_name + " interest is " + interest_name)
                    turn(-90) #chassis left
                    gg = post_message_request("feature", feature, "")
                    while True:
                        r = requests.get("http://192.168.60.20:8888/Fambot", timeout=10)
                        response_data = r.text
                        dictt = json.loads(response_data)
                        time.sleep(2)
                        if dictt["Steps"] == 100:
                            gg = post_message_request("-1", "", "")
                            aaa = dictt["Voice"].lower()
                            speech_robot_guest2 = final_speak_to_guest + aaa
                            break
                    speak("dear " + name + " this is the first guest" + pre_name + " favourite drink is " + pre_drink + " interest is " + pre_interest)
                    speak(speech_robot_guest2)
                step="tell"
            if step=="tell":
                seat_turn(host_seat)
                say("dear guest "+name+" the way I am facing is a empty seat, please have a sit, thank you")
                pre_name,pre_drink,pre_interest=name,drink_name,interest_name
                step="fp"
        say("Receptionist end")
