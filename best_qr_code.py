#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import cv2
import time, requests
import json
def callback_image2(msg):
    global _frame2
    _frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def callback_depth2(msg):
    global _depth2
    _depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
def speak(g):
    os.system(f'espeak "{g}"')
    # rospy.loginfo(g)
    print(g)
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
if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    print("gemini2 rgb")
    _frame2 = None
    _sub_down_cam_image = rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)
    print("gemini2 depth")
    _depth2 = None
    _sub_down_cam_depth = rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    for i in range(3):
        qr_code_detector = cv2.QRCodeDetector()
        data=0
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
        #continue
