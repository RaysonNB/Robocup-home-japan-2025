import requests
import cv2
import numpy as np

url = "http://192.168.50.147:8888/upload_image"
file_path = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/GSPR.jpg"

with open(file_path, 'rb') as f:
    files = {'image': (file_path.split('/')[-1], f)}
    response = requests.post(url, files=files)

print("Upload Status Code:", response.status_code)
upload_result = response.json()
print("Server Response:", upload_result)
print("link", upload_result['url'])
