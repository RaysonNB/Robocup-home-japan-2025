import google.generativeai as genai
import json
import time
import requests
import google.generativeai as genai
import os
import PIL.Image
import cv2
import numpy as np
from datetime import datetime

pathnum = r"C:/Users/rayso/Desktop/python/"
from Generate_command import kitchen_items

genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')
model = genai.GenerativeModel("gemini-2.0-flash")
cnt_yy = 0
while True:

    while True:
        r = requests.get("http://192.168.60.20:8888/Fambot", timeout=2.5)
        response_data = r.text
        print("Response_data", response_data)
        dictt = json.loads(response_data)
        if dictt["Steps"] == "-1" or dictt["Voice"] == "":
            time.sleep(2)
        else:
            break
    if dictt["Steps"] == "guest":
        image_url = f"http://192.168.60.20:8888{'/uploads/guest1.jpg'}"
        print("Fetching image from:", image_url)
        image_response = requests.get(image_url)
        image_array = np.frombuffer(image_response.content, dtype=np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Save the image using OpenCV
        cv2.imwrite(pathnum + "guest1.jpg", img)
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": "-1",
            "Voice": "",
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.60.20:8888/Fambot"
        response = requests.post(api_url, json=questions)
    if dictt["Steps"] == "feature":
        path_sample = "C:/Users/rayso/Desktop/python/guest1.jpg"  # Use raw string to handle backslashes
        # Prepare the prompt template
        sample_txt = ""
        img = PIL.Image.open(path_sample)
        promt = '''
        question1: how old is the guy, give me a range
        question2: he is male or female?
        question3: what color of colthes he is wearing
        
        answer the question in complete sentence
        entire_answer = question1 answer + question2 answer + question3 answer
        just need one sentence
        answer format: ******[entire_answer]******)
        '''
        response = model.generate_content([img, promt])
        # response = model.generate_content([img2,"where have empty seat , just give me number in [1,2,3,4,5], there should be 4 numbers, answer format: ******[numbers]******"])
        import re

        a = str(response)
        print(a)
        # Using a regular expression to find text between asterisks
        matches = re.findall(r'\*\*\*\*\*\*(.*?)\*\*\*\*\*\*', a)

        # Printing the extracted values
        answergg=""
        for match in matches:
            g = match.strip()
            answergg=g
            print(g)  # Output: [1, 2, 4, 5]
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": 100,
            "Voice": answergg,
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.60.20:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
    if dictt["Steps"] == "task3":
        promt = dictt["Questionasking"]
        image_url = f"http://192.168.60.20:8888{'/uploads/task3.jpg'}"
        print("Fetching image from:", image_url)
        image_response = requests.get(image_url)
        image_array = np.frombuffer(image_response.content, dtype=np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Save the image using OpenCV
        cv2.imwrite(pathnum + "task3.jpg", img)
        print("Image saved successfully using OpenCV!")
        # Configuration and setup
        genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')  # Replace with your actual API key
        model = genai.GenerativeModel("gemini-2.0-flash")
        path_sample = "C:/Users/rayso/Desktop/python/task3.jpg"  # Use raw string to handle backslashes
        # Prepare the prompt template
        sample_txt = promt
        img = PIL.Image.open(path_sample)
        response = model.generate_content([img, sample_txt])
        file_data_string = response.text
        print(file_data_string)
        file_data_string = file_data_string.replace("**", "")
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": 101,
            "Voice": file_data_string,
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.60.20:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
    if dictt["Steps"] == "seat1":
        promt = dictt["Questionasking"]
        image_url = f"http://192.168.60.20:8888{'/uploads/emptyseat.jpg'}"
        print("Fetching image from:", image_url)
        image_response = requests.get(image_url)
        image_array = np.frombuffer(image_response.content, dtype=np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Save the image using OpenCV
        cv2.imwrite(pathnum + "emptyseat.jpg", img)
        print("Image saved successfully using OpenCV!")
        # Configuration and setup
        genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')  # Replace with your actual API key
        model = genai.GenerativeModel("gemini-2.0-flash")
        path_sample = "C:/Users/rayso/Desktop/python/task3.jpg"  # Use raw string to handle backslashes
        # Prepare the prompt template
        sample_txt = promt
        img = PIL.Image.open(path_sample)
        response = model.generate_content([img, sample_txt])
        file_data_string = response.text
        print(file_data_string)
        file_data_string = file_data_string.replace("**", "")
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": 101,
            "Voice": file_data_string,
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.60.20:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
