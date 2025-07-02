import json
import time
import requests
import google.generativeai as genai
import PIL.Image
import cv2
import numpy as np
from Generate_command import kitchen_items

pathnum = r"C:/Users/rayso/Desktop/python/"
genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')
model = genai.GenerativeModel("gemini-2.0-flash")
cnt_yy = 0
while True:

    while True:
        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
        response_data = r.text
        print("Response_data", response_data)
        dictt = json.loads(response_data)
        if dictt["Steps"] == "-1":
            time.sleep(2)
        else:
            break
    if dictt["Steps"] == "guest1":
        image_url = f"http://192.168.50.147:8888{'/uploads/guest1.jpg'}"
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
        api_url = "http://192.168.50.147:8888/Fambot"
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
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
    if dictt["Steps"] == "task3":
        print("task3")
        promt = dictt["Questionasking"]
        image_url = f"http://192.168.50.147:8888{'/uploads/task3.jpg'}"
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
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
    if dictt["Steps"] == "seat1":
        promt = dictt["Questionasking"]
        image_url = f"http://192.168.50.147:8888{'/uploads/emptyseat.jpg'}"
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
        path_sample = "C:/Users/rayso/Desktop/python/emptyseat.jpg"  # Use raw string to handle backslashes
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
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
    if dictt["Steps"] == "seat2":
        promt = dictt["Questionasking"]
        image_url = f"http://192.168.50.147:8888{'/uploads/emptyseat.jpg'}"
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
        path_sample = "C:/Users/rayso/Desktop/python/emptyseat.jpg"  # Use raw string to handle backslashes
        # Prepare the prompt template
        sample_txt = promt
        img = PIL.Image.open(path_sample)
        img2 = PIL.Image.open("C:/Users/rayso/Desktop/python/guest1.jpg")
        response = model.generate_content([img2, "this is the first guest",img,"here have 5 seats please tell me where is he sitting, just give me number in [1,2,3,4,5], there should be 1 numbers, answer format: ******[...]******"])
        file_data_string = response.text
        print(file_data_string)
        time.sleep(1)
        file_data_string = file_data_string.replace("**", "")
        response = model.generate_content([img,"here have 5 seats please tell me where have empty seat(chair), just give me number in [1,2,3,4,5], there should be 3 numbers, answer format: ******[numbers]******, for example ******[1,2,3]******"])
        file_data_string1 = response.text
        print(file_data_string)
        file_data_string1 = file_data_string1.replace("**", "")
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": 101,
            "Voice": file_data_string,
            "Questionasking": "None",
            "answer": file_data_string1
        }
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
