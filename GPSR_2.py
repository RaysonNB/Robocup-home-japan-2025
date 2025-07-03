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
        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
        response_data = r.text
        print("Response_data", response_data)
        dictt = json.loads(response_data)
        if dictt["Steps"] == "-1" or dictt["Voice"] == "":
            time.sleep(2)
        else:
            break
    if dictt["Steps"] == "first":
        s1 = dictt["Voice"]
        s = "***The Sentence:" + s1
        print("question", s)
        sample_txt = """

        (The Sentence)(Task: Sentence Structure)(I given u)
        Manipulation1: Go to the $ROOM1, grasp the $OBJECT on the $PLACE1 and place it on the $PLACE2.
        Manipulation2: Go to the $ROOM1, grasp the $OBJECT on the $PLACE1 and give it to $PERSON on the $ROOM2.(if &PERSON is me than $ROOM2:"instruction point" just edit $ROOM2)
        Vision (Enumeration)1: Tell me how many $CATEGORY_OBJ here are on the $PLACE1.
        Vision (Enumeration)2: Tell me how many people in the $ROOM1 are $POSE/GESTURE.
        Vision (Description)1: Tell me what is the $OBJ_COMP object on the $PLACE1.
        Vision (Description)2: Tell me the $PERS_INFO of the person at the $PLACE1
        Navigation1: Go to the $ROOM1, find $POSE/GESTURE person and follow (him | her).
        Navigation2: Go to the $ROOM1, find $POSE/GESTURE person and guide (him|her) to the $ROOM2.
        Speech1: Go to the $ROOM1, find $PERSON at the $PLACE1 and answer (his | her) question.
        Speech2: Go to the $ROOM1, find the person who is $POSE/GESTURE and tell (him | her) $TELL_LIST.

        %possible information options
        %ROOM         : study room, living room, bed room, dining room
        %PLACE        : counter, left tray, right tray, pen holder, container, left kachaka shelf, right kachaka shelf, low table, tall table, trash bin, left chair, right chair, left kachaka station, right kachaka station, shelf, bed, dining table, couch, entrance, exit
        $OBJECT       : Noodles, Cookies, Potato Chips, Caramel Corn, Detergent, Cup, Lunch Box, Sponge, Dice, Light Bulb, Glue gun, Phone Stand
        $PERS_INFO    : name, shirt color, age, height
        $CATEGORY_OBJ : Food Item, Kitchen Item, Task Item


        (Questions)
        Question1: which Task is it(just one) [Manipulation1, Manipulation2, Vision (Enumeration)1, Vision (Enumeration)2, Vision (Description)1, Vision (Description)2, Navigation1, Navigation2, Speech1, Speech2] ?
        Question2: give me the $informations(make it in dictionary), for example {"$ROOM1":"Living room","$PLACE1":"Tray A"} ?
        Question3: what the sentence mean, and what I should do(20words)(just give me one sentence)?



        here is the eanswer_format (in python_dictronary_format)

        *** {"1":[],"2":[],"3":[]} ***
        """
        response = model.generate_content([s, sample_txt])
        file_data_string = response.text
        print(file_data_string)
        file_data_string = file_data_string.replace("```", "")
        file_data_string = file_data_string.replace("python", "")
        file_data_string = file_data_string.replace("***", "")
        file_data_string = file_data_string.replace("json", "")
        dict = json.loads(file_data_string)
        Question1 = dict["1"]
        s = str(dict["2"])
        if (s[0] == "["):
            Question2 = dict["2"][0]
        else:
            Question2 = dict["2"]
        Question3 = dict["3"]
        time.sleep(1)
        questions = {
            "Question1": Question1,
            "Question2": Question2,
            "Question3": Question3,
            "Steps": 1,
            "Voice": "Voice",
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        print("sent")
        time.sleep(2)
    elif dictt["Steps"] == "checkpeople":
        promt = dictt["Questionasking"] + " answer my question ys or no only"
        image_url = f"http://192.168.50.147:8888{'/uploads/GSPR_people.jpg'}"
        print("Fetching image from:", image_url)
        image_response = requests.get(image_url)

        image_array = np.frombuffer(image_response.content, dtype=np.uint8)

        # Decode the image using OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        name_img = 'Robot_view' + str(cnt_yy) + '.jpg'
        print(name_img)
        # Save the image using OpenCV

        cv2.imwrite(pathnum + name_img, img)
        print("Image saved successfully using OpenCV!")
        # Configuration and setup
        genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')  # Replace with your actual API key
        model = genai.GenerativeModel("gemini-2.0-flash")
        path_sample = "C:/Users/rayso/Desktop/python/" + name_img  # Use raw string to handle backslashes
        # Prepare the prompt template
        sample_txt = promt
        img = PIL.Image.open(path_sample)
        response = model.generate_content([img, sample_txt])
        file_data_string = response.text
        print(file_data_string)
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": 11,
            "Voice": file_data_string,
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
    elif dictt["Steps"] == "Description":
        promt2 = dictt["Voice"]
        objects = '''

        The Objects can only be : Noodles, Cookies, Potato Chips, Detergent, Cup, Lunch Box, Dice, Light Bulb, Glue gun
        '''
        promt = promt2 + objects
        image_url = f"http://192.168.50.147:8888{'/uploads/GSPR.jpg'}"
        print("Fetching image from:", image_url)
        image_response = requests.get(image_url)

        image_array = np.frombuffer(image_response.content, dtype=np.uint8)

        # Decode the image using OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Save the image using OpenCV
        cv2.imwrite(pathnum + "Robot_view.jpg", img)
        print("Image saved successfully using OpenCV!")
        # Configuration and setup
        genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')  # Replace with your actual API key
        model = genai.GenerativeModel("gemini-2.0-flash")
        path_sample = "C:/Users/rayso/Desktop/python/Robot_view.jpg"  # Use raw string to handle backslashes
        # Prepare the prompt template
        sample_txt = promt
        img = PIL.Image.open(path_sample)
        response = model.generate_content([img, sample_txt])
        file_data_string = response.text
        print(file_data_string)
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": 10,
            "Voice": file_data_string,
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
    elif dictt["Steps"] == "Enumeration":
        promt2 = dictt["Voice"].lower()
        if "food" in promt2 or "kitchen" in promt2 or "item" in promt2:
            promt1 = '''
                    (Category)      (Object)
                    Food:           Noodles, Cookies, Potato Chips, Caramel Corn
                    Kitchen Item:   Detergent, Cup, Lunch Box, Sponge
                    Task Item :     Light Bulb, Dice, Glue Gun, Phone Stand
                    Question:
                    '''
            promt = promt1 + promt2
        else:
            promt = promt2

        image_url = f"http://192.168.50.147:8888{'/uploads/GSPR.jpg'}"
        print("Fetching image from:", image_url)
        image_response = requests.get(image_url)

        image_array = np.frombuffer(image_response.content, dtype=np.uint8)

        # Decode the image using OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Save the image using OpenCV
        cv2.imwrite(pathnum + "Robot_view.jpg", img)
        print("Image saved successfully using OpenCV!")
        # Configuration and setup
        genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')  # Replace with your actual API key
        model = genai.GenerativeModel("gemini-2.0-flash")
        path_sample = "C:/Users/rayso/Desktop/python/Robot_view.jpg"  # Use raw string to handle backslashes
        # Prepare the prompt template
        sample_txt = promt
        img = PIL.Image.open(path_sample)
        response = model.generate_content([img, sample_txt])
        file_data_string = response.text
        print(file_data_string)
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": 10,
            "Voice": file_data_string,
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
    elif dictt["Steps"] == "talk_list":
        now = datetime.now()

        # Extract information
        current_time = now.strftime("%H:%M:%S")
        current_month = now.strftime("%B")  # Full month name
        current_day_name = now.strftime("%A")  # Full weekday name
        day_of_month = now.strftime("%d")  # Day of the month as zero-padded decimal
        genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')
        model = genai.GenerativeModel("gemini-2.0-flash")
        sample_txt = f'''
        #	Talk , Answer
        01	something about yourself: We are Fambot from Macau Puiching Middle School, and I was made in 2024
        02	what day today is: today is 10th April in 2025
        03	what day tomorrow is: tomorrow is 10th April in 2025
        04	where RoboCup is held this year: the robocup 2025 is held in Brazil,Salvador
        05	what the result of 3 plus 5 is: The result of 3 plus 5 is 8.
        06	your team's name: my team name is Fambot
        07	where you come from: We are Fambot from Macau Puiching Middle School
        08	what the weather is like today: today weather is Raining
        09	what the time is: the current time is {current_time}
        Answer my question
        '''
        s1 = dictt["Questionasking"]
        response = model.generate_content([s1, sample_txt])
        print(response.text)

        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": "answer",
            "Voice": "None",
            "Questionasking": "talk_list",
            "answer": response.text
        }
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
    elif dictt["Steps"] == "answer_list":
        now = datetime.now()

        # Extract information
        current_time = now.strftime("%H:%M:%S")
        current_month = now.strftime("%B")  # Full month name
        current_day_name = now.strftime("%A")  # Full weekday name
        day_of_month = now.strftime("%d")  # Day of the month as zero-padded decimal
        genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')
        model = genai.GenerativeModel("gemini-2.0-flash")
        sample_txt = f'''
        Today is 10th April in 2025
        #	Question	Answer format
        01	What day is it today?	It is {current_month} {day_of_month}
        02	What is your team's name?	My team's name is FAMBOT
        03	What day is it tomorrow?	It is {current_month} {day_of_month}
        04	What is your name?	My name is FAMBOT robot
        05	What time is it?	It is {current_time}
        06	How many days are there in a week?	There are seven days in a week
        07	What is the prefectural capital of Shiga?	Ōtsu is the capital of Shiga Prefecture, Japan
        08	Where are you from?	I am from Macau Puiching middle school, Macau China
        09	What is the name of the venue for RoboCup Japan Open 2024?	The name of the venue is Shiga Daihatsu Arena
        10	Who is the leader on your team?	Our leader is Wu Iat Long
        11	What day of the week is today?	It is {current_day_name}
        12	What day of the month is today?	It is {day_of_month}
        13	What can you tell me about yourself	I am a robot made in 2024
        14	How many members are in your team?	We are 4 members in our team
        15	What has to be broken before you can use it?	An egg
        16	What is half of 1 plus 1?	It is 1.5
        17	What is the only mammal that can fly?	It is the bat
        18	What is the only bird that can fly backward?	It is the hummingbird

        Answer my question
        '''
        s1 = dictt["Questionasking"]
        s = "My question:" + s1
        response = model.generate_content([s, sample_txt])
        print(response.text)
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": "answer",
            "Voice": "answer_list",
            "Questionasking": "None",
            "answer": response.text
        }
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
    elif dictt["Steps"] == "color":
        promt2 = '''
        what color of clothes is this(you only can answer the following colors)
        01	Red
        02	Orange
        03	Yellow
        04	Green
        05	Blue
        06	Purple
        07	Pink
        08	Black
        09	White
        10	Gray
        11	Brown
        (answer format)
        the guy is wearing **color** t-shirt
        '''
        promt = promt2
        image_url = f"http://192.168.50.147:8888{'/uploads/GSPR_color.jpg'}"
        print("Fetching image from:", image_url)
        image_response = requests.get(image_url)
        image_array = np.frombuffer(image_response.content, dtype=np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Save the image using OpenCV
        cv2.imwrite(pathnum + "Robot_view.jpg", img)
        print("Image saved successfully using OpenCV!")
        # Configuration and setup
        genai.configure(api_key='AIzaSyBdTRu-rcBKbf86gjiMNtezBu1dEuxrWyE')  # Replace with your actual API key
        model = genai.GenerativeModel("gemini-2.0-flash")
        path_sample = "C:/Users/rayso/Desktop/python/Robot_view.jpg"  # Use raw string to handle backslashes
        # Prepare the prompt template
        sample_txt = promt
        img = PIL.Image.open(path_sample)
        response = model.generate_content([img, sample_txt])
        file_data_string = response.text
        print(file_data_string)
        file_data_string=file_data_string.replace("**","")
        questions = {
            "Question1": "None",
            "Question2": "None",
            "Question3": "None",
            "Steps": 12,
            "Voice": file_data_string,
            "Questionasking": "None",
            "answer": "None"
        }
        api_url = "http://192.168.50.147:8888/Fambot"
        response = requests.post(api_url, json=questions)
        result = response.json()
        print(result)
        time.sleep(2)
