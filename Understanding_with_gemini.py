import google.generativeai as genai
import json
import time
import requests
genai.configure(api_key='AIzaSyAHGCTBQvnNMTIXhcAFt0gEkQvAeG9mQ5A')
model = genai.GenerativeModel("gemini-2.0-flash")
questions = {
    "Question1": "Question1",
    "Question2": "Question2",
    "Question3": "Question3",
    "Steps" : 2,
    "Voice" : "Voice"
}
api_url = "http://192.168.50.147:8888/Fambot"
response = requests.post(api_url, json=questions)
for i in range(3):
    while True:
        r = requests.get("http://192.168.50.147:8888/Fambot", timeout=2.5)
        response_data = r.text
        print("Response_data", response_data)
        dictt = json.loads(response_data)
        if dictt["Steps"]==0:
            s1 = dictt["Voice"]
            break
        time.sleep(2)
    s="***The Sentence:"+s1
    print("question",s)
    sample_txt ="""
    
    
    (The Sentence)(Task: Sentence Structure)(I give u)
    Manipulation1: Go to the $ROOM, grasp the $OBJECT on the $PLACE1 and place it on the $PLACE2.
    Manipulation2: Go to the $ROOM, grasp the $OBJECT on the $PLACE and give it to $PERSON.
    Vision (Enumeration)1: Tell me how many $CATEGORY_OBJ there are on the $PLACE.
    Vision (Enumeration)2: Tell me how many people in the $ROOM are $POSE/GESTURE.
    Vision (Description)1: Tell me what is the $OBJ_COMP object on the $PLACE.
    Vision (Description)2: Tell me the $PERS_INFO of the person at the $PLACE
    Navigation1: Go to the $ROOM, find $POSE/GESTURE person and follow (him | her).
    Navigation2: Go to the $ROOM, find $POSE/GESTURE person and guide (him|her) to the $ROOM.
    Speech1: Go to the $ROOM, find $PERSON at the $PLACE and answer (his | her) question.
    Speech2: Go to the $ROOM, find the person who is $POSE/GESTURE and tell (him | her) $TELL_LIST.
    
    (Questions)
    Question1: which Task is it(just one) [Manipulation1,Manipulation2,Vision (Enumeration)1,Vision (Enumeration)2,Vision (Description)1,Vision (Description)2,Navigation1,Navigation2,Speech1,Speech2] ?
    Question2: give me the $informations(make it in dictionary)?
    Question3: what the sentence mean, and what The robot should do(20words)(just give me one sentence)?
    
    
    
    (answer_format(python_dictronary_format),dont edit it)
    
    *** {"1":[],"2":[],"3":[]} ***
    
    
    """
    response = model.generate_content([s,sample_txt])
    file_data_string = response.text
    print(file_data_string)
    file_data_string=file_data_string.replace("```","")
    file_data_string=file_data_string.replace("python","")
    file_data_string=file_data_string.replace("***","")
    file_data_string=file_data_string.replace("json","")
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
        "Voice":"Voice"
    }
    api_url = "http://192.168.50.147:8888/Fambot"
    response = requests.post(api_url, json=questions)
    result = response.json()
    print(result)
    print("sent")
