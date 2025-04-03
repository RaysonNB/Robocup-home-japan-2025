import time

import google.generativeai as genai
from datetime import datetime

# Get current date and time
now = datetime.now()

# Extract information
current_time = now.strftime("%H:%M:%S")
current_month = now.strftime("%B")  # Full month name
current_day_name = now.strftime("%A")  # Full weekday name
day_of_month = now.strftime("%d")  # Day of the month as zero-padded decimal
genai.configure(api_key='AIzaSyAHGCTBQvnNMTIXhcAFt0gEkQvAeG9mQ5A')
model = genai.GenerativeModel("gemini-2.0-flash")
sample_txt=f'''
Today is 3rd April in 2025
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
for i in range(18):
    s=input("Question: ")
    s="My question:"+s
    response = model.generate_content([s,sample_txt])
    print(response.text)
