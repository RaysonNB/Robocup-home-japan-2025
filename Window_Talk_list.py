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
#	Talk , Answer
01	something about yourself: We are Fambot from Macau Puiching Middle School, and I was made in 2024
02	what day today is: today is 3rd April in 2025
03	what day tomorrow is: tomorrow is 4rd April in 2025
04	where RoboCup is held this year: the robocup 2025 is held in Brazil,Salvador
05	what the result of 3 plus 5 is: The result of 3 plus 5 is 8.
06	your team's name: my team name is Fambot
07	where you come from: We are Fambot from Macau Puiching Middle School
08	what the weather is like today: today weather is Raining
09	what the time is: the current time is {current_time}
Answer my question
'''
for i in range(18):
    s=input("Question: ")
    s="My question:"+s
    response = model.generate_content([s,sample_txt])
    print(response.text)
