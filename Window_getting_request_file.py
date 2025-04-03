import requests
import cv2
import google.generativeai as genai
import numpy as np
import PIL.Image
from PIL import Image
image_url = f"http://192.168.50.147:8888/uploads/image1.jpg"
print("Fetching image from:", image_url)
genai.configure(api_key='AIzaSyAHGCTBQvnNMTIXhcAFt0gEkQvAeG9mQ5A') # Replace with your actual API key
model = genai.GenerativeModel("gemini-2.0-flash")
image_response = requests.get(image_url)

if image_response.status_code == 200:
    # Convert the response content to a numpy array
    image_array = np.frombuffer(image_response.content, dtype=np.uint8)

    # Decode the image using OpenCV
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Save the image using OpenCV
    cv2.imwrite('niga.jpg', img)
    print("Image saved successfully using OpenCV!")
    path=r"C:\Users\rayso\Desktop\python\niga.jpg"
    img = PIL.Image.open(path)
    response = model.generate_content([img, sample_txt])
    file_data_string = response.text
    print(file_data_string)
else:
    print("Failed to fetch image:", image_response.status_code)
