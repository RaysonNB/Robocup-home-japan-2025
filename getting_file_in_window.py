import requests
import cv2
import numpy as np
image_url = f"http://192.168.50.147:8888{'/uploads/GSPR.jpg'}"
print("Fetching image from:", image_url)

image_response = requests.get(image_url)

if image_response.status_code == 200:
# Convert the response content to a numpy array
    image_array = np.frombuffer(image_response.content, dtype=np.uint8)

    # Decode the image using OpenCV
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Save the image using OpenCV
    cv2.imwrite('Robot_view.jpg', img)
    print("Image saved successfully using OpenCV!")
