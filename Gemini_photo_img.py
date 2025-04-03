import requests
import cv2
import numpy as np

url = "http://192.168.50.147:8888/upload_image"
file_path = "C:/Users/rayso/Desktop/python/image1.jpg"

try:
    # 1. Upload the image
    with open(file_path, 'rb') as f:
        files = {'image': (file_path.split('/')[-1], f)}
        response = requests.post(url, files=files)

    print("Upload Status Code:", response.status_code)
    upload_result = response.json()
    print("Server Response:", upload_result)
    print("link", upload_result['url'])
    # 2. Download the image using the returned URL
    if 'url' in upload_result:
        image_url = f"http://192.168.50.147:8888{upload_result['url']}"
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



        else:
            print("Failed to fetch image:", image_response.status_code)
    else:
        print("Error: Server did not return a valid image URL.")

except Exception as e:
    print("Error:", e)
