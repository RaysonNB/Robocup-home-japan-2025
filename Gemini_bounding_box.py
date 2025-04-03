import os
import requests
import json
import cv2
import numpy as np
import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Optional

# Configuration
GOOGLE_API_KEY = "AIzaSyAHGCTBQvnNMTIXhcAFt0gEkQvAeG9mQ5A"  # Replace with your API key

# Initialize the client
genai.configure(api_key=GOOGLE_API_KEY)


class BoundingBox(BaseModel):
    box_2d: List[int]  # [y1, x1, y2, x2] in normalized coordinates (0-1000)
    label: str


def plot_bounding_boxes(image_uri: str, bounding_boxes: List[BoundingBox]) -> None:
    """Plot bounding boxes on an image with labels using OpenCV."""
    try:
        # Download image and convert to OpenCV format
        resp = requests.get(image_uri, stream=True, timeout=10)
        resp.raise_for_status()
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        height, width = img.shape[:2]

        # Define colors (BGR format)
        colors = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for i, bbox in enumerate(bounding_boxes):
            # Convert normalized coordinates to absolute coordinates
            y1, x1, y2, x2 = bbox.box_2d
            abs_y1 = int(y1 / 1000 * height)
            abs_x1 = int(x1 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)

            color = colors[i % len(colors)]

            # Draw rectangle
            cv2.rectangle(
                img,
                (abs_x1, abs_y1),
                (abs_x2, abs_y2),
                color,
                2
            )

            # Draw label background
            label = bbox.label
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            cv2.rectangle(
                img,
                (abs_x1, abs_y1 - text_height - 10),
                (abs_x1 + text_width + 10, abs_y1),
                color,
                -1  # Filled rectangle
            )

            # Draw label text
            cv2.putText(
                img,
                label,
                (abs_x1 + 5, abs_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )

        # Display image
        cv2.imshow("Image with Bounding Boxes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error displaying image: {e}")


def analyze_image(image_uri: str, prompt: str) -> Optional[List[BoundingBox]]:
    """Analyze an image with Gemini and return bounding boxes."""
    try:
        # Download the image
        img_data = requests.get(image_uri, stream=True, timeout=10).content

        # Prepare the model with specific instructions
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Create a detailed prompt that asks for structured JSON response
        structured_prompt = f"""
        Analyze this image and identify objects. For each object, return:
        - A bounding box in normalized coordinates [y1, x1, y2, x2] where each value is 0-1000
        - A descriptive label

        Return ONLY a JSON array of objects with 'box_2d' and 'label' properties.
        Example format:
        {{
            "box_2d": [100, 200, 300, 400],
            "label": "red apple"
        }}

        Current task: {prompt}
        """

        # Generate content
        response = model.generate_content(
            [
                structured_prompt,
                {"mime_type": "image/jpeg", "data": img_data}
            ],
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 2000
            }
        )

        # Try to parse the JSON response
        try:
            # Extract JSON from the response (Gemini might add markdown syntax)
            json_str = response.text.strip().replace('```json', '').replace('```', '').strip()
            boxes_data = json.loads(json_str)

            # Convert to BoundingBox objects
            bounding_boxes = [BoundingBox(**box) for box in boxes_data]

            print("Detected objects:")
            for box in bounding_boxes:
                print(f"- {box.label} at {box.box_2d}")

            return bounding_boxes

        except json.JSONDecodeError as e:
            print("Could not parse JSON response from Gemini:")
            print(response.text)
            return None

    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Example 1: Cupcake detection
    image_uri = "https://storage.googleapis.com/generativeai-downloads/images/Cupcakes.jpg"
    prompt = "Identify all cupcakes and describe their toppings"

    bounding_boxes = analyze_image(image_uri, prompt)
    if bounding_boxes:
        plot_bounding_boxes(image_uri, bounding_boxes)
