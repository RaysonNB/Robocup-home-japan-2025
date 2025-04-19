import requests
import json
import mimetypes

def generate_content(prompt_text: str = None, image_path: str = None) -> dict:
    """
    Sends a request to the Gemini Flask API to generate content.

    Args:
        prompt_text:  Optional text prompt to send to the API.
        image_path:   Optional path to an image file to send to the API.

    Returns:
        A dictionary containing the API response, or None if an error occurred.
    """
    url = "http://localhost:5000/generate"  # Adjust if your server is running on a different host/port

    files = {}
    data = {}

    if prompt_text:
        data['prompt'] = prompt_text

    if image_path:
        try:
            # Determine the MIME type of the image file
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                print("Error: Could not determine MIME type of image file.")
                return None

            # Open the image file in binary read mode
            with open(image_path, 'rb') as image_file:
                # Read the file contents into memory
                image_data = image_file.read()

            # Add the image data to the data dictionary with the MIME type
            files['image'] = (image_path, image_data, mime_type)



        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error opening image file: {e}")
            return None

    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()  # Parse JSON response

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


if __name__ == '__main__':
    # Get user input for the text prompt
    text_prompt = input("Enter your text prompt (leave blank to skip): ")

    # Get user input for the image path (optional)
    image_path = input("Enter the path to an image file (leave blank to skip): ")

    # Call the generate_content function
    result = generate_content(prompt_text=text_prompt, image_path=image_path)

    if result:
        print("API Response:", result.get('generated_text'))
    else:
        print("Content generation failed.")
