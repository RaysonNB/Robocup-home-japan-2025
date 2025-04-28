import rospy
import re, json, requests
import cv2, os, time
import mimetypes
from loguru import logger
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from LemonEngine.sensors import Camera
from LemonEngine.hardwares.respeaker import Respeaker
from LemonEngine.hardwares.chassis import Chassis
from RobotChassis import RobotChassis

# TABLE_P = (3.498, 3.339, -1.664)
# FOOD_POINT = (6.34, 3.07, 1.5)
# TASK_POINT = (5.13, 2.90, 1.5)
# UNKNOWN_POINT = (3.97, 2.85, 1.5)
# KITCHEN_POINT = (2.05, 3.72, 0)

TABLE_P = (1.772, -0.118, 0.8)
FOOD_POINT = (0.278, -0.139, -2.462)
TASK_POINT = (-0.529, 0.443, 2.197)
UNKNOWN_POINT = (1.638, -0.231, -0.910)
KITCHEN_POINT = (1.638, -0.231, -0.910)



PROMPT = """
# Instruction
Analyze the input image. Detect distinct objects and try your best to classify them using the `Object List` below. 
If an object isn't listed, use category `Unknown`. Be careful not to leave any items behide
You Must output *only* a JSON list containing objects with keys `"object"` and `"category"`. 
If no object here, please output a empty json list ```json[]```


# Object List
| ID | Object        | Category     |
|----|---------------|--------------|
| 1  | Noodles       | Food         |
| 2  | Cookies       | Food         |
| 3  | Potato Chips  | Food         |
| 4  | Detergent     | Kitchen Item |
| 5  | Cup           | Kitchen Item |
| 6  | Lunch Box     | Kitchen Item |
| 7  | Dice          | Task Item    |
| 8  | Light Bulb    | Task Item    |
| 9  | Block         | Task Item    |

* Note: The `Light Bulb` will be packed in a box
* Furnitures (i.e. Table, Chair) is not an object

# Example Output
```json
[
  {"object": "Dice", "category": "Task Item"},
  {"object": "Cookies", "category": "Food"},
  {"object": "Pen", "category": "Unknown"}
]
```
"""

cmd_vel=rospy.Publisher("/cmd_vel",Twist,queue_size=10)


def generate_content(prompt_text: str = None, image_path: str = None) -> dict:
    """
    Sends a request to the Gemini Flask API to generate content.

    Args:
        prompt_text:  Optional text prompt to send to the API.
        image_path:   Optional path to an image file to send to the API.

    Returns:
        A dictionary containing the API response, or None if an error occurred.
    """
    url = "http://192.168.50.142:5000/generate"  # Adjust if your server is running on a different host/port
    files = {}
    data = {}
    if prompt_text:
        data['prompt'] = prompt_text
    if image_path:
        try:
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                print("Error: Could not determine MIME type of image file.")
                return None
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
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

def move(forward_speed: float = 0, turn_speed: float = 0):
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    cmd_vel.publish(msg)





def main():
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    
    
    chassis = RobotChassis()
    respeaker = Respeaker(enable_espeak_fix=True)

    # navigator = Navigator()
    cam1 = Camera("/camera/color/image_raw", "bgr8")
    cam2 = Camera("/camera/depth/image_raw", "passthrough")
    width, height = cam1.width, cam1.height
    cx, cy = width // 2, height // 2
    rate = rospy.Rate(20)
    
    def walk_to(point):
        logger.info(f"Walk to {point}")
        tried = 0
        chassis.move_to(*point)
        clear_costmaps
        while not rospy.is_shutdown():
            # 4. Get the chassis status.
            rate = rospy.Rate(20)
            code = chassis.status_code
            text = chassis.status_text
            if code == 3:
                logger.success("Point Reached!")
                return True
                
            if code == 4:
                logger.error(f"Plan Failed (tried {tried})")
                respeaker.say("I am blocked, please move aside")
                clear_costmaps
                chassis.move_to(*point)
                if tried > 5:
                    break
                tried += 1

        chassis.move_base.cancel_all_goals()
        return False
                
    def ask_gemini(text):
        match = False
        while True:
            logger.info("Asking Gemini for objects")
            frame = cam1.get_frame()
            cv2.imwrite("./image.jpg", frame)
            text = generate_content(text, "./image.jpg").get('generated_text')
            if text is None:
                respeaker.say("Failed")
                continue
            text = text.replace("\n", "")
            text = text.replace("\r", "")
            print("Gemini Res", text)
    
            pattern = r"```json(.*?)```"  # Corrected regex pattern
            match = re.search(pattern, text)
    
            if match:
                json_string = match.group(1)  # Extract the content inside ```json ... ```
                logger.debug(json_string)
                json_object = json.loads(json_string)
                return json_object


    ##################################
    
    # while not rospy.is_shutdown():
    #     frame = cam1.get_frame()
    #     depth_frame = cam2.get_frame()
    #     depth_frame = cv2.resize(depth_frame, (width, height))
    #     depth = depth_frame[cy, cx]

    #     frame = cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    #     frame = cv2.putText(frame, f"{depth}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     if depth > 2000:
    #         respeaker.say("The door is opened")

    #         for i in range(10):
    #             move(0.25, 0)
    #             time.sleep(0.1)
            
    #         move(0, 0)
    #         break

    #     cv2.imshow("depth", depth_frame)
    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    clear_costmaps
    walk_to(TABLE_P)
    
    while True:
        respeaker.say("I am recognizing objects")
        json_object = ask_gemini(PROMPT)
        
        if len(json_object) == 0:
            respeaker.say("It seems the table is empty, task end")
            break
        
        respeaker.say("I see")
        for a_object in json_object[:3]:
            print(a_object)
            respeaker.say("Help me put the " + a_object["object"] + "on my robot arm")
            print("**OPEN_ARM")
            time.sleep(5)
            print("**CLOSE_ARM")

            respeaker.say(a_object["category"])
            if a_object["category"].lower() == "unknown":        walk_to(UNKNOWN_POINT)            
            if a_object["category"].lower() == "task item":     walk_to(TASK_POINT)
            if a_object["category"].lower() == "kitchen item":    walk_to(KITCHEN_POINT)
            if a_object["category"].lower() == "food":        walk_to(FOOD_POINT)
            
            respeaker.say("Putting Object")
            print("**OPEN_ARM")
            time.sleep(5)

            walk_to(TABLE_P)
    

if __name__ == '__main__':
    rospy.init_node('tidyup', anonymous=True)
    main()
