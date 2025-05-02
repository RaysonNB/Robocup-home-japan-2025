import rospy
import cv2, os, time
from loguru import logger
from std_srvs.srv import Empty
from LemonEngine.sensors import Camera
from LemonEngine.hardwares.respeaker import Respeaker
from LemonEngine.hardwares.chassis import Chassis
from RobotChassis import RobotChassis

POINT_DINING_ROOM = (-0.913,0.257,0.001) ##



def main():
    clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
    chassis_move = Chassis()
    chassis = RobotChassis()

    # navigator = Navigator()
    respeaker = Respeaker(enable_espeak_fix=True)
    cam1 = Camera("/cam2/color/image_raw", "bgr8")
    cam2 = Camera("/cam2/depth/image_raw", "passthrough")
    width, height = cam1.width, cam1.height
    cx, cy = width // 2, height // 2
    rate = rospy.Rate(20)

    def walk_to(point):
        chassis.move_to(*point)
        while not rospy.is_shutdown():
            # 4. Get the chassis status.
            rate = rospy.Rate(20)
            code = chassis.status_code
            text = chassis.status_text
            logger.debug(f"Nav: {text}")
            rate.sleep()
            
            if code == 3:
                break
            if code == 4:
                logger.error("Fail to get a plan")
                respeaker.say("Fail to get a plan")
                clear_costmaps
                chassis.move_to(*point)
        return
    
    while not rospy.is_shutdown():
        frame = cam1.get_frame()
        depth_frame = cam2.get_frame()
        depth_frame = cv2.resize(depth_frame, (width, height))
        depth = depth_frame[cy, cx]

        frame = cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        frame = cv2.putText(frame, f"{depth}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if depth > 2000:
            respeaker.say("The door is opened")

            chassis_move.set_linear(0.25)
            time.sleep(5)

            chassis_move.stop_moving()
            break

        cv2.imshow("depth", depth_frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # navigate
    clear_costmaps
    walk_to([])
    clear_costmaps
    walk_to([])
    clear_costmaps
    walk_to([])


if __name__ == '__main__':
    rospy.init_node('test_camera', anonymous=True)
    main()
