from niryo_robot_python_ros_wrapper import *
import os
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class Vision:
    def listener(self):
        # Initialize the node

        #rospy.init_node('listener', anonymous=True)

        # Subscribe to the topic

        #NED
        sub = rospy.Subscriber('/niryo_robot_vision/compressed_video_stream', CompressedImage)
        msg = rospy.wait_for_message('/niryo_robot_vision/compressed_video_stream', CompressedImage)

        sub.unregister()

        return msg
    def getImage(self, name=None):
        bridge = CvBridge()

        l =self.listener()

        cv_image = bridge.compressed_imgmsg_to_cv2(l, "bgr8")
        if name:
            cv2.imwrite('picture/'+name+'.png',cv_image)
        return cv_image
