# A program which illustrates how subscribing to multiple topics can break cv2.imshow()
import threading
import time
import signal 
import os 
import sys

import cv2
import cv_bridge
import numpy as np

import rospy
import tf2_ros
import message_filters

from sensor_msgs.msg import Image, CameraInfo

image = np.zeros((720, 1280, 3), np.uint8)
image_lock = threading.Lock()
quit_event = threading.Event()


class OpenCVTest():

    def __init__(self):
        rospy.init_node('opencv_test')

        self.init_ros_pub_and_sub()
        self.init_tf_buffer

    def run(self):
        rospy.spin()

    def init_ros_pub_and_sub(self) -> None:
        """
        Initialize ROS nodes, register subscribers
        """
        color_image_sub = message_filters.Subscriber(
            '/camera_1/color/image_raw', Image, queue_size=2)
        
        camera_info_sub = message_filters.Subscriber(
            '/camera_1/color/camera_info', CameraInfo, queue_size=2)
        depth_image_sub = message_filters.Subscriber(
            '/camera_1/aligned_depth_to_color/image_raw', Image, queue_size=2)

        ts = message_filters.TimeSynchronizer(
                [color_image_sub, camera_info_sub, depth_image_sub], 1000)
        ts.registerCallback(self.callback)

    def init_tf_buffer(self):
        """
        Initialize buffer to listen to /tf and /tf_static
        """
        tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        tf2_ros.TransformListener(tf_buffer)

        return tf_buffer

    def callback(self, color_img, camera_info=None, depth_img=None):
        callback_start_time = time.time()

        bridge = cv_bridge.CvBridge()
        color_img_cv = bridge.imgmsg_to_cv2(color_img, 'bgr8')

        with image_lock:
            np.copyto(image, color_img_cv)

        callback_end_time = time.time()
        print('Callback Time :: {}'.format(callback_end_time-callback_start_time))
        

def display():
    local_image = np.zeros((720, 1280, 3), np.uint8)
    while not quit_event.is_set():
        with image_lock:
            np.copyto(local_image, image)
        try:
            cv2.imshow('test', local_image)
            cv2.waitKey(1)
        except:
            pass

if __name__ == '__main__':
    test = OpenCVTest()
    display_thread=threading.Thread(target=display)
    display_thread.start()

    def signal_handler(sig, frame):
        quit_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    test.run()