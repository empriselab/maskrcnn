# Play a bag file and extract image, depth, and joint data
#
# Author: Thomas Patton (tjp93)

import os
import sys
import time
import struct
import json
from pathlib import Path
import signal
import threading
import asyncio
from subprocess import Popen
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

import cv2
import cv_bridge

import rospy
import tf2_ros
import message_filters

from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped


class ImageExtractor():
    """
    A class for extracting the color images from ROSBAGs
    """

    def __init__(self, callback_to_save=0, callback_buffer=50, image_dim=(720,1280), rosbag_number=1):
        """
        Initialize

        Will save the `callback_buffer + callback_to_save`th callback
        """
        rospy.init_node('image_extractor')

        self.height, self.width = image_dim
        self.callback_to_save = callback_to_save
        self.callback_buffer = callback_buffer
        self.rosbag_number = rosbag_number

        self.callback_counter = 0

        self.training_dir, self.rosbag = self.init_filesystem()
        
        self.rosbag_process = self.play_rosbag()

        # spin up ROS
        self.tf_buffer = self.init_tf_buffer()
        self.init_ros_pub_and_sub()

    def run(self):
        """
        Run the module
        """
        rospy.spin()

    def init_filesystem(self):
        """
        Uses pathlib to get the `foodrecognition` base directory as well as the .json
        file containing the object segmentations
        """
        base_dir = Path(__file__).absolute().parents[1]
        training_dir = base_dir / "data" / "training"
        rosbag = base_dir / "data" / "feeding_infra_training_data" / "experiment_{}.bag".format(self.rosbag_number)
        return training_dir, rosbag

    def init_ros_pub_and_sub(self) -> None:
        """
        Initialize ROS nodes, register subscribers
        """
        camera_info_sub = message_filters.Subscriber(
            '/camera_1/color/camera_info', CameraInfo, queue_size=2)
        color_image_sub = message_filters.Subscriber(
            '/camera_1/color/image_raw', Image, queue_size=2)
        depth_image_sub = message_filters.Subscriber(
            '/camera_1/aligned_depth_to_color/image_raw', Image, queue_size=2)

        ts = message_filters.TimeSynchronizer(
            [camera_info_sub, color_image_sub, depth_image_sub], 1000)
        ts.registerCallback(self.callback)

    def init_tf_buffer(self):
        """
        Initialize buffer to listen to /tf and /tf_static
        """
        tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        tf2_ros.TransformListener(tf_buffer)

        return tf_buffer

    def play_rosbag(self):
        proc = Popen(['rosbag', 'play', str(self.rosbag)])
        return proc

    def convert_images(self, color_img: Image, depth_img: Image) -> tuple:
        """
        Use a CVBridge to convert our color and depth images
        """
        self.cv_bridge = cv_bridge.CvBridge()
        return (
            self.cv_bridge.imgmsg_to_cv2(
                color_img, "bgr8"),
            self.cv_bridge.imgmsg_to_cv2(depth_img, "32FC1")
        )

    def callback(self, camera_info: CameraInfo, color_img: Image, depth_img: Image):
        """
        Hook function for info published to camera topics
        """
        # populate tf buffer for first 50 callbacks
        if self.callback_counter < self.callback_buffer:
            self.callback_counter += 1
            return

        # convert from ROS msgs to np arrays
        color_img_cv, _ = self.convert_images(color_img, depth_img)

        if self.callback_counter == (self.callback_buffer + self.callback_to_save):
            save_location = str(self.training_dir / 'init_frame_bag_{}.png'.format(self.rosbag_number))
            cv2.imwrite(save_location, color_img_cv)
            
            # kill rosbag play and exit
            self.rosbag_process.terminate()
            rospy.signal_shutdown('terminated')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rosbag_number")
    args = parser.parse_args()

    module = ImageExtractor(
        rosbag_number=args.rosbag_number,
        callback_buffer=50,
        callback_to_save=0
    )

    def signal_handler(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    module.run()

if __name__ == '__main__':
    main()

 

