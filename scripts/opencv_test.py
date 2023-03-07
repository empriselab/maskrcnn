# A program which illustrates how subscribing to multiple topics can break cv2.imshow()

import cv2
import cv_bridge

import rospy
import tf2_ros
import message_filters

from sensor_msgs.msg import Image, CameraInfo


class OpenCVTest():

    def __init__(self, multiple_topics):
        rospy.init_node('opencv_test')

        # if True, will break OpenCV
        self.multiple_topics = multiple_topics

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
        
        if self.multiple_topics:
            camera_info_sub = message_filters.Subscriber(
                '/camera_1/color/camera_info', CameraInfo, queue_size=2)
            depth_image_sub = message_filters.Subscriber(
                '/camera_1/aligned_depth_to_color/image_raw', Image, queue_size=2)

            ts = message_filters.TimeSynchronizer(
                [color_image_sub, camera_info_sub, depth_image_sub], 10)
        else:
            ts = message_filters.TimeSynchronizer(
                [color_image_sub], 10)

        ts.registerCallback(self.callback)

    def init_tf_buffer(self):
        """
        Initialize buffer to listen to /tf and /tf_static
        """
        tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        tf2_ros.TransformListener(tf_buffer)

        return tf_buffer

    def callback(self, color_img, camera_info=None, depth_img=None):
        bridge = cv_bridge.CvBridge()
        color_img_cv = bridge.imgmsg_to_cv2(color_img, 'bgr8')

        cv2.imshow('test', color_img_cv)
        cv2.waitKey(1)


if __name__ == '__main__':
    test = OpenCVTest(multiple_topics=True)
    test.run()