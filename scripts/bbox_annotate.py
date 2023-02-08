# Play a bag file and extract image, depth, and joint data
#
# Author: Thomas Patton (tjp93)

import os
import time
import itertools
import argparse
import struct

import numpy as np
from scipy.spatial.transform import Rotation

import cv2
import cv_bridge

import rospy
import tf2_ros
import message_filters

from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointCloud, PointField
from geometry_msgs.msg import Point, TransformStamped, Point32
from image_geometry import PinholeCameraModel


class BBoxAnnotator():
    """
    A class for automatically labeling bounding boxes given a single bounding box
    """

    def __init__(self, image_dim=(720, 1280), show_cv=False, save_images=False, pointcloud=None):
        """
        Initialize BBoxAnnotator module
        """
        rospy.init_node('bbox_annotator')

        # init CV
        self.height, self.width = image_dim
        self.cv_bridge = cv_bridge.CvBridge()
        self.show_cv = show_cv
        self.save_images = save_images
        if self.show_cv:
            self.create_named_windows(self.height, self.width)

        # REFACTOR: USING SEMANTIC SEGMENTATION INSTEAD
        # currently based on AWS labeling format
        # self.given_bboxes = [
        #     {"label": "salami", "left": 645, "width": 95, "top": 115, "height": 95}
        # ]

        # placeholder for semantic segmentation, will eventually read from JSON
        # TODO: read from JSON
        self.segmentation = self.tmp_get_segmentation()

        # initial base to camera transform
        self.init_camera_tf = None

        # False for initial callback, True otherwise
        self.world_created = False

        # storage for 3D bounding prisms in the scene
        self.objects = []

        self.pointcloud = pointcloud

        # count number of callback invocations
        self.callback_counter = 0

        # spin up ROS
        self.init_ros_pub_and_sub()
        self.tf_buffer = self.init_tf_buffer()
        rospy.spin()

    def init_ros_pub_and_sub(self) -> None:
        """
        Initialize ROS nodes, register subscribers
        """
        camera_info_sub = message_filters.Subscriber(
            '/camera_1/aligned_depth_to_color/camera_info', CameraInfo, queue_size=10)
        color_image_sub = message_filters.Subscriber(
            '/camera_1/color/image_raw', Image, queue_size=10)
        depth_image_sub = message_filters.Subscriber(
            '/camera_1/aligned_depth_to_color/image_raw', Image, queue_size=10)

        ts = message_filters.TimeSynchronizer(
            [camera_info_sub, color_image_sub, depth_image_sub], 1)
        ts.registerCallback(self.callback)

        self.pointcloud_pub = rospy.Publisher(
            '/thomas/pointcloud2', PointCloud2, queue_size=10)

    def init_tf_buffer(self):
        """
        Initialize buffer to listen to /tf and /tf_static
        """
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer)
        return tf_buffer

    def create_named_windows(self, height: int, width: int) -> None:
        """
        Creates named CV2 windows
        """
        x, y = 25, 25
        topics = [
            '/camera_1/color/image_raw',
        ]
        for t in topics:
            cv2.namedWindow(t, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(t, width, height)
            cv2.moveWindow(t, x, y)
            y += 360

    def get_base_to_camera_tf(self):
        """
        Get transform from base_link to camera
        """
        rate = rospy.Rate(1000.0)
        while not rospy.is_shutdown():
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',  'camera_1_color_optical_frame', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue
        return transform

    def register_camera_info(self, camera_info: CameraInfo):
        """
        Extract fx, fy, cx, cy for future use
        """
        self.fx = camera_info.K[0]
        self.fy = camera_info.K[4]
        self.cx = camera_info.K[2]
        self.cy = camera_info.K[5]

    def transform_to_mat(self, transform: TransformStamped) -> np.array:
        """
        Create a homogenous transformation matrix from a transform message
        """
        R = np.zeros((4, 4))
        R[:3, :3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y,
                                       transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        R[:3, 3] = np.array([transform.transform.translation.x, transform.transform.translation.y,
                            transform.transform.translation.z]).reshape(1, 3)
        R[3, 3] = 1
        return R

    def convert_images(self, color_img: Image, depth_img: Image):
        """
        Use a CVBridge to convert our color and depth images
        """
        return (
            self.cv_bridge.imgmsg_to_cv2(
                color_img, "bgr8"),
            self.cv_bridge.imgmsg_to_cv2(depth_img, "32FC1")
        )

    def tmp_get_segmentation(self):
        """
        TEMPORARY! gets the (x,y) pixel locations corresponding to the piece of salami
        Will eventually use something sophisticated like a JSON file
        """
        blank = np.zeros((self.height, self.width))
        cv2.circle(blank, (694, 165), 47, 1, thickness=-1)    # manual config
        seg = np.where(blank == 1)
        return seg

    def vec_pixel_to_world(self, mat: np.array, depth: np.array):
        """
        A vectorized version of the above. `mat` represents a list of x,y,depth points
        """
        assert mat.shape[0] == depth.shape[0]

        world = np.zeros((mat.shape[0], 3))

        world[:, 0] = (depth / self.fx) * (mat[:, 0] - self.cx)    # world x
        world[:, 1] = (depth / self.fy) * (mat[:, 1] - self.cy)    # world y
        world[:, 2] = depth                                        # world z

        return world

    def create_3d_world(self, depth_img_cv: np.array) -> None:
        """
        Use human-annotated bboxes to render 3d bounding prisms of objects
        """
        depth_map = depth_img_cv / 1000.0    # mm -> m
        depth_values = depth_map[self.segmentation]

        # here row values correspond to y, column values correspond to x so we must invert
        reshape_coords = np.array(
            list(zip(self.segmentation[1], self.segmentation[0])))
        world = self.vec_pixel_to_world(reshape_coords, depth_values)
        self.objects.append(world)

        self.world_created = True

    def compute_projection(self, transform_mat: np.array = None) -> np.array:
        """
        Placeholder fn for rendering 3D -> 2D projections
        """
        if transform_mat is None:
            rotation_vec, _ = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))
            translation_vec = np.array([0.0, 0.0, 0.0])
        else:
            rotation_vec, _ = cv2.Rodrigues(transform_mat[:3, :3])
            translation_vec = transform_mat[0:3, -1]

        camera_mat = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        projection, _ = cv2.projectPoints(
            np.array(self.objects[0]),    # TODO: fix logic here
            rotation_vec,
            translation_vec,
            camera_mat,
            None
        )
        return projection[:, 0, :]    # ignore weird y coord

    def display_windows(self, color_img: np.array, projection: np.array) -> None:
        """
        Helper fn to display OpenCV window
        """
        segmentation_image = np.copy(color_img)
        try:
            for p in projection:
                cv2.circle(segmentation_image, (int(p[0]), int(p[1])),
                           1, color=(255, 0, 0), thickness=1)
        except:
            pass

        # imshow, push to top, save if needed
        cv2.imshow('/camera_1/color/image_raw', segmentation_image)
        if self.save_images:
            cv2.imwrite(
                '../data/interim/{0:05d}.png'.format(self.callback_counter), segmentation_image)

        cv2.setWindowProperty('/camera_1/color/image_raw',
                              cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)

        return segmentation_image

    def callback(self, camera_info: CameraInfo, color_img: Image, depth_img: Image):
        """
        Hook function for info published to camera topics
        """
        callback_start_time = time.time()

        # convert from ROS msgs to np arrays
        color_img_cv, depth_img_cv = self.convert_images(color_img, depth_img)

        # get current base to camera transform
        base_to_camera_tf = self.transform_to_mat(self.get_base_to_camera_tf())

        if not self.world_created:
            # initial callback: store current tf, create 3d world
            self.register_camera_info(camera_info)
            self.init_camera_tf = base_to_camera_tf
            self.create_3d_world(depth_img_cv)
            projection = self.compute_projection()
        else:
            # regular callback: find transform to position
            init_camera_to_camera_tf = np.linalg.inv(
                base_to_camera_tf) @ self.init_camera_tf
            # init_camera_to_camera_tf = np.linalg.inv(self.init_camera_tf ) @ base_to_camera_tf
            projection = self.compute_projection(init_camera_to_camera_tf)

        if self.show_cv:
            segmentation_image = self.display_windows(color_img_cv, projection)

        # create/publish a pointcloud
        if self.pointcloud == "segmented":
            self.create_pointcloud(
                color_img, color_img=segmentation_image, depth_img=depth_img_cv)
        elif self.pointcloud == "raw":
            self.create_pointcloud(
                color_img, color_img=color_img_cv, depth_img=depth_img_cv)

        callback_end_time = time.time()
        print("Callback Time : {}".format(
            callback_end_time - callback_start_time))

        self.callback_counter += 1

    def create_pointcloud(self, msg, color_img: np.array, depth_img: np.array) -> None:
        """
        Creates and publishes a pointcloud from provided rgb and depth images
        """
        header = msg.header

        fx_inv, fy_inv = 1.0/self.fx, 1.0/self.fy

        points = []
        for i in range(msg.width):
            for j in range(msg.height):
                if i % 5 == 0 and j % 5 == 0:

                    depth = depth_img[j, i] * 0.001
                    color = color_img[j, i]

                    x = depth * ((i - self.cx)) * fx_inv
                    y = depth * ((j - self.cy)) * fy_inv
                    z = depth
                    pt = [x, y, z, 0]

                    b, g, r, a = color[0], color[1], color[2], 255
                    rgb = struct.unpack(
                        'I', struct.pack('BBBB', b, g, r, a))[0]
                    pt[3] = rgb
                    points.append(pt)

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        pc2 = point_cloud2.create_cloud(header, fields, points)
        self.pointcloud_pub.publish(pc2)


def main():
    BBoxAnnotator(
        show_cv=True,
        save_images=False,
        pointcloud="raw"    # options "segmented", "raw", None
    )


if __name__ == '__main__':
    main()
