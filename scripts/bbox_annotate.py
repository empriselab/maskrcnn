# Play a bag file and extract image, depth, and joint data
#
# Author: Thomas Patton (tjp93)

import os
import time
import itertools
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

import cv2
import cv_bridge

import rospy
import tf2_ros
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, TransformStamped
from image_geometry import PinholeCameraModel


class BBoxAnnotator():
    """
    A class for automatically labeling bounding boxes given a single bounding box
    """

    def __init__(self, image_dim=(720, 1280), show_cv=False, save_images=False):
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
        self.max_depth = 0.49    # REPLACE WITH SOMETHING MORE SCIENTIFIC 

        # initial base to camera transform
        self.init_camera_tf = None

        # False for initial callback, True otherwise
        self.world_created = False

        # storage for 3D bounding prisms in the scene
        self.objects = []

        # count number of callback invocations
        self.callback_counter = 0

        self.init_ros_subscribers()
        self.tf_buffer = self.init_tf_buffer()
        rospy.spin()

    def init_ros_subscribers(self) -> None:
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

    def init_tf_buffer(self):
        """
        Initialize buffer to listen to /tf and /tf_static
        """
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer)
        return tf_buffer

    def create_named_windows(self, height:int, width:int) -> None:
        """
        Creates named CV2 windows
        """
        x, y = 25, 25
        topics = [
            '/camera_1/color/image_raw', 
            # '3d_projection'
        ]
        for t in topics:
            cv2.namedWindow(t, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(t, width, height)
            cv2.moveWindow(t, x, y)
            y += 360

    def get_base_2_camera_tf(self):
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

    def get_camera_info(self, camera_info: CameraInfo):
        """
        Extract fx, fy, cx, cy for future use
        """
        self.fx = camera_info.K[0]
        self.fy = camera_info.K[4]
        self.cx = camera_info.K[2]
        self.cy = camera_info.K[5]

    def transform_to_mat(self, transform:TransformStamped) -> np.array:
        """
        Create a homogenous transformation matrix from a transform message
        """
        R = np.zeros((4,4))
        R[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        R[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        R[3,3] = 1
        return R

    def convert_images(self, color_img: Image, depth_img: Image):
        """
        Use a CVBridge to convert our color and depth images
        """
        return (
            self.cv_bridge.imgmsg_to_cv2(
                color_img)[:, :, ::-1].astype('uint8'),    # rgb->bgr
            self.cv_bridge.imgmsg_to_cv2(depth_img, "64FC1")
        )

    def tmp_get_segmentation(self):
        """
        TEMPORARY! gets the (x,y) pixel locations corresponding to the piece of salami
        Will eventually use something sophisticated like a JSON file
        """
        blank = np.zeros((self.height, self.width))
        cv2.circle(blank, (694, 165), 47, 1, thickness=-1)    # 47
        seg = np.where(blank == 1)
        return seg 

    def vec_pixel_to_world(self, mat:np.array, depth:np.array, camera_info:CameraInfo):
        """
        A vectorized version of the above. `mat` represents a list of x,y,depth points
        """
        assert mat.shape[0] == depth.shape[0]

        self.get_camera_info(camera_info)
        world = np.zeros((mat.shape[0], 3))

        world[:, 0] = (depth / self.fx) * (mat[:, 0] - self.cx)
        world[:, 1] = (depth / self.fy) * (mat[:, 1] - self.cy)
        world[:, 2] = depth 

        return world

    def create_3d_world(self, camera_info: CameraInfo, depth_img_cv: np.array) -> None:
        """
        Use human-annotated bboxes to render 3d bounding prisms of objects
        """

        depth_map = depth_img_cv / 1000.    # mm -> m
        depth_values = depth_map[self.segmentation].flatten()

         # here row values correspond to y, column values correspond to x so we must invert
        reshape_coords = np.array(list(zip(self.segmentation[1], self.segmentation[0])))   
        world = self.vec_pixel_to_world(reshape_coords, depth_values, camera_info)
        self.objects.append(world)

        self.world_created = True

    def compute_projection(self, camera_info: CameraInfo, transform_mat:np.array=None) -> np.array:
        """
        Placeholder fn for rendering 3D -> 2D projections
        """
        if transform_mat is None:
            rotation_vec, _ = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))
            translation_vec = np.array([0.0, 0.0, 0.0])
        else:
            rotation_vec, _ = cv2.Rodrigues(transform_mat[:3, :3]) 
            rotation_vec = np.multiply(rotation_vec, np.array([[1.0], [1.1], [1.0]]))
            translation_vec = transform_mat[0:3, -1]

        self.get_camera_info(camera_info)
        camera_mat = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        projection, _ = cv2.projectPoints(
            np.array(self.objects[0]),    # TODO: fix logic here
            rotation_vec,
            -translation_vec,
            camera_mat,
            None
        )

        return projection[:, 0, :]    # ignore weird y coord

    def display_windows(self, color_img: np.array, depth_img: np.array, projection: np.array) -> None:
        """
        Helper fn to display all OpenCV windows
        """
        try:
            for p in projection:
                cv2.circle(color_img, (int(p[0]), int(p[1])),
                        1, color=(255, 0, 0), thickness=1)
        except:
            pass

        # imshow, push to top, save if needed
        cv2.imshow('/camera_1/color/image_raw', color_img)
        if self.save_images:
            cv2.imwrite('../data/interim/{0:05d}.png'.format(self.callback_counter), color_img)

        cv2.setWindowProperty('/camera_1/color/image_raw',
                              cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)

    # def tmp_callback(self, camera_info:CameraInfo, initial_base_to_camera_tf, current_base_to_camera_tf):

    #     current_camera_wrt_init_camera = np.linalg.inv(initial_base_to_camera_tf) @ current_base_to_camera_tf

    #     pinhole = PinholeCameraModel()
    #     pinhole.fromCameraInfo(camera_info)
    #     projections = []
    #     salami = self.objects[0]

    #     for point in salami:
    #         moved_point = current_camera_wrt_init_camera @ np.array([point[0], point[1], point[2], 1]) 
    #         out =  pinhole.project3dToPixel(tuple([moved_point[1], moved_point[0], moved_point[2]]))
    #         print(out)
    #         projections.append(out)

    #     return projections


    def callback(self, camera_info: CameraInfo, color_img: Image, depth_img: Image):
        """
        Hook function for info published to camera topics
        """
        callback_start_time = time.time()

        # convert from ROS msgs to np arrays
        color_img_cv, depth_img_cv = self.convert_images(color_img, depth_img)

        # get current base to camera transform
        base_to_camera_tf = self.transform_to_mat(self.get_base_2_camera_tf())

        if not self.world_created:
            # initial callback: store current tf, create 3d world
            self.get_camera_info(camera_info)
            self.init_camera_tf = base_to_camera_tf
            self.create_3d_world(camera_info, depth_img_cv)
            projection = self.compute_projection(camera_info)
        else:
            # regular callback: find transform to position
            # init_camera_to_camera_tf = np.linalg.inv(base_to_camera_tf) @ self.init_camera_tf
            init_camera_to_camera_tf = np.linalg.inv(self.init_camera_tf ) @ base_to_camera_tf
            projection = self.compute_projection(camera_info, init_camera_to_camera_tf)
            # projection = self.tmp_callback(camera_info, self.init_camera_tf, init_camera_to_camera_tf)


        if self.show_cv:
            self.display_windows(color_img_cv, depth_img_cv, projection)

        callback_end_time = time.time()
        print("Callback Time : {}".format(
            callback_end_time - callback_start_time))

        self.callback_counter += 1


def main():
    BBoxAnnotator(
        show_cv=True, save_images=False
    )


if __name__ == '__main__':
    main()


# general idea
# - we have a transform from the base link to depth camera
# - our bounding box gives us a rectangle on the depth image
#
# scan depth image rectangle, get depth max and depth min
# construct an approx 3d rectangular prism, using depth min/max
#
# now, at some rate:
# - check the base->camera transform
# - use these as a change of camera
# - use opencv to compute projection back to 2D
# - use convex hull as bbox
#

# need some sort of initial callback to use the initial bbox to
# render the real world objects.
# then regular callback


# GENERAL ALGORITHM
#
# phase 1: initial callback
# here, we have a list of human-annotated bounding boxes on the first
# frame of video data. we need to get transform data to get the camera
# location and then use depth data to turn those 2d bounding boxes into
# 3d bounding cubes


# NOTES
# any rosbag opening takes a long time after `roscore` has been shut down, cache
# seems to reset. overall weird behavior with reopening and such
#
# /tf has data from base_link to end_effector_link
# /tf_static has data for end_effector_link to camera stuff

# assume schema below
# "boundingBoxes": [
# {
#     "height": 2832,
#     "label": "bird",
#     "left": 681,
#     "top": 599,
#     "width": 1364
# }


# OLD CODE


# def create_3d_world(self, camera_info: CameraInfo, color_img_cv: np.array, depth_img_cv: np.array) -> None:
#     """
#     Use human-annotated bboxes to render 3d bounding prisms of objects
#     """
#     depth_map = depth_img_cv / 1000.    # mm -> m
#     for bbox in self.given_bboxes:
#         height, width = bbox['height'], bbox['width']
#         left, top = bbox['left'], bbox['top']

#         # find the min/max depth values in the region to build bounding cube
#         depth_region = depth_map[top:top+height, left:left+width].flatten()
#         trunc_depth_region = depth_region[np.where(
#             (depth_region > 0.1) & (depth_region < 1.0))]
#         min_depth, max_depth = np.min(
#             trunc_depth_region), np.max(trunc_depth_region)

#         # generate 3D mesh of a cube based on bbox and depth data
#         cube_mesh = self.generate_cube_mesh(
#             top, left, height, width, min_depth, max_depth, camera_info)
#         self.objects.append(cube_mesh)

#     # now that we have our objects, world has been created
#     self.world_created = True


# def generate_cube_mesh(self, top: int, left: int, height: int, width: int, min_depth: float, max_depth: float, camera_info: CameraInfo):
#     """
#     Generates a 3D mesh cube to represent a 3D bounding cube
#     TODO: Replace with parallelizable np method
#     """
#     x_1, y_1, z_1 = self.pixel_to_world(left, top, min_depth, camera_info)
#     x_2, y_2, z_2 = self.pixel_to_world(
#         left+width, top+height, max_depth, camera_info)

#     x_space = list(np.linspace(x_1, x_2, 10))
#     y_space = list(np.linspace(y_1, y_2, 10))
#     z_space = list(np.linspace(z_1, z_2, 10))

#     points = []
#     for (xi, yj, zk) in itertools.product(x_space, y_space, z_space):
#         points.append([xi, yj, zk])

#     return points

# def pixel_to_world(self, x, y, depth, camera_info):
#     """
#     Converts a pixel coordinate (x,y) into real world (x,y,z) space
#     """
#     fx, fy, cx, cy = self.get_camera_info(camera_info)
#     world_x = (depth / fx) * (x - cx)
#     world_y = (depth / fy) * (y - cy)
#     world_z = depth
#     return world_x, world_y, world_z

# DATA_PATH = "../data/feeding_infra_training_data/experiment_1.bag"
# TOPICS = [
#     '/camera_1/color/image_raw',
#     '/camera_1/aligned_depth_to_color/image_raw',
#     '/tf'
# ]


# def load_bagfile() -> rosbag.Bag:
#     """
#     Loads a bagfile from DATA_PATH with a given bagfile_name
#     """
#     bag = rosbag.Bag(DATA_PATH)
#     print("Loaded bag {}".format(DATA_PATH))
#     topics = bag.get_type_and_topic_info()[1].keys()
#     print("Available Bag Topics: {}".format(topics))
#     print()

#     return bag


# def get_bag_messages(bag: rosbag.Bag) -> Generator:
#     """
#     Return a generator of bag messages from the bagfile
#     """
#     bag_messages = bag.read_messages(topics=TOPICS)
#     return bag_messages


# def create_named_windows() -> None:
#     """
#     Creates named CV2 windows based on `TOPICS`
#     """
#     x, y = 25, 25
#     for t in TOPICS:
#         if 'camera' in t:
#             cv2.namedWindow(t, cv2.WINDOW_NORMAL)
#             cv2.resizeWindow(t, 360, 360)
#             cv2.moveWindow(t, x, y)
#             y += 500


# def parse_image(image_message, topic, bridge) -> None:
#     """
#     Uses a CV Bridge to parse a ROS image message
#     """
#     # image_bytes = image_message.data
#     # enc, step = image_message.encoding, image_message.step
#     # h, w = image_message.height, image_message.width
#     # print(h, w, enc, step)

#     cv2_image = bridge.imgmsg_to_cv2(image_message).astype('uint8')
#     if 'depth' not in topic:
#         cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
#     else:
#         # _, cv2_image = cv2.threshold(cv2_image, 127,255,cv2.THRESH_TOZERO)
#         pass
#     cv2.imshow(topic, cv2_image)
#     cv2.waitKey(2)


# def generate_animation(bag_messages:Generator) -> None:
#     """
#     Creates an animation of the bagfile
#     """
#     bridge = cv_bridge.CvBridge()
#     n_messgaes_to_show = 5000
#     i = 0
#     while i < n_messgaes_to_show:
#         message = next(bag_messages)
#         message_data, topic = message.message, message.topic

#         if 'camera' in topic:
#             parse_image(message_data, topic, bridge)
#         else:
#             msg = message.message
#             print(msg)
#             # transforms = msg.transforms
#             # print([(t.header.frame_id, t.child_frame_id) for t in transforms])
#             # print(type(message.message.transforms[0]))

#         i += 1

# def main():
#     """
#     Run the script
#     """
#     start_time = time.time()

#     # load our selected bagfile
#     bag = load_bagfile()
#     bag_messages = get_bag_messages(bag)

#     # run openCV
#     create_named_windows()
#     generate_animation(bag_messages)

#     end_time = time.time()
#     print('Script complete, total time: {}'.format(end_time-start_time))
