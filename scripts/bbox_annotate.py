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

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import cv2
import cv_bridge

import rospy
import tf2_ros
import message_filters

from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped


# configure global image and locks to avoid ROS/OpenCV threading issues
global_image = np.zeros((720,1280,3), dtype=np.uint8)
image_lock = threading.Lock()
quit_event = threading.Event()


class BBoxAnnotator():
    """
    A class for automatically labeling bounding boxes given a single bounding box
    """

    def __init__(self, bagfile="experiment_1.bag", image_dim=(720, 1280),
                 show_cv=False, save_images=False, pointcloud_type=None, save_callback_idxs=None):
        """
        Initialize BBoxAnnotator module
        """
        rospy.init_node('bbox_annotator')

        # configure files
        self.base_dir, self.segmentation_file = self.init_filesystem(bagfile)

        # init CV
        self.height, self.width = image_dim

        # segmentation of the scene
        self.segmentation = self.tmp_get_segmentation()

        # initial base to camera transform
        self.init_camera_tf = None

        # False for initial callback, True otherwise
        self.world_created = False
        self.pointcloud_type = pointcloud_type

        # storage for 3D bounding prisms in the scene
        self.objects = []
        self.save_callback_idxs = save_callback_idxs

        # a binary image that masks the forque in the image
        self.fork_mask = self.create_fork_mask()

        # count number of callback invocations
        self.callback_counter = 0

        # TMP STUFF FOR TEMPLATE MATCHING
        # self.template = self.tmp_salami_bbox()

        # spin up ROS
        self.tf_buffer = self.init_tf_buffer()
        self.init_ros_pub_and_sub()

    def run(self):
        """
        Run the annotator
        """
        rospy.spin()

    def init_filesystem(self, bagfile: str):
        """
        Uses pathlib to get the `foodrecognition` base directory as well as the .json
        file containing the object segmentations
        """
        base_dir = Path(__file__).absolute().parents[1]
        segmentation_file_name = os.path.splitext(bagfile)[0] + ".json"
        segmentation_file = base_dir / "data" / "segmentations" / segmentation_file_name
        return base_dir, segmentation_file

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

        self.pointcloud_pub = rospy.Publisher(
            '/thomas/pointcloud2', PointCloud2, queue_size=10)

    def init_tf_buffer(self):
        """
        Initialize buffer to listen to /tf and /tf_static
        """
        tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        tf2_ros.TransformListener(tf_buffer)

        # time.sleep(0.5)
        return tf_buffer

    def create_fork_mask(self):
        white = np.ones((self.height, self.width))
        fork_outline = np.array([
            [714, 717], [707, 640], [733, 636], [724, 534],
            [778, 534], [785, 567], [815, 577], [865, 637],
            [987, 635], [988, 657], [1059, 718]
        ])
        cv2.fillPoly(white, [fork_outline], 0)
        return white

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

    def get_base_to_camera_tf(self, stamp):
        """
        Get transform from base_link to camera
        """
        secs, nsecs = stamp.secs, stamp.nsecs

        rate = rospy.Rate(100.0)
        t0 = time.time()
        while not rospy.is_shutdown():
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',  'end_effector_link', rospy.Time(secs=secs, nsecs=nsecs))
                # transform = self.tf_buffer.lookup_transform(
                # 'base_link',  'camera_1_color_optical_frame', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rate.sleep()
                # print("Base to ee: ",secs, nsecs, e)
                continue

        while not rospy.is_shutdown():
            try:
                camera_link_to_color_optical_transform = self.tf_buffer.lookup_transform(
                    'camera_1_link',  'camera_1_color_optical_frame', rospy.Time(secs=secs, nsecs=nsecs))
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rate.sleep()
                print("camera links: ", secs, nsecs, e)
                continue

        base_to_ee = self.transform_to_mat(transform)

        quat = [0.50080661, 0.49902445, 0.50068027, -0.49948635]
        end_effector_to_camera = np.zeros((4, 4))
        end_effector_to_camera[:3, :3] = Rotation.from_quat(quat
            ).as_matrix()
        end_effector_to_camera[:3, 3] = np.array([0.014, 0.060, 0.034])
        end_effector_to_camera[3, 3] = 1

        # end_effector_to_camera[:3, 3] = np.array([0.0185, 0.058, 0.034])    # RAJAT VALS
        # # old camera calibration params
        # end_effector_to_camera[:3, :3] = Rotation.from_quat([0.50080661, 0.49902445, 0.50068027, -0.49948635]).as_matrix()

        camera_link_to_color_optical = self.transform_to_mat(
            camera_link_to_color_optical_transform)

        return base_to_ee @ end_effector_to_camera @ camera_link_to_color_optical

        # return self.transform_to_mat(transform)

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
                            transform.transform.translation.z])
        R[3, 3] = 1
        return R

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

    def vec_pixel_to_world(self, mat: np.array, depth: np.array) -> np.array:
        """
        A vectorized version of the above. `mat` represents a list of x,y,depth points
        """
        assert mat.shape[0] == depth.shape[0]

        world = np.zeros((mat.shape[0], 3))

        world[:, 0] = (depth / self.fx) * (mat[:, 0] - self.cx)    # world x
        world[:, 1] = (depth / self.fy) * (mat[:, 1] - self.cy)    # world y
        world[:, 2] = depth                                        # world z
        return world

    def create_3d_world(self, depth_img_cv: np.array, segmentation:bool) -> np.array:
        """
        Use human-annotated bboxes to render 3d bounding prisms of objects. Note: segmentation
        parameter represents the fraction of coordinates to segment. If not provided, the whole
        depth image is converted to XYZ space which in this code is used to create XYZ.npy files.
        For this reason, we don't append our XYZ coords to the world in this case.
        """
        depth_map = depth_img_cv / 1000.0    # mm -> m
        if segmentation:
            segmentation_coords = self.segmentation
        else:
            segmentation_coords = np.where(depth_img_cv != None)

        # here row values correspond to y, column values correspond to x so we must invert
        # only add objects to world when we're not computing full depth maps
        depth_values = depth_map[segmentation_coords]
        reshape_coords = np.array(
            list(zip(segmentation_coords[1], segmentation_coords[0])))
        world = self.vec_pixel_to_world(reshape_coords, depth_values)
 
        if segmentation:
            self.objects.append(world)
            self.world_created = True
        return world

    def compute_projection(self, transform_mat: np.array = None) -> np.array:
        """
        Compute projections from 3D back to 2D based on a transformation matrix
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
        return projection[:, 0, :]    # ignore nonexistent y coord

    def create_projection_image(self, color_img: np.array, projection: np.array):
        """
        Create a new image with our projection overlayed on top 
        """
        # bound the projection to our width and height
        canvas = np.zeros((self.height, self.width))
        bounded_projection = projection[
            np.logical_and(np.abs(projection[:, 0]) < self.width, np.abs(
                projection[:, 1]) < self.height)
        ]
        canvas[(bounded_projection[:, 1].astype('int'),
                bounded_projection[:, 0].astype('int'))] = 1

        # mask our projection coordinates with the fork mask
        masked_projection = cv2.bitwise_and(canvas, self.fork_mask)
        valid_projection = np.where(masked_projection == 1)

        segmentation_image = np.copy(color_img)
        segmentation_image[valid_projection] = (255, 0, 0)

        return cv2.addWeighted(segmentation_image, 0.5, color_img, 0.5, 1.0)

    def save_3d_config(self, depth_img_cv:np.array, transform:np.array) -> None:
        """
        Saves XYZ coords and a corresponding transform as .npy files for
        recalibration
        """
        xyz_coords = self.create_3d_world(depth_img_cv, segmentation=False)
        coords_name = "xyz_coords_callback_{}.npy".format(self.callback_counter)
        transform_name = "transform_calback_{}.npy".format(self.callback_counter)

        coords_save_location = self.base_dir / "data" / "3d" / coords_name
        transform_save_location = self.base_dir / "data" / "3d" / transform_name

        np.save(coords_save_location, xyz_coords)
        np.save(transform_save_location, transform)

    def callback(self, camera_info: CameraInfo, color_img: Image, depth_img: Image):
        """
        Hook function for info published to camera topics
        """
        callback_start_time = time.time()

        # populate tf buffer for first 50 callbacks
        if self.callback_counter < 50:
            self.callback_counter += 1
            return

        # if camera_info.header.stamp.nsecs != color_img.header.stamp.nsecs or color_img.header.stamp.nsecs != depth_img.header.stamp.nsecs:
        #     print("Misaligned frames! Press [ENTER] to continue:")
        #     lol = input()

        # print("camera_info.header: ",camera_info.header.stamp.secs, camera_info.header.stamp.nsecs)
        # print("color_img.header: ",color_img.header.stamp.secs, color_img.header.stamp.nsecs)
        # print("depth_img.header: ",depth_img.header.stamp.secs, depth_img.header.stamp.nsecs)
        # print("joint_state.header: ",joint_state.header.stamp.secs, joint_state.header.stamp.nsecs)

        # convert from ROS msgs to np arrays
        color_img_cv, depth_img_cv = self.convert_images(color_img, depth_img)

        # if True:
        #     print('here')
        #     res = cv2.matchTemplate(color_img_cv, self.template, cv2.TM_CCOEFF_NORMED)
        #     threshold = 0.8
        #     loc = np.where( res >= threshold)
        #     for pt in zip(*loc[::-1]):
        #         cv2.rectangle(color_img_cv, pt, (pt[0] + self.width, pt[1] + self.height), (0,0,255), 2)
        #     print('here2')
        #     thresh = 0.8
        #     print('here3')
        #     # res = np.where(res > thresh, 1, 0)
        #     print('here4')
        #     cv2.imshow('/camera_1/color/image_raw',color_img_cv)
        #     print('here5')
        #     cv2.waitKey(1)
        #     print('here6')


        #     return


        # get current base to camera transform
        base_to_camera_tf = self.get_base_to_camera_tf(
            camera_info.header.stamp)

        if not self.world_created:
            # initial callback: store current tf, create 3d world
            self.register_camera_info(camera_info)
            self.init_camera_tf = base_to_camera_tf
            self.create_3d_world(depth_img_cv, segmentation=True)
            projection = self.compute_projection()
        else:
            # regular callback: find transform to position
            init_camera_to_camera_tf = np.linalg.inv(
                base_to_camera_tf) @ self.init_camera_tf
            projection = self.compute_projection(init_camera_to_camera_tf)


        # avoid threading issues between OpenCV/ROS by copying within the lock
        final_image = self.create_projection_image(color_img_cv, projection)
        with image_lock:
            np.copyto(global_image, final_image)


        # create/publish a pointcloud
        if self.pointcloud_type == "segmented":
            self.create_pointcloud(
                color_img, color_img=final_image, depth_img=depth_img_cv)
        elif self.pointcloud_type == "raw":
            self.create_pointcloud(
                color_img, color_img=color_img_cv, depth_img=depth_img_cv)

        # save depth map if in list of indexes 
        # if self.callback_counter in self.save_callback_idxs:
        #     self.save_3d_config(depth_img_cv, init_camera_to_camera_tf)

        callback_end_time = time.time()
        print("Callback {} Time : {}".format(self.callback_counter, callback_end_time - callback_start_time))

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

    def tmp_get_segmentation(self):
        """
        Creates a segementation mask of all objects in the scene
        """
        blank = np.zeros((self.height, self.width))
        cv2.circle(blank, (694, 165), 47, 1, thickness=-
                   1)         # salami ad hoc for now

        with open(self.segmentation_file) as f:
            segmentations = json.load(f)['segmentations']
            for seg in segmentations:
                pts = np.array(seg['segmentation'])
                cv2.fillPoly(blank, [pts], 1)

        return np.where(blank == 1)
    
    # NOTE
    # tmp functions to try adding template matching to automatic segmentation algo!
    def tmp_salami_bbox(self):
        frame_0 = cv2.imread('../data/video/00000.png')
        salami_seg = frame_0[114:208, 635:745]
        return salami_seg

def display() -> None:
    """
    Display utility to decouple OpenCV image showing and ROS operations
    """
    local_image = np.zeros((720, 1280, 3), np.uint8)
    while not quit_event.is_set():
        with image_lock:
            np.copyto(local_image, global_image)
        try:
            cv2.imshow('test', local_image)
            cv2.waitKey(1)
        except:
            pass


def main():
    annotator = BBoxAnnotator()
    display_thread = threading.Thread(target=display)

    def signal_handler(sig, frame):
        quit_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    display_thread.start()
    annotator.run()

if __name__ == '__main__':
    main()
