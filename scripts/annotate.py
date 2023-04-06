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
import argparse
from subprocess import Popen

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


# configure global image and locks to avoid ROS/OpenCV threading issues
global_image = np.zeros((720,1280,3), dtype=np.uint8)
image_lock = threading.Lock()
quit_event = threading.Event()


class Annotator():
    """
    A class for automatically labeling images given a single semantic segmentation 
    """

    def __init__(self, bagfile_number, create_training_set, training_set_version=None, display=False, publish_pointcloud=False, image_dim=(720, 1280), n_classes=33):
        """
        Initialize Annotator module
        """
        rospy.init_node('annotator')

        # configure basics 
        self.bagfile_number = bagfile_number
        self.create_training_set = create_training_set
        self.training_set_version = training_set_version
        self.n_classes = n_classes
        self.base_dir, self.initial_segmentation, self.labeling_metadata, self.rosbag_path, self.id_to_color = self.init_filesystem()

        # init CV
        self.height, self.width = image_dim
        self.display = display 

        # segmentation of the scene
        self.food_segmentation = self.get_segmentation()

        # initial base to camera transform
        self.init_camera_tf = None

        # False for initial callback, True otherwise
        self.world_created = False
        self.publish_pointcloud = publish_pointcloud

        # storage for 3D bounding prisms in the scene
        self.objects = {i:None for i in range(1, self.n_classes+1)}   # for each object category `i`, store the point locations in 3d space

        # a binary image that masks the forque in the image
        self.fork_mask = self.create_fork_mask()

        # count number of callback invocations / set a timer on them
        self.timer = rospy.Timer(rospy.Duration(15), self.timer_callback)
        self.callback_counter = 0
        self.total_time = 0.0

        # spin up ROS
        self.rosbag_process = self.play_rosbag()
        self.last_msg_time = rospy.Time.now()
        self.tf_buffer = self.init_tf_buffer()
        self.init_ros_pub_and_sub()

    def run(self):
        """
        Run the annotator
        """
        rospy.spin()
    
    def timer_callback(self, event):
        """
        Once there are no new messages, clean up the program
        """
        # program has started (callback_counter >0) and we haven't recieved any new messages in 30s
        if self.callback_counter > 0 and rospy.Time.now() - self.last_msg_time > rospy.Duration(30):
            self.rosbag_process.kill()
            rospy.signal_shutdown('No new messages in 15 seconds')
            os.kill(os.getpid(), signal.SIGINT)

    def play_rosbag(self, supress=True):
        print('Playing Rosbag...')
        if supress:
            with open(os.devnull, 'w') as fp:
                proc = Popen(['rosbag', 'play', str(self.rosbag_path)], stdout=fp)
        else:   
            proc = Popen(['rosbag', 'play', str(self.rosbag_path)])
        return proc

    def get_segmentation(self):
        """
        Load in the initial segmentation file
        """
        return cv2.imread(self.initial_segmentation, cv2.IMREAD_GRAYSCALE)    # grayscale b/c we just have class integers

    def check_valid_experiment(self, labeling_metadata_fpath):
        """
        Uses the metadata JSON to determine if the provided bagfile has been annotated
        """
        with open(str(labeling_metadata_fpath), 'r') as f:
            metadata = json.load(f)['dataset']['samples']

        def nbr(x):
            return x.split('_')[-1].split('.')[0]

        valid = [nbr(x['name']) for x in metadata if x['labels']['ground-truth']['label_status'] == 'LABELED']
        return self.bagfile_number in valid 

    def init_filesystem(self):
        """
        Uses pathlib to get the `foodrecognition` base directory as well as other important files
        """
        base_dir = Path(__file__).absolute().parents[1]
        initial_segmentation = str(base_dir / "data" / "segmentations" / "processed" / "init_frame_bag_{}.png".format(self.bagfile_number))
        labeling_metadata_fpath = str(base_dir / "data" / "json" / "emprise-feeding-infra-ground-truth.json")
        rosbag_path = base_dir / "data" / "bagfile" / "experiment_{}.bag".format(self.bagfile_number)

        valid_experiment = self.check_valid_experiment(labeling_metadata_fpath)
        if not valid_experiment:
            print('Bagfile or segmentation file missing... exiting')
            os.kill(os.getpid(), signal.SIGINT)

        print('Filesystem is good, starting collection')

        # load .json files for labeling meta data
        with open(labeling_metadata_fpath, 'r') as f:
            labeling_metadata = json.load(f)
        # load id->color .json file
        with open(str(base_dir / "data" / "json" / "id_to_color.json"), 'r') as f:
            id_to_color = {int(k):v for k,v in json.load(f).items()}    # keys are str by default
            id_to_color[0] = [0,0,0]

        return base_dir, initial_segmentation, labeling_metadata, rosbag_path, id_to_color 

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

        self.pointcloud_init_pub = rospy.Publisher(
            '/generated/init_frame/pointcloud2', PointCloud2, queue_size=10)
        self.pointcloud_current_pub = rospy.Publisher(
            '/generated/current_frame/pointcloud2', PointCloud2, queue_size=10)

    def init_tf_buffer(self):
        """
        Initialize buffer to listen to /tf and /tf_static
        """
        tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        tf2_ros.TransformListener(tf_buffer)
        return tf_buffer

    def create_fork_mask(self):
        """
        Creates a mask image represeting the fork in the frame
        """
        white = np.ones((self.height, self.width))
        fork_outline = np.array([
            [714, 717], [707, 640], [733, 636], [724, 534],
            [778, 534], [785, 567], [815, 577], [865, 637],
            [987, 635], [988, 657], [1059, 718]
        ])
        cv2.fillPoly(white, [fork_outline], 0)
        return white

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

        # end_effector_to_camera[:3, 3] = np.array([0.014, 0.060, 0.034])   # BEST VALUES
        # end_effector_to_camera[:3, 3] = np.array([0.0185, 0.058, 0.034])    # RAJAT VALS
        # # old camera calibration params
        # end_effector_to_camera[:3, :3] = Rotation.from_quat([0.50080661, 0.49902445, 0.50068027, -0.49948635]).as_matrix()

        camera_link_to_color_optical = self.transform_to_mat(
            camera_link_to_color_optical_transform)

        return base_to_ee @ end_effector_to_camera @ camera_link_to_color_optical

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

    def create_3d_world(self, depth_img_cv: np.array, segmentation:np.array, create_world:bool=False) -> np.array:
        """
        Use human-annotated bboxes to render 3d bounding prisms of objects. Note: segmentation
        parameter represents the fraction of coordinates to segment. If not provided, the whole
        depth image is converted to XYZ space which in this code is used to create XYZ.npy files.
        For this reason, we don't append our XYZ coords to the world in this case.
        """
        depth_map = depth_img_cv / 1000.0    # mm -> m

        # 2D segmentation has number between 0 and n_classes at each (i,j) index
        # Need to separate out into discrete segmentations for each object class so that we can
        # track how individual objects move within the scene
        present_category_ids = []
        for i in range(1, self.n_classes+1):
            subsegmentation = np.where(segmentation == i)
            if len(subsegmentation[0]) > 0:    # if we actually have pixels corresponding to this category id
                present_category_ids.append(i)
                depth_values = depth_map[subsegmentation]
                reshape_coords = np.array(
                    list(zip(subsegmentation[1], subsegmentation[0])))
                subsegmentation_world = self.vec_pixel_to_world(reshape_coords, depth_values)
                self.objects[i] = subsegmentation_world

        self.world_created = True
        self.present_category_ids = present_category_ids    # keep track of which of the 33 classes are actually in the scene

    def compute_projection(self, points:np.array, transform_mat: np.array = None) -> np.array:
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
            points, 
            rotation_vec,
            translation_vec,
            camera_mat,
            None
        )
        return projection[:, 0, :]    # ignore nonexistent y coord

    def create_projected_mask(self, transform_mat:np.array=None):
        """
        Use the transformation matrix for this callback to create an updated mask of the scene
        TODO: Find some way to model better model occlusion? Apply subsegmentations by mean y-val? Get sorted list of mean z-val?
        """
        # for each object in our `objects` dictionary, compute_projection(). output will be a dict mapping i -> 2d points
        projections = {i:self.compute_projection(self.objects[i], transform_mat) for i in self.present_category_ids}
        canvas = np.zeros((self.height, self.width))
        for category_id in self.present_category_ids: 
            projection = projections[category_id]
            bounded_projection = projection[
                np.logical_and(np.abs(projection[:, 0]) < self.width, np.abs(
                    projection[:, 1]) < self.height)
            ]
            canvas[(bounded_projection[:, 1].astype('int'),
                    bounded_projection[:, 0].astype('int'))] = category_id    # want to fill our mask image with integers representing the category id
            
        mask = canvas * self.fork_mask    # here we take our 2D integer array and multiply it by the boolean mask array
        return mask

    def save_images(self, color_img, mask_img) -> None:
        """
        Write our color image and mask image to the appropriate place in the training directory
        """
        training_dir = self.base_dir / 'data' / 'training' / 'v{}'.format(self.training_set_version)
        # img_dir, mask_dir = training_dir / "images", training_dir / "masks"
        bag_dir = training_dir / "bag_{}".format(self.bagfile_number)
        if not os.path.exists(str(bag_dir)):
            os.mkdir(str(bag_dir))
        img_fpath = str(bag_dir / 'bag_{}_callback_{}_img.png'.format(self.bagfile_number, self.callback_counter)) 
        mask_fpath = str(bag_dir / 'bag_{}_callback_{}_mask.png'.format(self.bagfile_number, self.callback_counter))
        cv2.imwrite(img_fpath, color_img)
        cv2.imwrite(mask_fpath, mask_img)

    def create_projection_image(self, color_img: np.array, projected_mask: np.array):
        """
        Create a new image with our projection overlayed on top 
        """
        b, g, r = [{k:v[i] for k,v in self.id_to_color.items()} for i in range(3)]
        f_b, f_g, f_r = [np.vectorize(x.get) for x in [b,g,r]]
        b_layer, g_layer, r_layer = [f(projected_mask).astype('uint8') for f in [f_b, f_g, f_r]]
        color_mapped_mask = np.dstack((b_layer, g_layer, r_layer))
        return cv2.addWeighted(color_mapped_mask, 0.5, color_img, 0.5, 1.0)

    def callback(self, camera_info: CameraInfo, color_img: Image, depth_img: Image):
        """
        Hook function for info published to camera topics
        """
        callback_start_time = time.time()
        self.last_msg_time = rospy.Time.now()

        # populate tf buffer for first 50 callbacks
        if self.callback_counter < 50:
            self.callback_counter += 1
            callback_end_time = time.time()
            callback_time = float(callback_end_time - callback_start_time)
            self.total_time += callback_time
            return

        # convert from ROS msgs to np arrays
        color_img_cv, depth_img_cv = self.convert_images(color_img, depth_img)

        # get current base to camera transform
        base_to_camera_tf = self.get_base_to_camera_tf(
            camera_info.header.stamp)

        if not self.world_created:
            # initial callback: store current tf, create 3d world
            self.register_camera_info(camera_info)
            self.init_camera_tf = base_to_camera_tf
            self.create_3d_world(depth_img_cv, segmentation=self.food_segmentation, create_world=True)
            projected_mask = self.create_projected_mask()
        else:
            # regular callback: find transform to position
            init_camera_to_current_camera_tf = np.linalg.inv(
                base_to_camera_tf) @ self.init_camera_tf
            projected_mask = self.create_projected_mask(transform_mat=init_camera_to_current_camera_tf)

        # for training, save both the color image and the projected mask
        if self.create_training_set:
            self.save_images(color_img_cv, projected_mask)

        # if display, create a superimposition of img and mask
        if self.display:
            # avoid threading issues between OpenCV/ROS by copying within the lock
            display_image = self.create_projection_image(color_img_cv, projected_mask)
            with image_lock:
                np.copyto(global_image, display_image)

        # TODO: Clean up PC artifacts
        # create/publish a pointcloud
        # if self.callback_counter == 50:
        #     self.storage['color'] = color_img_cv
        #     self.storage['depth'] = depth_img_cv
        # if self.publish_pointcloud == True and self.callback_counter % 5 == 0 and self.callback_counter != 50:
        #     init_pc = self.create_pointcloud(color_img, self.storage['color'], self.storage['depth'], \
        #                         transform=self.init_camera_tf, topic='/generated/init_frame/pointcloud2')
        #     current_pc = self.create_pointcloud(color_img, color_img_cv, depth_img_cv, \
        #                         transform=base_to_camera_tf, topic='/generated/current_frame/pointcloud2')

        #     self.pointcloud_init_pub.publish(init_pc)
        #     self.pointcloud_current_pub.publish(current_pc)


        # log appropriate callback variables
        callback_end_time = time.time()
        callback_time = float(callback_end_time - callback_start_time)
        self.total_time += callback_time
        self.callback_counter += 1
        mean_time = self.total_time / self.callback_counter
        print(f"BAG {self.bagfile_number} :: CALLBACK {self.callback_counter:04d} :: TIME {callback_time:.5f}")
        

    def create_pointcloud(self, msg, color_img: np.array, depth_img: np.array, transform:np.array) -> None:
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
                    # pt = [x, y, z, 0]
                    pt = np.array([x,y,z,1]).T 
                    tf_pt = list(transform @ pt).T

                    b, g, r, a = color[0], color[1], color[2], 255
                    rgb = struct.unpack(
                        'I', struct.pack('BBBB', b, g, r, a))[0]
                    tf_pt[3] = rgb
                    points.append(tf_pt)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        pc2 = point_cloud2.create_cloud(header, fields, points)
        return pc2
    

def display() -> None:
    """
    Display utility to decouple OpenCV image showing and ROS operations
    """
    local_image = np.zeros((720, 1280, 3), np.uint8)
    while not quit_event.is_set():
        with image_lock:
            np.copyto(local_image, global_image)
        try:
            cv2.imshow('display', local_image)
            cv2.waitKey(1)
        except:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile_number")
    parser.add_argument("-v", "--version", default=None)    # useless if --create-training-set is False
    parser.add_argument("--create-training-set", dest='create_training_set', choices=["True", "False"], default="True")
    parser.add_argument("-d", "--display", choices=["True", "False"], default="False")
    args = parser.parse_args()

    annotator = Annotator(
        bagfile_number=args.bagfile_number,
        training_set_version=args.version,
        create_training_set=args.create_training_set == "True",
        display=args.display == "True"
    )
    display_thread = threading.Thread(target=display)

    # set up display thread to die on exit as well
    def signal_handler(sig, frame):
        quit_event.set()
        annotator.rosbag_process.kill()
        rospy.signal_shutdown("")
        sys.exit("")
    signal.signal(signal.SIGINT, signal_handler)

    if args.display == "True": display_thread.start()
    annotator.run()

if __name__ == '__main__':
    main()
