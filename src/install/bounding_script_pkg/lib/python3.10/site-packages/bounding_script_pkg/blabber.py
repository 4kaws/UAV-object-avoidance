#!/usr/bin/env python3
import os
import csv
import time
import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from datetime import datetime
from rclpy.executors import SingleThreadedExecutor  # Make sure this line is added

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(Image, '/drone/front/image_raw', self.image_callback, 10)
        self.pose_subscription = self.create_subscription(PoseStamped, '/drone/gt_pose', self.gt_pose_callback, 10)
        self.bridge = CvBridge()
        self.last_save_time = time.time()
        self.save_interval = 5  # seconds
        self.image_folder = '/home/andrei/ros2_ws/images'
        self.csv_file_path = '/home/andrei/ros2_ws/dataset.csv'
        os.makedirs(self.image_folder, exist_ok=True)
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["timestamp", "image_name", "object_name", "x", "y", "z", "rx", "ry", "rz", "rw", "lx", "ly", "lz",
             "x_min_bb", "y_min_bb", "x_max_bb", "y_max_bb"])
        self.image_counter = 0
        self.objects_data = {
            "Dumpster": {
                "pose": [1.10581, -9.13614, 0.051381],
                "euler_orientation": [-1.1e-05, 3e-06, -3.13677],
                "size": [2.66, 1.33, 1.03],
                "scale": [1.5, 1.5, 1.5],
            },
            "Control Console": {
                "pose": [0.044032, 9.47148, 0],
                "euler_orientation": [0, 0, 0],
                "size": [1.6818342, 1.000181, 2.625193],
                "scale": [1, 1, 1],
            },
            "Construction Cone": {
                "pose": [3.44074, 0.667633, 0.049998],
                "euler_orientation": [0, -2e-06, -0.004192],
                "size": [1.98, 1.98, 4.286964],
                "scale": [10, 10, 10],
            },
            "number1": {
                "pose": [-3.44359, -9.53926, 0.4],
                "euler_orientation": [0, 0, 0],  # Placeholder, adjust as needed
                "size": [2, 2.000002, 2],
                "scale": [1, 1, 1],
            },
            "number2": {
                "pose": [9.34806, 5.92857, 0.4],
                "euler_orientation": [0, 0, 0],  # Placeholder, adjust as needed
                "size": [2, 2.000002, 2],
                "scale": [1, 1, 1],
            },
            "number3": {
                "pose": [4.34527, 9.36089, 0.4],
                "euler_orientation": [0, 0, 0],  # Placeholder, adjust as needed
                "size": [2, 2.000002, 2],
                "scale": [1, 1, 1],
            },
            "number4": {
                "pose": [9.51457, -4.51109, 0.4],
                "euler_orientation": [0, 0, 0],  # Placeholder, adjust as needed
                "size": [2, 2.000002, 2],
                "scale": [1, 1, 1],
            },
            "person_walking": {
                "pose": [-7.531469, 5.773241, 0.049999],
                "euler_orientation": [0, 0, 0],  # Placeholder, adjust as needed
                "size": [0.5427376, 0.8829857, 1.86117434],
                "scale": [1, 1, 1],
            },
            "person_standing": {
                "pose": [0.739916, -7.874418, 0.05],
                "euler_orientation": [0, 0, 0],  # Placeholder, adjust as needed
                "size": [0.5442286, 0.32510104, 1.90364231],
                "scale": [1, 1, 1],
            },
        }

        # Conversion from Euler to Quaternion for each object
        for obj_id, obj_data in self.objects_data.items():
            euler_orientation = obj_data.get('euler_orientation', [0, 0, 0])  # Default to no rotation if not provided
            quaternion_orientation = self.euler_to_quaternion(*euler_orientation)  # Convert Euler angles to quaternion
            obj_data['orientation'] = quaternion_orientation  # Update object data with quaternion orientation
            # Optional: Remove 'euler_orientation' if it's no longer needed
            # del obj_data['euler_orientation']

        # Camera intrinsic parameters
        self.fov_horizontal = 2.09  # This is presumably in radians
        self.image_width = 640
        self.image_height = 360
        # Assuming these are pixels, the following calculations should be fine
        self.focal_length_x = self.image_width / (2 * np.tan(self.fov_horizontal / 2))
        self.focal_length_y = self.focal_length_x
        self.c_x = self.image_width / 2
        self.c_y = self.image_height / 2
        self.K = np.array([[self.focal_length_x, 0, self.c_x], [0, self.focal_length_y, self.c_y], [0, 0, 1]])

        # Initialize extrinsic parameters
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.update_camera_projection_matrix()

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion.

        Parameters:
        roll (float): Rotation around the x-axis in radians.
        pitch (float): Rotation around the y-axis in radians.
        yaw (float): Rotation around the z-axis in radians.

        Returns:
        tuple: Quaternion (x, y, z, w)
        """
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(
            pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(
            pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(
            pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(
            pitch / 2) * math.sin(yaw / 2)

        return (qx, qy, qz, qw)

    def quaternion_to_rotation_matrix(self, quaternion):
        # NumPy-based quaternion to rotation matrix conversion
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < np.finfo(q.dtype).eps:
            return np.eye(3)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]
        ])

    def update_camera_projection_matrix(self):
        # Update the camera projection matrix
        self.camera_projection_matrix = np.dot(self.K, np.hstack((self.R, self.t)))

    def gt_pose_callback(self, msg):
        # Extract pose data from the message
        position = msg.pose.position
        orientation = msg.pose.orientation
        print(f"Drone Position: {position.x}, {position.y}, {position.z}")
        print(f"Drone Orientation: {orientation.x}, {orientation.y}, {orientation.z}, {orientation.w}")

        # Convert quaternion to rotation matrix
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.R = self.quaternion_to_rotation_matrix(quaternion)

        # Update the translation vector
        self.t = np.array([[position.x], [position.y], [position.z]])

    def calc(self):
        # get camera pos and orientation
        self.R = np.eye(3)
        self.t = np.array([[0, 0, 0]]).T
        self.camera_projection_matrix = np.dot(self.K, np.hstack((self.R, self.t)))

    def project_to_image(self, world_point):
        # Project a 3D point in world coordinates onto the 2D image plane.
        world_point_homogeneous = np.append(world_point, 1)
        image_point_homogeneous = np.dot(self.camera_projection_matrix, world_point_homogeneous)
        image_point_homogeneous /= image_point_homogeneous[2]  # Normalize
        u, v = image_point_homogeneous[:2]
        return u, v

    def image_callback(self, msg):
        print("Entered image_callback")
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                bounding_boxes = []
                for object_id, data in self.objects_data.items():
                    bbox = self.calculate_bounding_box(object_id)
                    if bbox:

                        bounding_boxes.append((object_id, bbox))

                if bounding_boxes:
                    # Make sure to calculate and verify bounding boxes first
                    for object_id in self.objects_data:
                        bbox = self.calculate_bounding_box(object_id)
                        if bbox and self.is_bounding_box_in_image(bbox):
                            bounding_boxes.append((object_id, bbox))
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                            image_name = f'image_{timestamp}.png'
                            image_path = os.path.join(self.image_folder, image_name)
                            cv2.imwrite(image_path, cv_image)
                            self.save_object_data(image_name, bounding_boxes, timestamp)
                            self.image_counter += 1
                            self.last_save_time = current_time

            except CvBridgeError as e:
                print(f"CVBridgeError: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def calculate_bounding_box(self, object_id):
        data = self.objects_data[object_id]
        pose = np.array(data["pose"])
        orientation = data.get("orientation", [0, 0, 0, 1])  # Default orientation if not provided
        size = np.array(data["size"])
        scale = np.array(data["scale"])
        size_scaled = size * scale

        # Calculate object corners in the object's local coordinate system
        half_size = size_scaled / 2
        corners_local = [np.array([sx, sy, sz]) for sx in [-half_size[0], half_size[0]]
                         for sy in [-half_size[1], half_size[1]] for sz in [-half_size[2], half_size[2]]]

        # Transform corners to the global coordinate system
        R_obj = self.quaternion_to_rotation_matrix(orientation)
        corners_global = [pose + np.dot(R_obj, corner) for corner in corners_local]

        # Project corners onto the image plane
        projected_corners = [self.project_to_image(corner) for corner in corners_global]

        # Filter out None values and points outside of the image frame
        projected_corners = [p for p in projected_corners if p is not None and
                             0 <= p[0] < self.image_width and 0 <= p[1] < self.image_height]


        # Calculate bounding box from projected corners
        if projected_corners:
            x_min, y_min = np.min(projected_corners, axis=0)
            x_max, y_max = np.max(projected_corners, axis=0)
            return [x_min, y_min, x_max, y_max]
        else:
            return None  # Object is not visible

    def is_bounding_box_in_image(self, bounding_box):
        x_min, y_min, x_max, y_max = bounding_box
        return (0 <= x_min < self.image_width and 0 <= x_max < self.image_width and
                0 <= y_min < self.image_height and 0 <= y_max < self.image_height)

    def save_object_data(self, image_name, bounding_boxes, timestamp):
        for object_id, bounding_box in bounding_boxes:
            print(f"Saving Bounding Box for Object ID: {object_id}, Bounding Box: {bounding_box}")
            # If the bounding box is in the image, save the data
            if self.is_bounding_box_in_image(bounding_box):
                x_min, y_min, x_max, y_max = bounding_box
                data = self.objects_data[object_id]
                pose = data['pose']
                size = data['size']
                scale = data['scale']
                self.csv_writer.writerow([
                    timestamp,
                    image_name,
                    object_id,
                    pose[0], pose[1], pose[2],
                    size[0], size[1], size[2],
                    scale[0], scale[1], scale[2],
                    x_min, y_min, x_max, y_max
                ])

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()


def main(args=None):
    print('GOGO')
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    executor = SingleThreadedExecutor()

    try:
        executor.add_node(image_subscriber)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Safe shutdown process
        if rclpy.ok():
            image_subscriber.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

##############################################################################################    V2

import os
import csv
import time
import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(Image, '/drone/front/image_raw', self.image_callback, 10)
        self.pose_subscription = self.create_subscription(PoseStamped, '/drone/gt_pose', self.gt_pose_callback, 10)
        self.bridge = CvBridge()
        self.last_save_time = time.time()
        self.save_interval = 5  # seconds
        self.image_folder = '/home/andrei/ros2_ws/images'
        self.csv_file_path = '/home/andrei/ros2_ws/dataset.csv'
        os.makedirs(self.image_folder, exist_ok=True)
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["timestamp", "image_name", "object_name", "x", "y", "z", "roll", "pitch", "yaw", "size_x", "size_y",
             "size_z",
             "x_min_bb", "y_min_bb", "x_max_bb", "y_max_bb"])
        self.image_counter = 0
        self.objects_data = {
            "asphalt_plane": {
                "pose": [0.027118, -0.028406, 0],
                "orientation": [0, -0, 0],
                "size": [20.000000, 20.000000, 0.100000],
            },
            "control_console": {
                "pose": [0.044032, 9.471480, 0],
                "orientation": [0, 0, 0],
                "size": [1.780000, 0.050000, 2.602250],
            },

            "unit_box": {
                "pose": [9.406795, -9.399209, 0.550000],
                "orientation": [0, 0, 0.000034],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone": {
                "pose": [9.424715, 9.274771, 0.550000],
                "orientation": [0, 0, 0.000040],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone_clone": {
                "pose": [-9.261930, 9.274990, 0.550000],
                "orientation": [0, 0, 0.000016],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone_clone_clone": {
                "pose": [-8.978432, -9.372633, 0.549995],
                "orientation": [-0.000010, 0.000001, 0.111119],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number1": {
                "pose": [-3.443590, -9.539260, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number2": {
                "pose": [-9.348060, 5.928570, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number3": {
                "pose": [4.345270, 9.360890, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number4": {
                "pose": [9.514570, -4.511090, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "Construction Cone": {
                "pose": [3.440740, 0.667633, 0.049999],
                "orientation": [-0.000002, 0.000002, -0.004192],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_0": {
                "pose": [2.936450, 0.915504, 0.050000],
                "orientation": [0, 0, -0.001118],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_1": {
                "pose": [2.419270, 1.061460, 0.049991],
                "orientation": [0, -0.000002, 0.000614],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_2": {
                "pose": [1.916210, 1.225750, 0.050000],
                "orientation": [0, 0, -0.000372],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_3": {
                "pose": [1.412490, 1.358500, 0.049991],
                "orientation": [0, 0, -0.014740],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_4": {
                "pose": [0.906406, 1.475050, 0.050000],
                "orientation": [0, 0, 0.002556],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_5": {
                "pose": [0.377529, 1.550320, 0.050001],
                "orientation": [0, -0.000002, -0.000256],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_6": {
                "pose": [-0.141692, 1.602430, 0.050000],
                "orientation": [0, 0, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_7": {
                "pose": [-0.669365, 1.771780, 0.049990],
                "orientation": [-0.000002, 0, -0.000256],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_8": {
                "pose": [-1.185310, 1.961130, 0.050000],
                "orientation": [0.000002, -0.000002, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_9": {
                "pose": [-1.720400, 2.194210, 0.049992],
                "orientation": [0, 0.000002, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_10": {
                "pose": [-2.251430, 2.447520, 0.050000],
                "orientation": [0, 0, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Dumpster": {
                "pose": [1.105810, -9.136140, 0.051370],
                "orientation": [0.000010, 0, -3.136771],
                "size": [1.773333, 0.886666, 0.686666],
            },
            "hoop_red": {
                "pose": [-3.397180, -10.136400, 0],
                "orientation": [0, 0, 0],
                "size": [0.300000, 0.100000, 5.000000],
            },
            "hoop_red_0": {
                "pose": [-9.971850, 3.162810, 0],
                "orientation": [0, 0, -1.556320],
                "size": [0.300000, 0.100000, 5.000000],
            },
            "person_standing": {
                "pose": [0.739916, -7.874418, 0.050000],
                "orientation": [0, -0.000002, 0.001970],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_walking": {
                "pose": [-7.531469, 5.773241, 0.049999],
                "orientation": [0, 0.000006, 0.005452],
                "size": [0.350000, 0.750000, 0.020000],
            },
            "person_standing_0": {
                "pose": [0.592537, 8.466297, 0.050000],
                "orientation": [0, -0.000002, -3.140959],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_1": {
                "pose": [-0.195836, 2.403852, 0.050000],
                "orientation": [0, 0.000002, 0.001718],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_2": {
                "pose": [-9.561253, -8.743303, 0.294097],
                "orientation": [-1.636140, 0.909863, 0.015181],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_3": {
                "pose": [1.827380, 6.648900, 0.000537],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_4": {
                "pose": [-2.140890, 5.534660, 0.002202],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_5": {
                "pose": [-2.494090, 3.011030, -0.007182],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_6": {
                "pose": [4.132250, 5.527540, 0.007169],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_7": {
                "pose": [-3.036670, -5.568780, -0.009657],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
        }

        # Camera intrinsic parameters
        self.fov_horizontal = 2.09  # This is presumably in radians
        self.image_width = 640
        self.image_height = 360
        self.focal_length_x = self.image_width / (2 * np.tan(self.fov_horizontal / 2))
        self.focal_length_y = self.focal_length_x
        self.c_x = self.image_width / 2
        self.c_y = self.image_height / 2
        self.K = np.array([[self.focal_length_x, 0, self.c_x], [0, self.focal_length_y, self.c_y], [0, 0, 1]])

        # Initialize extrinsic parameters
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.R_with_translation = None  # Initialize R_with_translation to None
        self.update_camera_projection_matrix()  # Update camera projection matrix

    def gt_pose_callback(self, msg):
        # Extract pose data from the message
        position = msg.pose.position
        orientation = msg.pose.orientation
        print(f"Drone Position: {position.x}, {position.y}, {position.z}")
        print(f"Drone Orientation: {orientation.x}, {orientation.y}, {orientation.z}, {orientation.w}")

        # Use the provided Euler angles directly
        roll, pitch, yaw = orientation.x, orientation.y, orientation.z
        self.R = self.euler_to_rotation_matrix(roll, pitch, yaw)

        # Update the translation vector
        self.t = np.array([[position.x], [position.y], [position.z]])

        # Update camera projection matrix after extrinsic parameters change
        self.update_camera_projection_matrix()

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        Convert Euler angles to rotation matrix.
        """
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cr = math.cos(roll)
        sr = math.sin(roll)

        # Rotation matrix from Euler angles
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        return R

    def calculate_R_with_translation(self):
        # Combine rotation matrix with translation vector
        self.R_with_translation = np.hstack((self.R, self.t))

    def update_camera_projection_matrix(self):
        # Check if R_with_translation is properly initialized
        if self.R_with_translation is not None:
            # Calculate R_with_translation
            self.calculate_R_with_translation()

            # Check if the dimensions of K and R_with_translation are compatible
            if self.K.shape[1] == self.R_with_translation.shape[0]:
                # Calculate camera projection matrix
                self.camera_projection_matrix = np.dot(self.K, self.R_with_translation.T)
                print("Camera projection matrix updated successfully.")
            else:
                print("Dimension mismatch between K and R_with_translation.")
        else:
            print("R_with_translation is not properly initialized.")

    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                # Draw points on the objects
                self.draw_points(cv_image)

                # Display the image with the drawn points
                cv2.imshow('Objects with Points', cv_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except CvBridgeError as e:
                print(f"CVBridgeError: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def draw_points(self, image):
        # Iterate through objects in the dictionary
        for object_id, data in self.objects_data.items():
            # Convert 3D position to homogeneous coordinates
            pose = np.array(data["pose"])
            pose_homogeneous = np.hstack((pose, [[1]]))

            # Project 3D point to 2D image coordinates
            projected_point = np.dot(self.camera_projection_matrix, pose_homogeneous)
            image_coordinates = (
                int(projected_point[0] / projected_point[2]), int(projected_point[1] / projected_point[2]))

            # Draw point on the image
            cv2.circle(image, image_coordinates, radius=5, color=(0, 255, 0),
                       thickness=-1)  # Green color, filled circle

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

def main(args=None):
    print('GOGO')
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    executor = SingleThreadedExecutor()
    executor.add_node(image_subscriber)
    executor.spin()
    executor.shutdown()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


----------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import csv
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R


def euler_to_rotation_matrix(roll, pitch, yaw):
    # Calculate rotation matrix from Euler angles
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_processor')
        # Define a custom QoS profile with the correct access to reliability and durability
        custom_qos_profile = QoSProfile(
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)

        # Use the custom QoS profile for the subscription
        self.subscription = self.create_subscription(
            Image,
            '/drone/front/image_raw',
            self.image_callback,
            custom_qos_profile)
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/drone/gt_pose',
            self.gt_pose_callback,
            custom_qos_profile)
        self.create_subscription(Image, '/drone/front/image_raw', self.image_callback, custom_qos_profile)
        self.bridge = CvBridge()
        self.last_save_time = time.time()
        self.save_interval = 5  # seconds
        self.image_folder = '/home/andrei/ros2_ws/images'
        self.csv_file_path = '/home/andrei/ros2_ws/dataset.csv'
        os.makedirs(self.image_folder, exist_ok=True)
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["timestamp", "image_name", "object_name", "x", "y", "z", "roll", "pitch", "yaw", "size_x", "size_y",
             "size_z", "x_min_bb", "y_min_bb", "x_max_bb", "y_max_bb"])
        self.image_counter = 0
        self.objects_data = {
            "asphalt_plane": {
                "pose": [0.027118, -0.028406, 0],
                "orientation": [0, -0, 0],
                "size": [20.000000, 20.000000, 0.100000],
            },
            "control_console": {
                "pose": [0.044032, 9.471480, 0],
                "orientation": [0, 0, 0],
                "size": [1.780000, 0.050000, 2.602250],
            },

            "unit_box": {
                "pose": [9.406795, -9.399209, 0.550000],
                "orientation": [0, 0, 0.000034],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone": {
                "pose": [9.424715, 9.274771, 0.550000],
                "orientation": [0, 0, 0.000040],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone_clone": {
                "pose": [-9.261930, 9.274990, 0.550000],
                "orientation": [0, 0, 0.000016],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "unit_box_clone_clone_clone": {
                "pose": [-8.978432, -9.372633, 0.549995],
                "orientation": [-0.000010, 0.000001, 0.111119],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number1": {
                "pose": [-3.443590, -9.539260, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number2": {
                "pose": [-9.348060, 5.928570, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number3": {
                "pose": [4.345270, 9.360890, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "number4": {
                "pose": [9.514570, -4.511090, 0.400000],
                "orientation": [0, 0, 0],
                "size": [1.000000, 1.000000, 1.000000],
            },
            "Construction Cone": {
                "pose": [3.440740, 0.667633, 0.049999],
                "orientation": [-0.000002, 0.000002, -0.004192],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_0": {
                "pose": [2.936450, 0.915504, 0.050000],
                "orientation": [0, 0, -0.001118],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_1": {
                "pose": [2.419270, 1.061460, 0.049991],
                "orientation": [0, -0.000002, 0.000614],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_2": {
                "pose": [1.916210, 1.225750, 0.050000],
                "orientation": [0, 0, -0.000372],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_3": {
                "pose": [1.412490, 1.358500, 0.049991],
                "orientation": [0, 0, -0.014740],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_4": {
                "pose": [0.906406, 1.475050, 0.050000],
                "orientation": [0, 0, 0.002556],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_5": {
                "pose": [0.377529, 1.550320, 0.050001],
                "orientation": [0, -0.000002, -0.000256],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_6": {
                "pose": [-0.141692, 1.602430, 0.050000],
                "orientation": [0, 0, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_7": {
                "pose": [-0.669365, 1.771780, 0.049990],
                "orientation": [-0.000002, 0, -0.000256],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_8": {
                "pose": [-1.185310, 1.961130, 0.050000],
                "orientation": [0.000002, -0.000002, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_9": {
                "pose": [-1.720400, 2.194210, 0.049992],
                "orientation": [0, 0.000002, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Construction Cone_10": {
                "pose": [-2.251430, 2.447520, 0.050000],
                "orientation": [0, 0, 0],
                "size": [0.198, 0.198, 0.428696],
            },
            "Dumpster": {
                "pose": [1.105810, -9.136140, 0.051370],
                "orientation": [0.000010, 0, -3.136771],
                "size": [1.773333, 0.886666, 0.686666],
            },
            "hoop_red": {
                "pose": [-3.397180, -10.136400, 0],
                "orientation": [0, 0, 0],
                "size": [0.300000, 0.100000, 5.000000],
            },
            "hoop_red_0": {
                "pose": [-9.971850, 3.162810, 0],
                "orientation": [0, 0, -1.556320],
                "size": [0.300000, 0.100000, 5.000000],
            },
            "person_standing": {
                "pose": [0.739916, -7.874418, 0.050000],
                "orientation": [0, -0.000002, 0.001970],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_walking": {
                "pose": [-7.531469, 5.773241, 0.049999],
                "orientation": [0, 0.000006, 0.005452],
                "size": [0.350000, 0.750000, 0.020000],
            },
            "person_standing_0": {
                "pose": [0.592537, 8.466297, 0.050000],
                "orientation": [0, -0.000002, -3.140959],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_1": {
                "pose": [-0.195836, 2.403852, 0.050000],
                "orientation": [0, 0.000002, 0.001718],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_2": {
                "pose": [-9.561253, -8.743303, 0.294097],
                "orientation": [-1.636140, 0.909863, 0.015181],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_3": {
                "pose": [1.827380, 6.648900, 0.000537],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_4": {
                "pose": [-2.140890, 5.534660, 0.002202],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_5": {
                "pose": [-2.494090, 3.011030, -0.007182],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_6": {
                "pose": [4.132250, 5.527540, 0.007169],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
            "person_standing_7": {
                "pose": [-3.036670, -5.568780, -0.009657],
                "orientation": [0, 0, 0],
                "size": [0.500000, 0.350000, 0.020000],
            },
        }

        #Camera parameters
        self.horizontal_fov = 2.09  # Horizontal field of view from URDF
        self.image_width = 640  # Image width
        self.image_height = 360  # Image height
        self.focal_length_x = self.image_width / (2 * np.tan(self.horizontal_fov / 2))
        self.focal_length_y = self.focal_length_x  # Assuming square pixels for simplicity
        self.c_x = self.image_width / 2
        self.c_y = self.image_height / 2

        # Camera intrinsic matrix
        #self.K = np.array([[self.focal_length_x, 0, self.c_x],
         #                  [0, self.focal_length_y, self.c_y],
          #                 [0, 0, 1]])

        # Initialize extrinsic parameters (will be updated based on drone's pose)
        #self.R = np.eye(3)
        #self.t = np.array([[0.2], [0], [0]])  # Position of camera relative to base_link (0.2 meters forward)
        # Initialize camera parameters with placeholders (they will be updated in callbacks)
        self.K = np.eye(3)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape((3, 3))

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert ROS Image message to OpenCV image: {str(e)}")
            return

        # Process the image here (e.g., draw 3D points projected onto the 2D image)
        processed_image = self.process_and_draw_objects(cv_image)

        # Show the image with points drawn on objects
        cv2.imshow("Image with Objects", processed_image)
        cv2.waitKey(1)

    def gt_pose_callback(self, msg):
        # Extract the position of the drone
        drone_position = np.array([[msg.pose.position.x],
                                   [msg.pose.position.y],
                                   [msg.pose.position.z]])

        # Extract orientation from the PoseStamped message and convert to a rotation matrix
        quaternion = (msg.pose.orientation.x,
                      msg.pose.orientation.y,
                      msg.pose.orientation.z,
                      msg.pose.orientation.w)
        drone_orientation_matrix = R.from_quat(quaternion).as_matrix()

        # Combine drone's pose and the fixed camera translation to get the camera's extrinsic parameters
        self.R = drone_orientation_matrix
        self.t = np.dot(drone_orientation_matrix,
                        self.t) + drone_position  # Update translation based on drone's orientation and position
        # Print the updated rotation matrix and translation vector
        print("Updated Extrinsic Parameters:")
        print("Rotation (R):")
        print(self.R)
        print("Translation (t):")
        print(self.t)

    def update_camera_projection_matrix(self):
        # Update your camera projection matrix here if necessary
        pass

    # Assuming the extrinsic parameters (self.R and self.t) are correctly set in gt_pose_callback

    def process_and_draw_objects(self, image):
        # Print the intrinsic and extrinsic parameters
        print(f"Intrinsic Matrix (K):\n{self.K}")
        print(f"Extrinsic Parameters:\nRotation (R):\n{self.R} \nTranslation (t):\n{self.t}")

        for object_name, data in self.objects_data.items():
            pose_3d_world = np.array(data["pose"])
            # Print original 3D position of the object
            print(f"Object {object_name} 3D World Pose:\n{pose_3d_world}")
            # Transform to camera frame
            pose_cam = np.dot(self.R, pose_3d_world.reshape((3, 1))) + self.t
            # Print the pose in camera frame
            print(f"Pose in camera frame:\n{pose_cam}")
            # Project onto the image plane
            point_2d_homogeneous = np.dot(self.K, pose_cam)
            if point_2d_homogeneous[2] != 0:
                point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
                print(f"Projected 2D point (normalized):\n{point_2d}")
                # Check image bounds
                x_img, y_img = point_2d.flatten()
                if 0 <= x_img < self.image_width and 0 <= y_img < self.image_height:
                    projected_point = (int(x_img), int(y_img))
                    # If visible, draw on image
                    cv2.circle(image, projected_point, 5, (0, 255, 0), -1)
                    print(f"Object {object_name} is visible at {projected_point}")
                else:
                    print(f"Projected 2D point out of image bounds: {point_2d.flatten()}")
                    print(f"Object {object_name} is not visible.")
            else:
                print(f"Point behind camera: z={point_2d_homogeneous[2]}")
                print(f"Object {object_name} is not visible.")

        return image

    def project_3d_to_2d(self, pose_3d):
        # Transform from world to camera coordinate frame
        pose_cam = np.dot(self.R, pose_3d) + self.t
        print(f"Pose in camera frame: {pose_cam}")

        # Projection onto the camera's image plane
        point_2d_homogeneous = np.dot(self.K, pose_cam)
        print(f"Homogeneous image point: {point_2d_homogeneous}")

        if point_2d_homogeneous[2] <= 0:
            print(f"Point behind camera: z={point_2d_homogeneous[2]}")
            return (0, 0), False

        point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
        x_img, y_img = point_2d.flatten()
        print(f"Projected 2D point: {point_2d}")

        # Check against clipping planes and image bounds
        if 0 <= x_img < self.image_width and 0 <= y_img < self.image_height:
            return (int(x_img), int(y_img)), True
        else:
            print(f"Point out of image bounds: ({x_img}, {y_img})")
            return (0, 0), False

    def project_and_check_visibility(self, pose_cam):
        point_2d_homogeneous = np.dot(self.K, pose_cam)
        if point_2d_homogeneous[2] != 0:
            # Convert from homogeneous coordinates to pixel coordinates
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            x_img, y_img = point_2d.flatten()

            # Check if the point is within the image bounds
            if 0 <= x_img < self.image_width and 0 <= y_img < self.image_height:
                projected_point = (int(x_img), int(y_img))
                print(f"Object {object_name} is visible at {projected_point}")
                # Optionally, draw the point on the image
                cv2.circle(image, projected_point, 5, (0, 255, 0), -1)
            else:
                print(f"Projected 2D point out of image bounds: {point_2d.flatten()}")
                print(f"Object {object_name} is not visible.")
        else:
            print(f"Point behind camera: z={point_2d_homogeneous[2]}")
            print(f"Object {object_name} is not visible.")


# ROS2 Node execution
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


    def process_and_draw_objects(self, image):
        # Transformam pozitia obiectelor din world frame in camera frame , le proiectam apoi in planul imaginii si le
        # desenam in imagine daca sunt vizibile
        # Print the intrinsic and extrinsic parameters
        print(f"Matirce patrametrii intrinseci (K):\n{self.K}")
        print(f"Parametrii extrinseci:\nRotatia (R):\n{self.R} \nTranslatia (t):\n{self.t}")

        for object_name, data in self.objects_data.items():
            pose_3d_world = np.array(data["pose"])
            # Print original 3D position of the object
            print(f"Obiect {object_name} Pozitia in 3D:\n{pose_3d_world}")
            # Transform to camera frame
            pose_cam = np.dot(self.R, pose_3d_world.reshape((3, 1))) + self.t
            # Print the pose in camera frame
            print(f"Pozitie in camera frame:\n{pose_cam}")

            # Project onto the image plane (valoarea normalizata a pozitiei obiectelor)
            point_2d_homogeneous = np.dot(self.K, pose_cam)
            if point_2d_homogeneous[2] != 0:
                point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
                print(f"Proiectia punctului 2D (normalizat):\n{point_2d}")
                # Check image bounds
                x_img, y_img = point_2d.flatten()
                if 0 <= x_img < self.image_width and 0 <= y_img < self.image_height:
                    projected_point = (int(x_img), int(y_img))
                    # If visible, draw on image
                    cv2.circle(image, projected_point, 5, (0, 255, 0), -1)
                    print(f"Obiectul {object_name} este vizibil la coordonatele {projected_point}")
                else:
                    print(f"Proiectia punctului 2D este in afara marginilor imaginii: {point_2d.flatten()}")
                    print(f"Obiectul {object_name} nu este vizibil.")
            else:
                print(f"Punct in spatele camerei: z={point_2d_homogeneous[2]}")
                print(f"Obiectul {object_name} nu este vizibil.")

        return image

    def process_and_draw_objects(self, image):
        # Print the intrinsic and extrinsic parameters
        print(f"Intrinsic Matrix (K):\n{self.K}")
        print(f"Extrinsic Parameters:\nRotation (R):\n{self.R} \nTranslation (t):\n{self.t}")

        for object_name, data in self.objects_data.items():
            # Add Z adjustment to the object's pose
            pose_3d_world = np.array(data["pose"])
            pose_3d_world[2] += 5.2  # Adjust Z value

            # Transform to camera frame
            pose_cam = np.dot(self.R, pose_3d_world.reshape((3, 1))) + self.t
            # Print the pose in camera frame
            print(f"Pose in camera frame for {object_name}:\n{pose_cam}")

            # Project onto the image plane
            point_2d_homogeneous = np.dot(self.K, pose_cam)
            if point_2d_homogeneous[2] != 0:
                point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]

                x_img, y_img = point_2d.flatten()
                if 0 <= x_img < self.image_width and 0 <= y_img < self.image_height:
                    projected_point = (int(x_img), int(y_img))
                    cv2.circle(image, projected_point, 5, (0, 255, 0), -1)
                    print(f"Object {object_name} is visible at {projected_point}")
                # The else part is removed so it won't print when the object is out of bounds
            # The check for points behind the camera is also omitted

        return image

    #######################################################################################3
 # ASTA E ALA BUN
    import os
    import csv
    import time
    import cv2
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge, CvBridgeError
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy, DurabilityPolicy
    from scipy.spatial.transform import Rotation as R
    from sensor_msgs.msg import CameraInfo


    def euler_to_rotation_matrix(roll, pitch, yaw):
        # Calculate rotation matrix from Euler angles
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R


    class ImageSubscriber(Node):
        def __init__(self):
            super().__init__('image_processor')
            # Define a custom QoS profile with the correct access to reliability and durability
            custom_qos_profile = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE)

            # Use the custom QoS profile for the subscription
            self.subscription = self.create_subscription(
                Image,
                '/drone/front/image_raw',
                self.image_callback,
                custom_qos_profile)
            self.pose_subscription = self.create_subscription(
                PoseStamped,
                '/drone/gt_pose',
                self.gt_pose_callback,
                custom_qos_profile)
            self.camera_info_subscription = self.create_subscription(
                CameraInfo,
                '/drone/front/camera_info',
                self.camera_info_callback,
                custom_qos_profile)
            self.bridge = CvBridge()
            self.last_save_time = time.time()
            self.save_interval = 5  # seconds
            self.image_folder = '/home/andrei/ros2_ws/images'
            self.csv_file_path = '/home/andrei/ros2_ws/dataset.csv'
            os.makedirs(self.image_folder, exist_ok=True)
            self.csv_file = open(self.csv_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # self.csv_writer.writerow(
            #    ["timestamp", "image_name", "object_name", "x", "y", "z", "roll", "pitch", "yaw", "size_x", "size_y",
            #     "size_z", "x_min_bb", "y_min_bb", "x_max_bb", "y_max_bb"])
            self.image_counter = 0
            self.objects_data = {
                "asphalt_plane": {
                    "pose": [0.027118, -0.028406, 0],
                    "orientation": [0, -0, 0],
                    "size": [20.000000, 20.000000, 0.100000],
                },
                "control_console": {
                    "pose": [0.044032, 9.471480, 0],
                    "orientation": [0, 0, 0],
                    "size": [1.780000, 0.050000, 2.602250],
                },

                "unit_box": {
                    "pose": [9.406795, -9.399209, 0.550000],
                    "orientation": [0, 0, 0.000034],
                    "size": [1.000000, 1.000000, 1.000000],
                },
                "unit_box_clone": {
                    "pose": [9.424715, 9.274771, 0.550000],
                    "orientation": [0, 0, 0.000040],
                    "size": [1.000000, 1.000000, 1.000000],
                },
                "unit_box_clone_clone": {
                    "pose": [-9.261930, 9.274990, 0.550000],
                    "orientation": [0, 0, 0.000016],
                    "size": [1.000000, 1.000000, 1.000000],
                },
                "unit_box_clone_clone_clone": {
                    "pose": [-8.978432, -9.372633, 0.549995],
                    "orientation": [-0.000010, 0.000001, 0.111119],
                    "size": [1.000000, 1.000000, 1.000000],
                },
                "number1": {
                    "pose": [-3.443590, -9.539260, 0.400000],
                    "orientation": [0, 0, 0],
                    "size": [1.000000, 1.000000, 1.000000],
                },
                "number2": {
                    "pose": [-9.348060, 5.928570, 0.400000],
                    "orientation": [0, 0, 0],
                    "size": [1.000000, 1.000000, 1.000000],
                },
                "number3": {
                    "pose": [4.345270, 9.360890, 0.400000],
                    "orientation": [0, 0, 0],
                    "size": [1.000000, 1.000000, 1.000000],
                },
                "number4": {
                    "pose": [9.514570, -4.511090, 0.400000],
                    "orientation": [0, 0, 0],
                    "size": [1.000000, 1.000000, 1.000000],
                },
                "Construction Cone": {
                    "pose": [3.440740, 0.667633, 0.049999],
                    "orientation": [-0.000002, 0.000002, -0.004192],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_0": {
                    "pose": [2.936450, 0.915504, 0.050000],
                    "orientation": [0, 0, -0.001118],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_1": {
                    "pose": [2.419270, 1.061460, 0.049991],
                    "orientation": [0, -0.000002, 0.000614],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_2": {
                    "pose": [1.916210, 1.225750, 0.050000],
                    "orientation": [0, 0, -0.000372],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_3": {
                    "pose": [1.412490, 1.358500, 0.049991],
                    "orientation": [0, 0, -0.014740],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_4": {
                    "pose": [0.906406, 1.475050, 0.050000],
                    "orientation": [0, 0, 0.002556],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_5": {
                    "pose": [0.377529, 1.550320, 0.050001],
                    "orientation": [0, -0.000002, -0.000256],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_6": {
                    "pose": [-0.141692, 1.602430, 0.050000],
                    "orientation": [0, 0, 0],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_7": {
                    "pose": [-0.669365, 1.771780, 0.049990],
                    "orientation": [-0.000002, 0, -0.000256],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_8": {
                    "pose": [-1.185310, 1.961130, 0.050000],
                    "orientation": [0.000002, -0.000002, 0],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_9": {
                    "pose": [-1.720400, 2.194210, 0.049992],
                    "orientation": [0, 0.000002, 0],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Construction Cone_10": {
                    "pose": [-2.251430, 2.447520, 0.050000],
                    "orientation": [0, 0, 0],
                    "size": [0.198, 0.198, 0.428696],
                },
                "Dumpster": {
                    "pose": [1.105810, -9.136140, 0.051370],
                    "orientation": [0.000010, 0, -3.136771],
                    "size": [1.773333, 0.886666, 0.686666],
                },
                "hoop_red": {
                    "pose": [-3.397180, -10.136400, 0],
                    "orientation": [0, 0, 0],
                    "size": [0.300000, 0.100000, 5.000000],
                },
                "hoop_red_0": {
                    "pose": [-9.971850, 3.162810, 0],
                    "orientation": [0, 0, -1.556320],
                    "size": [0.300000, 0.100000, 5.000000],
                },
                "person_standing": {
                    "pose": [0.739916, -7.874418, 0.050000],
                    "orientation": [0, -0.000002, 0.001970],
                    "size": [0.500000, 0.350000, 0.020000],
                },
                "person_walking": {
                    "pose": [-7.531469, 5.773241, 0.049999],
                    "orientation": [0, 0.000006, 0.005452],
                    "size": [0.350000, 0.750000, 0.020000],
                },
                "person_standing_0": {
                    "pose": [0.592537, 8.466297, 0.050000],
                    "orientation": [0, -0.000002, -3.140959],
                    "size": [0.500000, 0.350000, 0.020000],
                },
                "person_standing_1": {
                    "pose": [-0.195836, 2.403852, 0.050000],
                    "orientation": [0, 0.000002, 0.001718],
                    "size": [0.500000, 0.350000, 0.020000],
                },
                "person_standing_2": {
                    "pose": [-9.561253, -8.743303, 0.294097],
                    "orientation": [-1.636140, 0.909863, 0.015181],
                    "size": [0.500000, 0.350000, 0.020000],
                },
                "person_standing_3": {
                    "pose": [1.827380, 6.648900, 0.000537],
                    "orientation": [0, 0, 0],
                    "size": [0.500000, 0.350000, 0.020000],
                },
                "person_standing_4": {
                    "pose": [-2.140890, 5.534660, 0.002202],
                    "orientation": [0, 0, 0],
                    "size": [0.500000, 0.350000, 0.020000],
                },
                "person_standing_5": {
                    "pose": [-2.494090, 3.011030, -0.007182],
                    "orientation": [0, 0, 0],
                    "size": [0.500000, 0.350000, 0.020000],
                },
                "person_standing_6": {
                    "pose": [4.132250, 5.527540, 0.007169],
                    "orientation": [0, 0, 0],
                    "size": [0.500000, 0.350000, 0.020000],
                },
                "person_standing_7": {
                    "pose": [-3.036670, -5.568780, -0.009657],
                    "orientation": [0, 0, 0],
                    "size": [0.500000, 0.350000, 0.020000],
                },
            }

            # Camera parameters
            self.horizontal_fov = 2.09  # Horizontal field of view from URDF
            self.image_width = 640  # Image width
            self.image_height = 360  # Image height
            self.focal_length_x = self.image_width / (2 * np.tan(self.horizontal_fov / 2))
            self.focal_length_y = self.focal_length_x  # Assuming square pixels for simplicity
            self.c_x = self.image_width / 2
            self.c_y = self.image_height / 2

            # Camera intrinsic matrix
            # self.K = np.array([[self.focal_length_x, 0, self.c_x],
            #                  [0, self.focal_length_y, self.c_y],
            #                 [0, 0, 1]])

            # Initialize extrinsic parameters (will be updated based on drone's pose)
            # self.R = np.eye(3)
            # self.t = np.array([[0.2], [0], [0]])  # Position of camera relative to base_link (0.2 meters forward)

            # Initialize camera parameters with placeholders (they will be updated in callbacks)
            self.K = np.eye(3)
            self.R = np.eye(3)
            self.t = np.zeros((3, 1))

        def camera_info_callback(self, msg):
            # Get the current time for timestamp
            current_time = self.get_clock().now().to_msg()
            self.K = np.array(msg.k).reshape((3, 3))
            # Print the intrinsic matrix and timestamp
            self.get_logger().info(f'[{current_time}] Parametrii intrinseci updatati (K):\n{self.K}')

        def image_callback(self, msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                processed_image = self.process_and_draw_objects(cv_image)
                cv2.imshow("Imagine cu obiecte", processed_image)
                cv2.waitKey(1)
            except CvBridgeError as e:
                self.get_logger().error(f"Could not convert ROS Image message to OpenCV image: {str(e)}")

        def gt_pose_callback(self, msg):
            # Get the current time for timestamp
            current_time = self.get_clock().now().to_msg()

            # Extract position
            position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

            # Extract orientation and convert to rotation matrix
            orientation_q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                             msg.pose.orientation.w]
            self.R = R.from_quat(orientation_q).as_matrix()

            # Update extrinsic parameters
            self.t = position.reshape((3, 1))

            # Print the updated extrinsic parameters and timestamp
            self.get_logger().info(
                f'[{current_time}] Parametrii extrinseci updatati:\nMatricea de rotatie (R):\n{self.R}\nMatricea de translatie (t):\n{self.t}')

        def process_and_draw_objects(self, image):
            # Print the intrinsic and extrinsic parameters
            print(f"Intrinsic Matrix (K):\n{self.K}")
            print(f"Extrinsic Parameters:\nRotation (R):\n{self.R} \nTranslation (t):\n{self.t}")

            for object_name, data in self.objects_data.items():
                # Add Z adjustment to the object's pose
                pose_3d_world = np.array(data["pose"])
                pose_3d_world[2] += 5.2  # Adjust Z value

                # Transform to camera frame
                pose_cam = np.dot(self.R, pose_3d_world.reshape((3, 1))) + self.t
                # Print the pose in camera frame
                print(f"Pose in camera frame for {object_name}:\n{pose_cam}")

                # Project onto the image plane
                point_2d_homogeneous = np.dot(self.K, pose_cam)
                if point_2d_homogeneous[2] != 0:
                    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]

                    x_img, y_img = point_2d.flatten()
                    if 0 <= x_img < self.image_width and 0 <= y_img < self.image_height:
                        projected_point = (int(x_img), int(y_img))
                        cv2.circle(image, projected_point, 5, (0, 255, 0), -1)
                        print(f"Object {object_name} is visible at {projected_point}")
                    # The else part is removed so it won't print when the object is out of bounds
                # The check for points behind the camera is also omitted

            return image

        def project_3d_to_2d(self, pose_3d):
            # Proiecteaza un punct 3D in 2D
            # Transform from world to camera coordinate frame
            pose_cam = np.dot(self.R, pose_3d) + self.t
            # Projection onto the camera's image plane
            point_2d_homogeneous = np.dot(self.K, pose_cam)
            # Convert from homogeneous coordinates to pixel coordinates
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            return point_2d

        def project_and_check_visibility(self, pose_cam):
            # Verificam daca punctul 3D proiectat in 2D se afla in cadrul imaginii
            point_2d_homogeneous = np.dot(self.K, pose_cam)
            if point_2d_homogeneous[2] != 0:
                # Convert from homogeneous coordinates to pixel coordinates
                point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
                x_img, y_img = point_2d.flatten()

                # Check if the point is within the image bounds
                if 0 <= x_img < self.image_width and 0 <= y_img < self.image_height:
                    projected_point = (int(x_img), int(y_img))
                    print(f"Obiect {object_name} vizibil la {projected_point}")
                    # Optionally, draw the point on the image
                    cv2.circle(image, projected_point, 5, (0, 255, 0), -1)
                else:
                    print(f"Proiectia punctului 2D este in afara marginilor imaginii: {point_2d.flatten()}")
                    print(f"Obiectul {object_name} nu e vizbil.")
            else:
                print(f"Punct in spatele camerei: z={point_2d_homogeneous[2]}")
                print(f"Obiectul {object_name} nu e vizibil.")


    # ROS2 Node execution
    def main(args=None):
        rclpy.init(args=args)
        image_subscriber = ImageSubscriber()
        cv2.namedWindow("Imagine cu obiecte", cv2.WINDOW_NORMAL)  # Initialize the window
        rclpy.spin(image_subscriber)
        cv2.destroyAllWindows()  # Destroy the window when done
        image_subscriber.destroy_node()
        rclpy.shutdown()


    if __name__ == '__main__':
        main()

###############################################################################################################3
## ASTA E ALA BUN2