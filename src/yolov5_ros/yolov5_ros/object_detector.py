#!/usr/bin/env python

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
from sklearn.cluster import DBSCAN
import torch
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class YOLOv5Detector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.bridge = CvBridge()
        self.image_width = None
        self.image_height = None
        self.point_cloud = None
        self.camera_intrinsics = None
        self.obstacles = []
        self.last_yolo_bboxes = []

        # DWA configuration parameters
        self.config = {
            "max_speed": 0.5,  # m/s
            "min_speed": 0.0,  # m/s
            "max_yawrate": 40.0 * np.pi / 180.0,  # rad/s
            "max_accel": 0.2,  # m/s^2
            "max_dyawrate": 40.0 * np.pi / 180.0,  # rad/s^2
            "v_reso": 0.01,  # m/s
            "yawrate_reso": 0.1 * np.pi / 180.0,  # rad/s
            "dt": 0.1,  # s
            "predict_time": 3.0,  # s
            "to_goal_cost_gain": 1.0,
            "speed_cost_gain": 1.0,
            "obstacle_cost_gain": 1.0,
            "robot_radius": 0.5,  # m
        }

        # Camera image subscription
        self.subscription_image = self.create_subscription(
            Image,
            '/drone/front/image_raw',
            self.image_callback,
            10)

        # LiDAR data subscription
        self.subscription_lidar = self.create_subscription(
            PointCloud2,
            '/points_transformed',
            self.lidar_callback,
            10)

        # Camera info subscription
        self.subscription_camera_info = self.create_subscription(
            CameraInfo,
            '/drone/front/camera_info',
            self.camera_info_callback,
            10)

        # Velocity publisher
        self.publisher_cmd_vel = self.create_publisher(Twist, '/drone/cmd_vel', 10)

        # Bounding box marker publisher
        self.publisher = self.create_publisher(MarkerArray, 'bounding_box', 10)

        # Load the YOLOv5 model
        model_dir = os.path.expanduser('~/ros2_ws/src/yolov5')
        model_path = os.path.join(model_dir, 'runs/train/exp/weights/best.pt')

        self.model = torch.hub.load(model_dir, 'custom', path=model_path, source='local')

        self.state = [0.0, 0.0, 0.0, 0.0, 0.0]  # x, y, theta, speed, yaw rate

    def camera_info_callback(self, msg):
        self.camera_intrinsics = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image_width = cv_image.shape[1]
        self.image_height = cv_image.shape[0]
        results = self.model(cv_image)
        results.print()

        cv2.imshow("YOLOv5 Detection", results.render()[0])
        cv2.waitKey(1)

        self.process_detection_results(results)

    def process_detection_results(self, results):
        if self.camera_intrinsics is not None and self.point_cloud is not None:
            marker_array = MarkerArray()
            marker_id = 0
            self.last_yolo_bboxes = []

            for bbox in results.xyxy[0]:

                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                self.last_yolo_bboxes.append(bbox)

                distance, min_point = self.find_corresponding_range(cx, cy)
                if distance != float('inf'):
                    self.create_3d_bounding_box(bbox, min_point, marker_id, marker_array)
                    marker_id += 1

            self.publisher.publish(marker_array)

    def find_corresponding_range(self, cx, cy):
        u = (cx - self.camera_intrinsics[0, 2]) / self.camera_intrinsics[0, 0]
        v = (cy - self.camera_intrinsics[1, 2]) / self.camera_intrinsics[1, 1]

        points = list(pc2.read_points(self.point_cloud, field_names=("x", "y", "z"), skip_nans=True))
        min_distance = float('inf')
        min_point = None
        for point in points:
            x, y, z = point
            if z != 0:
                projected_u = x / z
                projected_v = y / z
                distance = np.sqrt((projected_u - u) ** 2 + (projected_v - v) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    min_point = point
        if min_distance == float('inf'):
            print(f"Warning: No corresponding point found for pixel coordinates ({cx}, {cy}).")
        return min_distance, min_point

    def create_3d_bounding_box(self, bbox, min_point, marker_id, marker_array):
        if min_point is None:
            return

        x, y, z = min_point

        width = (bbox[2] - bbox[0]) / self.image_width
        height = (bbox[3] - bbox[1]) / self.image_height
        depth = 1.0

        width *= z
        height *= z

        print(f"3D Bounding Box - Center: ({x}, {y}, {z}), Dimensions: ({width}, {height}, {depth})")

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounding_boxes"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.scale.x = float(width)
        marker.scale.y = float(height)
        marker.scale.z = float(depth)
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker_array.markers.append(marker)

    def lidar_callback(self, msg):
        self.point_cloud = msg
        self.get_logger().info("LiDAR data received")

        points = list(pc2.read_points(self.point_cloud, field_names=("x", "y", "z"), skip_nans=True))
        points = np.array([(point[0], point[1], point[2]) for point in points], dtype=float)

        if len(points) > 0:
            self.obstacles = points[:, :2]

        clusters = self.cluster_lidar_points(points)
        self.process_clusters(clusters)
        self.get_logger().info(f"Obstacles updated: {len(self.obstacles)} points")

    def cluster_lidar_points(self, points):
        if len(points) == 0:
            return []

        dbscan = DBSCAN(eps=0.5, min_samples=10)
        labels = dbscan.fit_predict(points[:, :2])
        unique_labels = set(labels)

        clusters = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster = points[labels == label]
            clusters.append(cluster)

        return clusters

    def is_within_2d_bbox(self, bbox, point):
        x, y, _ = point
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    def process_clusters(self, clusters):
        for cluster in clusters:
            x_mean = np.mean(cluster[:, 0])
            y_mean = np.mean(cluster[:, 1])
            z_mean = np.mean(cluster[:, 2])
            cluster_center = np.array([x_mean, y_mean, z_mean])

            for bbox in self.last_yolo_bboxes:
                if self.is_within_2d_bbox(bbox, cluster_center):
                    self.create_3d_bounding_box(bbox, cluster_center)

    def calc_dynamic_window(self, state):
        Vs = [self.config["min_speed"], self.config["max_speed"],
              -self.config["max_yawrate"], self.config["max_yawrate"]]

        Vd = [state[3] - self.config["max_accel"] * self.config["dt"],
              state[3] + self.config["max_accel"] * self.config["dt"],
              state[4] - self.config["max_dyawrate"] * self.config["dt"],
              state[4] + self.config["max_dyawrate"] * self.config["dt"]]

        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw

    def motion(self, state, v, y):
        state[0] += v * np.cos(state[2]) * self.config["dt"]
        state[1] += v * np.sin(state[2]) * self.config["dt"]
        state[2] += y * self.config["dt"]
        state[3] = v
        state[4] = y
        state[5] = 0.0
        return state

    def calc_trajectory(self, state, v, y):
        traj = np.array(state)
        time = 0
        while time <= self.config["predict_time"]:
            state = self.motion(state, v, y)
            traj = np.vstack((traj, state))
            time += self.config["dt"]
        return traj

    def calc_control_and_trajectory(self, state, goal, ob):
        dw = self.calc_dynamic_window(state)
        x_init = state[:]
        x_init[5] = 0.0
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_traj = np.array([state])

        for v in np.arange(dw[0], dw[1], self.config["v_reso"]):
            for y in np.arange(dw[2], dw[3], self.config["yawrate_reso"]):
                traj = self.calc_trajectory(x_init, v, y)
                to_goal_cost = self.config["to_goal_cost_gain"] * np.linalg.norm(traj[-1, 0:2] - goal[0:2])
                speed_cost = self.config["speed_cost_gain"] * (self.config["max_speed"] - traj[-1, 3])
                ob_cost = self.config["obstacle_cost_gain"] * self.calc_obstacle_cost(traj, ob)
                final_cost = to_goal_cost + speed_cost + ob_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_traj = traj

        return best_u, best_traj

    def calc_obstacle_cost(self, traj, ob):
        if ob.size == 0:
            return 0.0

        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = traj[:, 0][:, np.newaxis] - ox[np.newaxis, :]
        dy = traj[:, 1][:, np.newaxis] - oy[np.newaxis, :]
        r = np.hypot(dx, dy)
        if np.array(r <= self.config["robot_radius"]).any():
            return float("Inf")
        return 1.0 / np.min(r)

    def dwa_control(self, state, goal, ob):
        u, traj = self.calc_control_and_trajectory(state, goal, ob)
        return u, traj

    def publish_cmd_vel(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.publisher_cmd_vel.publish(twist)

    def main_loop(self):

        while self.point_cloud is None:
            self.get_logger().info('Waiting for LiDAR data...')
            rclpy.spin_once(self, timeout_sec=1.0)

        state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        goal = [-5.794233, -0.114089]
        ob = np.array(self.obstacles)

        while rclpy.ok():
            if self.point_cloud is not None:
                u, traj = self.dwa_control(state, goal, ob)
                self.publish_cmd_vel(u[0], u[1])
                state = self.motion(state, u[0], u[1])
                self.get_logger().info(f'Updated state: {state}')
            rclpy.spin_once(self, timeout_sec=0.1)

def main(args=None):
    rclpy.init(args=args)
    yolov5_detector = YOLOv5Detector()
    yolov5_detector.main_loop()
    yolov5_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()