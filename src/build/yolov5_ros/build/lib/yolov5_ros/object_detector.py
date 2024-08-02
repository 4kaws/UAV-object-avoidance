#!/usr/bin/env python

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from laser_geometry import LaserProjection

class YOLOv5Detector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.bridge = CvBridge()
        self.image_width = None  # Variable to store image width
        self.lidar_data = None
        self.point_cloud = None
        self.laser_projector = LaserProjection()

        # Camera image subscription
        self.subscription_image = self.create_subscription(
            Image,
            '/drone/front/image_raw',
            self.image_callback,
            10)

        # LiDAR data subscription
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/drone/robo_scan/out',
            self.lidar_callback,
            10)

        # Bounding box marker publisher
        self.publisher = self.create_publisher(Marker, 'bounding_box', 10)

        # Load the YOLOv5 model
        model_dir = os.path.expanduser('~/ros2_ws/src/yolov5')
        model_path = os.path.join(model_dir, 'runs/train/exp/weights/best.pt')
        self.model = torch.hub.load(model_dir, 'custom', path=model_path, source='local')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image_width = cv_image.shape[1]  # Store image width
        results = self.model(cv_image)
        results.print()

        # Display the detection results
        cv2.imshow("YOLOv5 Detection", results.render()[0])
        cv2.waitKey(1)

        # Process detection results
        self.process_detection_results(results)

    def process_detection_results(self, results):
        if self.image_width is not None and self.point_cloud is not None:
            for bbox in results.xyxy[0]:
                cx = (bbox[0] + bbox[2]) / 2  # Calculate center x-coordinate
                angle = self.bbox_to_angle(cx)
                distance = self.find_corresponding_range(angle)
                print(f"Object at angle {angle} is {distance} meters away.")
                if distance != float('inf'):
                    self.create_3d_bounding_box(bbox, distance, angle)

    def bbox_to_angle(self, cx):
        relative_x = (cx / self.image_width) * 2 - 1
        angle = relative_x * (self.lidar_data.angle_max - self.lidar_data.angle_min) / 2 + \
                (self.lidar_data.angle_max + self.lidar_data.angle_min) / 2
        return angle.cpu().numpy()  # Move to CPU and then convert to NumPy array

    def find_corresponding_range(self, angle):
        """
        Find the closest range measurement in the LiDAR scan to a given angle.
        """
        angles = np.linspace(self.lidar_data.angle_min, self.lidar_data.angle_max, len(self.lidar_data.ranges))
        angle = angle.cpu().numpy() if isinstance(angle, torch.Tensor) else angle  # Ensure angle is a NumPy array
        index = np.argmin(np.abs(angles - angle))
        range_value = self.lidar_data.ranges[index]
        if range_value == float('inf'):
            print(f"Warning: Object at angle {angle} out of LiDAR range.")
        return range_value

    def create_3d_bounding_box(self, bbox, distance, angle):
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = 0  # Assuming ground level for simplicity

        # Process point cloud to refine the bounding box
        points = self.point_cloud_to_xyz()
        if points is not None:
            cluster_boxes = self.fit_bounding_boxes(points)
            if cluster_boxes:
                for cluster_box in cluster_boxes:
                    self.publish_bounding_box(cluster_box['center'], cluster_box['dimensions'])
            else:
                self.publish_bounding_box((x, y, z), (1.0, 1.0, 1.0))  # Fallback if clustering fails

    def point_cloud_callback(self, msg):
        self.point_cloud = msg

    def lidar_callback(self, msg):
        self.lidar_data = msg
        self.point_cloud = self.laser_projector.projectLaser(msg)

    def point_cloud_to_xyz(self):
        if self.point_cloud is None:
            return None
        points = []
        for point in pc2.read_points(self.point_cloud, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def fit_bounding_boxes(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=10))
        cluster_boxes = []
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster = points[labels == label]
            if len(cluster) > 0:
                aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(cluster))
                center = aabb.get_center()
                dimensions = aabb.get_extent()
                cluster_boxes.append({'center': center, 'dimensions': dimensions})
        return cluster_boxes

    def publish_bounding_box(self, center, dimensions):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounding_boxes"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        marker.scale.x = float(dimensions[0])
        marker.scale.y = float(dimensions[1])
        marker.scale.z = float(dimensions[2])
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    yolov5_detector = YOLOv5Detector()
    rclpy.spin(yolov5_detector)
    yolov5_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
