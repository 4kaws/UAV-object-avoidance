import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time


class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/drone/front/image_raw',
            self.image_callback,
            10)
        self.last_saved_time = time.time()
        self.save_interval = 3  # Save image every 3 seconds
        self.image_count = 0
        self.save_directory = os.path.expanduser('~/ros2_ws/images')
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        self.get_logger().info('Image Saver Node has been started.')
        cv2.namedWindow("Drone Camera", cv2.WINDOW_NORMAL)

    def image_callback(self, msg):
        current_time = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Display the image in an OpenCV window
        cv2.imshow("Drone Camera", cv_image)
        cv2.waitKey(1)

        if current_time - self.last_saved_time >= self.save_interval:
            image_filename = f'image_{int(current_time)}.jpg'
            image_path = os.path.join(self.save_directory, image_filename)
            cv2.imwrite(image_path, cv_image)
            self.get_logger().info(f'Saved image: {image_filename}')
            self.last_saved_time = current_time
            self.image_count += 1


def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()
    try:
        rclpy.spin(image_saver)
    except KeyboardInterrupt:
        pass
    image_saver.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
