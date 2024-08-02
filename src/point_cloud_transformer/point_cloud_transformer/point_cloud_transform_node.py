import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from geometry_msgs.msg import TransformStamped

class PointCloudTransformer(Node):
    def __init__(self):
        super().__init__('point_cloud_transformer')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/points',
            self.point_cloud_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/points_transformed', 10)
        
        # Initialize the tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info('PointCloudTransformer node has been started.')

    def point_cloud_callback(self, msg):
        try:
            transform_stamped = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
            transformed_cloud = do_transform_cloud(msg, transform_stamped)
            transformed_cloud.header.frame_id = 'base_link'
            self.publisher.publish(transformed_cloud)
        except Exception as e:
            self.get_logger().error(f'Error transforming point cloud: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
