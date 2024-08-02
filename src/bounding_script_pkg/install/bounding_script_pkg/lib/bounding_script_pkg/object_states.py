#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from bounding_script_pkg.msg import ObjectState  # Assuming ObjectState is your custom message type
import gazebo_interface  # Replace with your actual module for interacting with Gazebo

class GazeboObjectStatePublisher(Node):
    def __init__(self):
        super().__init__('gazebo_object_state_publisher')
        self.publisher = self.create_publisher(ObjectState, '/object_states', 10)
        self.timer = self.create_timer(1.0, self.publish_object_states)

    def publish_object_states(self):
        object_states = gazebo_interface.get_object_states()  # Replace with your method to get states from Gazebo

        for state in object_states:
            msg = ObjectState()
            msg.name = state.name
            msg.position = state.position
            msg.size = state.size
            # Set other fields as needed

            self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GazeboObjectStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
