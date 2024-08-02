#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

class PointCloudTransformer : public rclcpp::Node
{
public:
    PointCloudTransformer()
    : Node("point_cloud_transformer_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points", 10,
            std::bind(&PointCloudTransformer::pointCloudCallback, this, std::placeholders::_1)
        );

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points_transformed", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        geometry_msgs::msg::TransformStamped transform_stamped;
        try
        {
            // Directly use 'base_link' since 'world' is not available
            std::string target_frame = "base_link";
            std::string source_frame = msg->header.frame_id;
            if (!source_frame.empty() && source_frame[0] == '/') {
                source_frame = source_frame.substr(1);
            }

            transform_stamped = tf_buffer_.lookupTransform(target_frame, source_frame, tf2::TimePointZero);
            
            sensor_msgs::msg::PointCloud2 transformed_cloud;
            tf2::doTransform(*msg, transformed_cloud, transform_stamped);
            
            publisher_->publish(transformed_cloud);
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudTransformer>());
    rclcpp::shutdown();
    return 0;
}
