import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import numpy as np
import yaml
import os
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation


class HandeyeTFPublisher(Node):
    def __init__(self):
        super().__init__('handeye_tf_publisher')

        config_path = os.path.join(
            get_package_share_directory('beep_jetcobot_control'),
            'config', 'handeye_result.yaml'
        )
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        R = np.array(data['R_cam2ee']['data']).reshape(3, 3)
        t = np.array(data['t_cam2ee']['data'])

        q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]

        broadcaster = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'joint6_flange'
        tf.child_frame_id  = 'camera_frame'
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])
        tf.transform.rotation.x = float(q[0])
        tf.transform.rotation.y = float(q[1])
        tf.transform.rotation.z = float(q[2])
        tf.transform.rotation.w = float(q[3])

        broadcaster.sendTransform(tf)
        self.get_logger().info('handeye static TF 발행: joint6_flange → camera_frame')


def main(args=None):
    rclpy.init(args=args)
    node = HandeyeTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
