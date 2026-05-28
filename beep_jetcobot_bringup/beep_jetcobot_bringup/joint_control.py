import rclpy as rp
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pymycobot.mycobot import MyCobot
import math

class JointControlNode(Node):
    def __init__(self):
        # 노드 이름 등록
        super().__init__('joint_control_node')

    
        self.joint_pub = self.create_publisher(
            JointState,
            'joint_states', 
            10
            )
        
        self.mc = MyCobot("/dev/ttyJETCOBOT", 1000000)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('jetcobot 관절 제어 노드가 켜졌습니다')

    def timer_callback(self):
        msg = JointState()
        msg.name = [
            'joint2_to_joint1',
            'joint3_to_joint2',
            'joint4_to_joint3',
            'joint5_to_joint4',
            'joint6_to_joint5',
            'joint6output_to_joint6'
        ]
            
        
        angles = self.mc.get_angles()
        if angles and len(angles) == 6:
            msg.position = [math.radians(a) for a in angles]

        else:
            return
        
        self.joint_pub.publish(msg)



def main(args=None):
    rp.init(args=args)
    node = JointControlNode()

    try:
        rp.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rp.shutdown()


if __name__ == '__main__':
    main()


