import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
from sensor_msgs.msg import JointState
import tf2_ros
import numpy as np
import math
import time

# 서보 파라미터
K1        = 2.0
K2        = 2.0
THRESHOLD = 0.05
MAX_ITER  = 30
MAX_DELTA = 5.0

# 픽 파라미터
TABLE_Z  = 30.0   # 물체 표면 Z (base 기준, mm) — 실측 후 조정
PICK_Z   = 105.8  # 그리퍼 픽 높이 (mm)
APPROACH_Z = 200.0  # 접근 높이 (mm)


class PickPlaceTFNode(Node):
    def __init__(self):
        super().__init__('pick_place_tf_node')

        self.speed = 30
        self.home_angles   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ready_coords  = [129.1, -62.8, 317.1, -162.04, -18.67, -42.35]
        self.middle_coords = [150.0,  60.0, 200.0, -180.0,    0.0,  90.0]
        self.place_coords  = [  0.0, 150.0, 120.0, -180.0,    0.0,  90.0]

        self.marker_error   = None
        self.current_angles = None
        self.ee_coords      = None

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.joint_pub   = self.create_publisher(Float32MultiArray, '/joint_command',   10)
        self.coord_pub   = self.create_publisher(Float32MultiArray, '/coord_command',   10)
        self.gripper_pub = self.create_publisher(Int32,             '/gripper_command', 10)

        self.create_subscription(Float32MultiArray, '/marker_error', self.marker_error_cb,  10)
        self.create_subscription(JointState,        'joint_states',  self.joint_states_cb,  10)
        self.create_subscription(Float32MultiArray, '/ee_coords',    self.ee_coords_cb,     10)

        self.get_logger().info('pick_place_tf_node 시작')
        self.run()

    def marker_error_cb(self, msg):
        self.marker_error = list(msg.data)

    def joint_states_cb(self, msg):
        self.current_angles = [math.degrees(a) for a in msg.position]

    def ee_coords_cb(self, msg):
        self.ee_coords = list(msg.data)

    def send_angles(self, angles):
        msg = Float32MultiArray()
        msg.data = [float(a) for a in angles]
        self.joint_pub.publish(msg)

    def send_coords(self, coords):
        msg = Float32MultiArray()
        msg.data = [float(c) for c in coords]
        self.coord_pub.publish(msg)

    def send_gripper(self, value):
        msg = Int32()
        msg.data = value
        self.gripper_pub.publish(msg)

    def go_home(self):
        self.send_angles(self.home_angles)
        time.sleep(3)

    def go_ready(self):
        self.get_logger().info('레디포즈 이동')
        self.send_coords(self.ready_coords)
        time.sleep(4)

    def open_gripper(self):
        self.send_gripper(100)
        time.sleep(1)

    def close_gripper(self):
        self.send_gripper(0)
        time.sleep(1)

    def move_to(self, coords, label=''):
        self.get_logger().info(f'{label} 이동')
        self.send_coords(coords)
        time.sleep(3)

    def get_fresh_error(self):
        self.marker_error = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.marker_error is not None:
                if self.marker_error[2] > 0.5:
                    return self.marker_error
                self.marker_error = None
        return None

    def get_fresh_angles(self):
        self.current_angles = None
        deadline = time.time() + 1.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_angles is not None:
                return self.current_angles
        return None

    def angle_servo(self):
        """J1/J2 증분으로 물체를 이미지 중앙으로 수렴"""
        self.get_logger().info('각도 서보 시작')

        for i in range(MAX_ITER):
            error = self.get_fresh_error()
            if error is None:
                self.get_logger().warn(f'[{i}] 미감지')
                continue

            e_x, e_y, _ = error
            self.get_logger().info(f'[{i}] e_x={e_x:.4f} e_y={e_y:.4f}')

            if abs(e_x) < THRESHOLD and abs(e_y) < THRESHOLD:
                self.get_logger().info('서보 수렴')
                return True

            angles = self.get_fresh_angles()
            if angles is None:
                continue

            dj1 = max(-MAX_DELTA, min(MAX_DELTA, -K1 * e_x))
            dj2 = max(-MAX_DELTA, min(MAX_DELTA, -K2 * e_y))
            angles[0] += dj1
            angles[1] += dj2

            self.get_logger().info(f'  ΔJ1={dj1:+.2f}° ΔJ2={dj2:+.2f}°')
            self.send_angles(angles)
            time.sleep(1.5)

        self.get_logger().warn('서보 최대 반복 도달')
        return False

    def get_object_base_coords(self, e_x, e_y):
        """
        카메라 픽셀 오차 + TF + 알려진 테이블 Z로
        물체의 base 프레임 좌표를 계산.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                'g_base', 'camera_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f'TF 조회 실패: {e}')
            return None

        # 카메라 원점 (base 기준, m → mm)
        cx = tf.transform.translation.x * 1000.0
        cy = tf.transform.translation.y * 1000.0
        cz = tf.transform.translation.z * 1000.0

        # 카메라 회전
        q = tf.transform.rotation
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        # 카메라 ray (e_x = (u-cx)/fx, e_y = (v-cy)/fy 이미 정규화됨)
        ray_cam = np.array([e_x, e_y, 1.0])
        ray_base = R @ ray_cam

        # 테이블 평면 (Z = TABLE_Z) 과의 교점
        if abs(ray_base[2]) < 1e-6:
            self.get_logger().error('ray가 테이블과 평행')
            return None

        lam = (TABLE_Z - cz) / ray_base[2]
        obj_x = cx + lam * ray_base[0]
        obj_y = cy + lam * ray_base[1]

        self.get_logger().info(f'물체 base 좌표: x={obj_x:.1f} y={obj_y:.1f} z={TABLE_Z}')
        return obj_x, obj_y

    def run(self):
        time.sleep(1.0)
        self.get_logger().info('--- Pick and place (TF mode) start ---')

        self.open_gripper()
        self.go_ready()
        self.marker_error = None

        # 1. 서보로 물체 중앙 맞추기
        if not self.angle_servo():
            self.get_logger().error('서보 실패')
            self.go_home()
            return

        # 2. 마지막 오차로 base 좌표 계산
        error = self.get_fresh_error()
        if error is None:
            self.get_logger().error('물체 미감지')
            self.go_home()
            return

        result = self.get_object_base_coords(error[0], error[1])
        if result is None:
            self.go_home()
            return

        obj_x, obj_y = result

        # 현재 EE 자세 유지
        if self.ee_coords is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        rx = self.ee_coords[3] if self.ee_coords else self.ready_coords[3]
        ry = self.ee_coords[4] if self.ee_coords else self.ready_coords[4]
        rz = self.ee_coords[5] if self.ee_coords else self.ready_coords[5]

        # 3. 물체 위 접근 높이로 이동
        self.get_logger().info(f'접근: ({obj_x:.1f}, {obj_y:.1f}, {APPROACH_Z})')
        self.send_coords([obj_x, obj_y, APPROACH_Z, rx, ry, rz])
        time.sleep(3)

        # 4. PICK_Z 까지 직선 하강
        self.get_logger().info('하강')
        self.send_coords([obj_x, obj_y, PICK_Z, rx, ry, rz])
        time.sleep(3)

        # 5. 픽
        self.close_gripper()
        self.move_to(self.middle_coords, 'middle')
        self.move_to(self.place_coords,  'place')
        self.open_gripper()
        self.go_home()
        self.get_logger().info('--- 완료 ---')


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceTFNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
