import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
from sensor_msgs.msg import JointState
import math
import time

# 서보 파라미터 (부호 반대면 발산 → 음수로 바꾸기)
K1        = 2.0   # e_x → J1(base) gain  [deg/normalized_px]
K2        = 2.0   # e_y → J1(base)로 내려가면서 보정용
THRESHOLD = 0.05
MAX_ITER        = 30   # 서보용
MAX_DESCEND_ITER = 60  # 하강용 (넉넉히)
MAX_DELTA = 5.0   # 한 스텝 최대 관절 변화량 [deg]

# 하강 파라미터
PICK_Z   = 105.8  # 픽 목표 Z [mm]
STEP_J3  = 5.0    # 한 스텝 J3 증분 [deg] — 실측 후 조정

# CAM → TCP 오프셋 [mm] (서보 수렴 후 TCP를 마커 위로 보정)
CAM_TO_TCP_X = 10
CAM_TO_TCP_Y = 10


class PickPlaceAngleNode(Node):
    def __init__(self):
        super().__init__('pick_place_angle_node')

        self.speed = 30
        self.home_angles   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ready_coords  = [129.1, -62.8, 317.1, -162.04, -18.67, -42.35]
        self.middle_coords = [150.0,  60.0, 200.0, -180.0,    0.0,  90.0]
        self.place_coords  = [  0.0, 150.0, 120.0, -180.0,    0.0,  90.0]

        self.marker_error   = None
        self.current_angles = None  # degrees
        self.ee_coords      = None

        self.joint_pub   = self.create_publisher(Float32MultiArray, '/joint_command',   10)
        self.coord_pub   = self.create_publisher(Float32MultiArray, '/coord_command',   10)
        self.gripper_pub = self.create_publisher(Int32,             '/gripper_command', 10)

        self.create_subscription(Float32MultiArray, '/marker_error', self.marker_error_cb,  10)
        self.create_subscription(JointState,        'joint_states',  self.joint_states_cb,  10)
        self.create_subscription(Float32MultiArray, '/ee_coords',    self.ee_coords_cb,     10)

        self.get_logger().info('pick_place_angle_node 시작')
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
        self.get_logger().info('홈 이동')
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

    def get_fresh_ee(self):
        self.ee_coords = None
        deadline = time.time() + 1.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.ee_coords is not None:
                return self.ee_coords
        return None

    def angle_servo(self):
        """
        e_x → J1(base 회전), e_y → J2(shoulder) 증분으로 마커 중앙 수렴.
        카메라 ~90° 회전 전제. 발산하면 K1/K2 부호 반전.
        """
        self.get_logger().info('각도 서보 시작')

        for i in range(MAX_ITER):
            error = self.get_fresh_error()
            if error is None:
                self.get_logger().warn(f'[{i}] 마커 미감지')
                continue

            e_x, e_y, _ = error
            self.get_logger().info(f'[{i}] e_x={e_x:.4f}  e_y={e_y:.4f}')

            if abs(e_x) < THRESHOLD and abs(e_y) < THRESHOLD:
                self.get_logger().info('수렴 완료')
                return True

            angles = self.get_fresh_angles()
            if angles is None:
                self.get_logger().warn('관절 각도 수신 실패')
                continue

            dj1 = max(-MAX_DELTA, min(MAX_DELTA, -K1 * e_x))
            dj2 = max(-MAX_DELTA, min(MAX_DELTA, -K2 * e_y))

            new_angles = angles[:]
            new_angles[0] += dj1
            new_angles[1] += dj2

            self.get_logger().info(f'  → ΔJ1={dj1:+.2f}°  ΔJ2={dj2:+.2f}°')
            self.send_angles(new_angles)
            time.sleep(1.5)

        self.get_logger().warn('최대 반복 도달')
        return False

    def angle_descend(self):
        """
        J3를 STEP_J3씩 증가시켜 팔을 내림.
        매 스텝마다 J1으로 XY 오차 보정.
        ee_coords Z가 PICK_Z 이하가 되면 종료.
        """
        self.get_logger().info('각도 하강 시작')

        for i in range(MAX_DESCEND_ITER):
            ee = self.get_fresh_ee()
            if ee is None:
                self.get_logger().warn(f'[{i}] ee_coords 수신 실패')
                continue

            cur_z = ee[2]
            self.get_logger().info(f'[{i}] 현재 Z={cur_z:.1f}')

            if cur_z <= PICK_Z + 2.0:
                self.get_logger().info('PICK_Z 도달')
                return True

            angles = self.get_fresh_angles()
            if angles is None:
                continue

            # XY 오차 보정 — 마커 안 보여도 하강은 계속
            error = self.get_fresh_error()
            if error is not None and error[2] > 0.5:
                e_x, _ , _ = error
                dj1 = max(-MAX_DELTA, min(MAX_DELTA, -K1 * e_x))
                angles[0] += dj1

            angles[2] += STEP_J3
            self.get_logger().info(f'  → J3={angles[2]:.2f}°  Z={cur_z:.1f}')
            self.send_angles(angles)
            time.sleep(1.5)

        self.get_logger().warn('하강 최대 반복 도달')
        return False

    def run(self):
        time.sleep(1.0)
        self.get_logger().info('--- Pick and place (angle servo) start ---')

        self.open_gripper()
        self.go_ready()
        self.marker_error = None

        if not self.angle_servo():
            self.get_logger().error('서보 실패 — 중단')
            self.go_home()
            return

        # CAM → TCP 보정 (coord 한방, 소량 이동이라 IK 안정)
        ee = self.get_fresh_ee()
        if ee is None:
            self.get_logger().error('ee_coords 없음 — 중단')
            return
        self.get_logger().info('CAM→TCP 보정')
        self.send_coords([
            ee[0] + CAM_TO_TCP_X,
            ee[1] + CAM_TO_TCP_Y,
            ee[2], ee[3], ee[4], ee[5]
        ])
        time.sleep(1.5)

        if not self.angle_descend():
            self.get_logger().error('하강 실패 — 중단')
            self.go_home()
            return

        self.close_gripper()
        self.move_to(self.middle_coords, 'middle')
        self.move_to(self.place_coords, 'place')
        self.open_gripper()
        self.go_home()
        self.get_logger().info('--- 완료 ---')


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceAngleNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
