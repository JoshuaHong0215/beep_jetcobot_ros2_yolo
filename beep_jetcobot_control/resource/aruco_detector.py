import threading
import socket
import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import yaml
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np

from std_msgs.msg import Float32MultiArray


ARUCO_DICT = aruco.DICT_6X6_250
UDP_PORT   = 9998

# 카메라는 preview_loop에서만 읽고 최신 프레임을 여기에 저장
_latest_frame = None
_frame_lock   = threading.Lock()

_clients      = set()
_clients_lock = threading.Lock()


def client_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UDP_PORT))
    while True:
        _, addr = sock.recvfrom(16)
        with _clients_lock:
            _clients.add(addr[0])
        print(f'클라이언트 등록: {addr[0]}')


def camera_loop(cap, aruco_dict, aruco_params):
    """카메라 읽기 + UDP 스트리밍 전담 스레드"""
    global _latest_frame

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 최신 프레임 저장 (detect()가 여기서 가져감)
        with _frame_lock:
            _latest_frame = frame.copy()

        # UDP 프리뷰
        small = cv2.resize(frame, (320, 240))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            aruco.drawDetectedMarkers(small, corners, ids)
            cx_m = int(corners[0][0][:, 0].mean())
            cy_m = int(corners[0][0][:, 1].mean())
            cv2.circle(small, (cx_m, cy_m), 5, (0, 255, 0), -1)
            cv2.putText(small, f'DETECTED id={ids.flatten()[0]}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(small, 'NOT DETECTED', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.line(small, (160, 0), (160, 240), (255, 255, 0), 1)
        cv2.line(small, (0, 120), (320, 120), (255, 255, 0), 1)

        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 40])
        data = buf.tobytes()

        with _clients_lock:
            for ip in list(_clients):
                try:
                    sock.sendto(data, (ip, UDP_PORT))
                except Exception:
                    pass


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        config_path = os.path.join(
            get_package_share_directory('beep_jetcobot_control'),
            'config', 'camera_cali.yaml'
        )
        with open(config_path, 'r') as f:
            calib = yaml.safe_load(f)

        K = np.array(calib['camera_matrix']['data']).reshape(3, 3)
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.dist_coeffs = np.array(calib['distortion_coefficients']['data'])

        self.error_pub = self.create_publisher(Float32MultiArray, '/marker_error', 10)

        self.cap = cv2.VideoCapture('/dev/jetcocam0')
        if not self.cap.isOpened():
            self.get_logger().error('카메라를 열 수 없습니다')
            return

        self.aruco_dict   = aruco.Dictionary_get(ARUCO_DICT)
        self.aruco_params = aruco.DetectorParameters_create()
        # 인식률 향상 파라미터
        self.aruco_params.adaptiveThreshWinSizeMin  = 3
        self.aruco_params.adaptiveThreshWinSizeMax  = 53
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.minMarkerPerimeterRate     = 0.03
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize   = 5
        self._miss_count = 0
        self._MISS_THRESH = 5

        threading.Thread(target=client_listener, daemon=True).start()
        threading.Thread(
            target=camera_loop,
            args=(self.cap, self.aruco_dict, self.aruco_params),
            daemon=True
        ).start()

        self.timer = self.create_timer(0.05, self.detect)
        self.get_logger().info('aruco_detector_node 시작 — python view_udp.py <로봇IP>')

    def detect(self):
        global _latest_frame

        with _frame_lock:
            if _latest_frame is None:
                return
            frame = _latest_frame.copy()

        frame = cv2.undistort(frame, self.K, self.dist_coeffs)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        msg = Float32MultiArray()

        if ids is None:
            self._miss_count += 1
            if self._miss_count >= self._MISS_THRESH:
                msg.data = [0.0, 0.0, 0.0]
                self.error_pub.publish(msg)
            return

        self._miss_count = 0

        u = corners[0][0][:, 0].mean()
        v = corners[0][0][:, 1].mean()

        e_x = (u - self.cx) / self.fx
        e_y = (v - self.cy) / self.fy

        msg.data = [float(e_x), float(e_y), 1.0]
        self.error_pub.publish(msg)

        self.get_logger().info(
            f'마커 | 픽셀({u:.0f},{v:.0f}) | 오차 e_x={e_x:.4f} e_y={e_y:.4f}'
        )

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
