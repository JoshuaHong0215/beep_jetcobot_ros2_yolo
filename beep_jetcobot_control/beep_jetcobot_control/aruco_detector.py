import threading
import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
from flask import Flask, Response
import yaml
# Float32MultiArray: 여러 개의 float 값을 하나의 메시지로 묶어 topic으로 보낼 수 있는 ROS2 표준 타입
from std_msgs.msg import Float32MultiArray
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np


ARUCO_DICT = aruco.DICT_4X4_250
FLASK_PORT = 5000

_latest_frame = None
_frame_lock = threading.Lock()

flask_app = Flask(__name__)


@flask_app.route('/stream')
def stream():
    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame is None:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        # 패키지 경로를 기반으로 켈리브레이션 yaml파일의 절대경로를 구성함
        config_path = os.path.join(
            get_package_share_directory('beep_jetcobot_control'),
            'config', 'camera_cali.yaml'
        )
        # yaml파일을 읽어 카메라 고유 파라미터(내부 행렬 및 왜곡 계수) 로드
        with open(config_path, 'r') as f:
            calib = yaml.safe_load(f)

        # 로드한 데이터를 추후 pose estimation(3D위치 추정)에 사용할 수 있도록 numpy배열로 변홤
        # self.K = 카메라 내부 행렬(초점거리 및 광학 중심 정보를 포함한 3x3 행렬)
        # self.D = 렌즈 왜곡 계수(방사형 및 접선 왜곡 보정용 벡터)
        self.K = np.array(calib['camera_matrix']['data']).reshape(3, 3)
        self.D = np.array(calib['distortion_coefficients']['data']) 
        


        self.cap = cv2.VideoCapture('/dev/jetcocam0')
        if not self.cap.isOpened():
            self.get_logger().error('카메라를 열 수 없습니다')
            return

        self.aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
        self.aruco_params = aruco.DetectorParameters_create()

        flask_thread = threading.Thread(
            target=lambda: flask_app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True),
            daemon=True
        )
        flask_thread.start()

        # 마커의 3D 좌표(x,y,z) + 고정 접근각도(rx,ry,rz) 6개 값을 pick_place에게 보내는 publisher
        self.marker_pub = self.create_publisher(Float32MultiArray, '/marker_coord', 10)

        self.timer = self.create_timer(0.03, self.detect)
        self.get_logger().info(f'aruco_detector_node 시작 — 스트림: http://0.0.0.0:{FLASK_PORT}/stream')

    def detect(self):
        global _latest_frame
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('프레임을 읽을 수 없습니다')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                cx = int(corners[i][0][:, 0].mean())
                cy = int(corners[i][0][:, 1].mean())
                self.get_logger().info(f'마커 ID: {marker_id} | 중심: ({cx}, {cy})')

        _, buf = cv2.imencode('.jpg', frame)
        with _frame_lock:
            _latest_frame = buf.tobytes()


        if ids is not None:
            # 카메라 행렬(K)과 왜곡 계소(D)를 사용하여 마커의 3D자세( 위치 및 회전)를 추정
            # 0.025는 켈리브레이션 체커보드의 한 칸, 한 변의 길이
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.025, self.K, self.D  # 0.05 = 마커 실제 크기(m)
            )
            # 검출된 각 마커의 ID별로 로봇이 이해할 수 있는 좌표(x, y, z) 정보 로깅
            for i, marker_id in enumerate(ids.flatten()):
                # tvec[i]는 카메라 중심으로부터 마커 중심까지의 [x, y, z] 이동 벡터
                tvec = tvecs[i][0]  # x, y, z (미터)

                # tvec은 미터 단위 → MyCobot은 mm 단위를 사용하므로 *1000 변환
                # 뒤 세 값(-180, 0, 90)은 로봇의 접근 각도(rx, ry, rz) 고정값
                msg = Float32MultiArray()
                msg.data = [
                    tvec[0] * 1000,
                    tvec[1] * 1000,
                    tvec[2] * 1000,
                    -180.0, 0.0, 90.0
                ]
                # /marker_coord topic으로 6개 값 전송 → pick_place가 subscribe해서 사용
                self.marker_pub.publish(msg)

                self.get_logger().info(
                    f'ID: {marker_id} | x={tvec[0]*1000:.1f} y={tvec[1]*1000:.1f} z={tvec[2]*1000:.1f} (mm)'
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
