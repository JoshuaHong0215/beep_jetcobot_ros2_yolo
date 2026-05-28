import rclpy as np
from rclpy.node import Node

from pymycobot.mycobot280 import MyCobot280
import time

# 클래스 생성
class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        self.mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
        self.speed = 30

        
        self.home_angles = [0, 0, 0, 0, 0, 0]


    # home pose 정의
    def go_home(self):
        pass

    # gripper open
    def open_gripper(self):
        pass

    # gripper close
    def close_gripper(self):
        pass

    # 이동
    def move_to(self):
        pass

    # 동작 전체 구현
    def run(self):
        pass

