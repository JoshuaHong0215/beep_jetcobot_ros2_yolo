# beep_jetcobot_ros2_yolo

**YOLOv8 + IBVS 기반 로봇팔 Pick & Place 시스템**

ROS2 환경에서 YOLOv8 객체 탐지와 IBVS(Image-Based Visual Servoing)를 결합하여  
Jetcobot 로봇팔의 자율 Pick & Place를 구현한 프로젝트입니다.

---

## Overview

OpenCV Contour 기반의 기존 방식에서 더 나아가, Roboflow로 구축한 커스텀 데이터셋으로  
YOLOv8 모델을 학습시켜 보다 강건한 객체 탐지 성능을 확보하였습니다.  
탐지된 바운딩박스의 중점을 추출하고 IBVS를 통해 카메라 중점과 정렬한 후  
TCP를 align하여 하강 및 피킹을 수행합니다.

---

## System Architecture

```
[Camera] → [YOLOv8 Detector] → [IBVS Controller] → [Jetcobot Arm]
                                      ↑
                              Closed Loop Feedback
```

### Network Structure

AI 서버, 랩탑, 로봇을 **Tailscale Mesh VPN**으로 연결하여 멀티 디바이스 환경을 구성하였습니다.

- 평균 Ping: **57ms**
- 패킷 손실: **0%**

---

## Pick & Place Process

단일 `/pick_place` 액션 서버에서 `task_id` × `mode` 분기로 **6가지 동작**을 통합 처리합니다.

| task_id | mode | 동작 |
|---------|------|------|
| 0 | 0 | blue box 입고 |
| 0 | 1 | blue box 출고 |
| 1 | 0 | red box 입고 |
| 1 | 1 | red box 출고 |
| 2 | 0 | yellow box 입고 |
| 2 | 1 | yellow box 출고 |

---

## IBVS Convergence

Visual Servoing의 수렴성을 실측 데이터로 검증하였습니다.

- 초기 오차 `e_y`: **+280px**
- **8회** 보정을 거쳐 수렴 범위 내 진입 확인

---

## AI Model (YOLOv8)

- 데이터셋: 영상 촬영 후 프레임 단위 Labeling (Roboflow)
- Augmentation 적용으로 조명 변화에 강건한 모델 구현
- **mAP50-95: 0.995**

> 단순한 형태와 뚜렷한 색상 대비의 통제된 환경에서의 학습 결과임을 감안해야 합니다.

---

## Package Structure

```
beep_jetcobot_ros2_yolo/
├── beep_jetcobot_bringup/       # Launch files
├── beep_jetcobot_control/       # YOLO detector & Pick & Place action server
├── beep_jetcobot_description/   # Robot URDF description
├── beep_jetcobot_moveit_config/ # MoveIt2 configuration
└── beep_jetcobot_msgs/          # Custom ROS2 messages
```

---

## Tech Stack

| Category | Stack |
|----------|-------|
| Framework | ROS2 |
| AI | YOLOv8, Roboflow |
| Vision | IBVS |
| Network | Tailscale Mesh VPN |
| Language | Python |

---

## Period

2026.05 ~ 06 @ AddinEdu_PinkLAB
