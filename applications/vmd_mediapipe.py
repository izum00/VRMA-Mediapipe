#!/usr/bin/env python3
# vmd_mediapipe.py

import argparse
import os
import cv2
import mediapipe as mp

from VmdWriter import VmdWriter, VmdBoneFrame
import posisions as ps
import pos2vmd

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = os.path.realpath(DIR_PATH + '/..')
MODEL_PATH = os.path.join(PROJECT_PATH, 'data/saved_sessions/pose_landmarker_full.task')
HAND_MODEL_PATH = os.path.join(PROJECT_PATH, 'data/saved_sessions/hand_landmarker.task')

# MediaPipe Hands のランドマーク番号
HL = {
    'WRIST': 0,
    'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
    'INDEX_MCP': 5, 'INDEX_PIP': 6, 'INDEX_DIP': 7, 'INDEX_TIP': 8,
    'MIDDLE_MCP': 9, 'MIDDLE_PIP': 10, 'MIDDLE_DIP': 11, 'MIDDLE_TIP': 12,
    'RING_MCP': 13, 'RING_PIP': 14, 'RING_DIP': 15, 'RING_TIP': 16,
    'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20,
}

# ここはあなたのMMDモデルのボーン名に合わせて変更
HAND_BONE_NAMES = {
    'left': {
        'thumb':  ['左親指１', '左親指２', '左親指３'],
        'index':  ['左人指１', '左人指２', '左人指３'],
        'middle': ['左中指１', '左中指２', '左中指３'],
        'ring':   ['左薬指１', '左薬指２', '左薬指３'],
        'pinky':  ['左小指１', '左小指２', '左小指３'],
    },
    'right': {
        'thumb':  ['右親指１', '右親指２', '右親指３'],
        'index':  ['右人指１', '右人指２', '右人指３'],
        'middle': ['右中指１', '右中指２', '右中指３'],
        'ring':   ['右薬指１', '右薬指２', '右薬指３'],
        'pinky':  ['右小指１', '右小指２', '右小指３'],
    }
}

def _to_vec(lm):
    return QVector3D(float(lm.x), float(lm.y), float(lm.z))

def _make_bone_frame(name, frame_num, direction, rest_dir=QVector3D(0, 1, 0)):
    bf = VmdBoneFrame()
    bf.name = name
    bf.frame = frame_num
    bf.position = QVector3D(0, 0, 0)

    if direction.lengthSquared() > 1e-12:
        direction.normalize()
        bf.rotation = QQuaternion.rotationTo(rest_dir, direction)
    else:
        bf.rotation = QQuaternion()

    return bf

def hand_world_landmarks_to_frames(hand_world_landmarks, handedness, frame_num):
    # handedness は "Left" / "Right" 想定
    side = 'left' if handedness.lower().startswith('left') else 'right'
    bones = HAND_BONE_NAMES[side]

    # 各指を「関節間ベクトル」に向ける簡易版
    chains = [
        ('thumb',  [(HL['THUMB_CMC'], HL['THUMB_MCP']),
                    (HL['THUMB_MCP'], HL['THUMB_IP']),
                    (HL['THUMB_IP'],  HL['THUMB_TIP'])]),
        ('index',  [(HL['INDEX_MCP'], HL['INDEX_PIP']),
                    (HL['INDEX_PIP'], HL['INDEX_DIP']),
                    (HL['INDEX_DIP'], HL['INDEX_TIP'])]),
        ('middle', [(HL['MIDDLE_MCP'], HL['MIDDLE_PIP']),
                    (HL['MIDDLE_PIP'], HL['MIDDLE_DIP']),
                    (HL['MIDDLE_DIP'], HL['MIDDLE_TIP'])]),
        ('ring',   [(HL['RING_MCP'], HL['RING_PIP']),
                    (HL['RING_PIP'], HL['RING_DIP']),
                    (HL['RING_DIP'], HL['RING_TIP'])]),
        ('pinky',  [(HL['PINKY_MCP'], HL['PINKY_PIP']),
                    (HL['PINKY_PIP'], HL['PINKY_DIP']),
                    (HL['PINKY_DIP'], HL['PINKY_TIP'])]),
    ]

    frames = []
    for finger_name, joint_pairs in chains:
        bone_names = bones[finger_name]
        for bone_name, (a, b) in zip(bone_names, joint_pairs):
            pa = _to_vec(hand_world_landmarks[a])
            pb = _to_vec(hand_world_landmarks[b])
            direction = pb - pa
            frames.append(_make_bone_frame(bone_name, frame_num, direction))
    return frames

def vmd_convert(image_file, vmd_file, center_enabled=False):
    image_file_path = os.path.realpath(image_file)
    cap = cv2.VideoCapture(image_file_path)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_landmarker = PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    positions_list = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 30.0

    frame_num = 0
    print('pose estimation start. fps:%.1f' % (fps))

    try:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timestamp_ms = int((frame_num / fps) * 1000)

            try:
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as ex:
                print(ex)
                frame_num += 1
                continue

            try:
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as ex:
                print(ex)
                hand_result = None

            pose_2d = pose_result.pose_landmarks
            pose_3d = pose_result.pose_world_landmarks
            if not pose_2d or not pose_3d:
                frame_num += 1
                continue

            positions = ps.convert(pose_3d, pose_2d)
            positions_list.append({
                'position': positions['position'],
                'hands': hand_result
            })

            print('frame_num:', frame_num)
            frame_num += 1

    finally:
        pose_landmarker.close()
        hand_landmarker.close()
        cap.release()

    ps.refine([x if x is None else {'position': x['position']} for x in positions_list])

    bone_frames = []
    frame_num = 0
    for item in positions_list:
        if item is None:
            frame_num += 1
            continue

        positions = {'position': item['position']}
        frames = pos2vmd.positions_to_frames(positions['position'], frame_num)

        if center_enabled:
            ps.center(positions, frames, frame_num)

        hand_result = item.get('hands')
        if hand_result and hand_result.hand_world_landmarks:
            for i, hand_world in enumerate(hand_result.hand_world_landmarks):
                handedness = 'Right'
                if hand_result.handedness and len(hand_result.handedness) > i and hand_result.handedness[i]:
                    handedness = hand_result.handedness[i][0].category_name
                frames.extend(
                    hand_world_landmarks_to_frames(hand_world, handedness, frame_num)
                )

        bone_frames.extend(frames)
        frame_num += 1

    showik_frames = pos2vmd.make_showik_frames()
    writer = VmdWriter()
    writer.write_vmd_file(vmd_file, bone_frames, showik_frames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='estimate 3D pose and generate VMD motion')
    parser.add_argument('--center', action='store_true', help='move center bone (experimental)')
    parser.add_argument('IMAGE_FILE')
    parser.add_argument('VMD_FILE')

    arg = parser.parse_args()
    vmd_convert(arg.IMAGE_FILE, arg.VMD_FILE, arg.center)
