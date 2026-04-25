# -*- coding: utf-8 -*-

import json
import struct
from collections import defaultdict
from PyQt6.QtGui import QQuaternion, QVector3D


class VmdBoneFrame():
    def __init__(self):
        self.name = ''
        self.frame = 0
        self.position = QVector3D(0, 0, 0)
        self.rotation = QQuaternion()

    def write(self, fout):
        # 互換用に残しています（VMDバイナリ出力）
        name = self.name.encode('ms932')
        fout.write(name)
        fout.write(bytearray([0 for i in range(len(name), 15)]))
        fout.write(struct.pack('<L', self.frame))
        fout.write(struct.pack('<f', self.position.x()))
        fout.write(struct.pack('<f', self.position.y()))
        fout.write(struct.pack('<f', self.position.z()))
        v = self.rotation.toVector4D()
        fout.write(struct.pack('<f', v.x()))
        fout.write(struct.pack('<f', v.y()))
        fout.write(struct.pack('<f', v.z()))
        fout.write(struct.pack('<f', v.w()))
        fout.write(bytearray([0 for i in range(0, 64)]))


class VmdInfoIk():
    def __init__(self, name='', onoff=0):
        self.name = name
        self.onoff = onoff


class VmdShowIkFrame():
    def __init__(self):
        self.frame = 0
        self.show = 0
        self.ik = []

    def write(self, fout):
        # 互換用に残しています（VMDバイナリ出力）
        fout.write(struct.pack('<L', self.frame))
        fout.write(struct.pack('b', self.show))
        fout.write(struct.pack('<L', len(self.ik)))
        for k in self.ik:
            name = k.name.encode('ms932')
            fout.write(name)
            fout.write(bytearray([0 for i in range(len(name), 20)]))
            fout.write(struct.pack('b', k.onoff))


class _GlbBuilder:
    def __init__(self):
        self._buffer = bytearray()
        self.buffer_views = []
        self.accessors = []

    def _align4(self):
        while len(self._buffer) % 4 != 0:
            self._buffer.append(0)

    def add_float_accessor(self, values, item_type):
        """
        values: flat float list
        item_type: 'SCALAR' / 'VEC3' / 'VEC4'
        """
        flat = list(values)
        self._align4()
        byte_offset = len(self._buffer)
        if flat:
            self._buffer.extend(struct.pack('<%sf' % len(flat), *flat))
        byte_length = len(self._buffer) - byte_offset
        self._align4()

        buffer_view_index = len(self.buffer_views)
        self.buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length,
        })

        accessor = {
            "bufferView": buffer_view_index,
            "componentType": 5126,  # FLOAT
            "count": len(flat) // (1 if item_type == "SCALAR" else 3 if item_type == "VEC3" else 4),
            "type": item_type,
        }

        if item_type == "SCALAR" and flat:
            accessor["min"] = [min(flat)]
            accessor["max"] = [max(flat)]

        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def build_glb(self, gltf_dict):
        json_bytes = json.dumps(
            gltf_dict,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        json_pad = (4 - (len(json_bytes) % 4)) % 4
        json_chunk = json_bytes + (b" " * json_pad)

        bin_bytes = bytes(self._buffer)
        bin_pad = (4 - (len(bin_bytes) % 4)) % 4
        bin_chunk = bin_bytes + (b"\x00" * bin_pad)

        total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)

        header = struct.pack("<4sII", b"glTF", 2, total_length)
        json_header = struct.pack("<I4s", len(json_chunk), b"JSON")
        bin_header = struct.pack("<I4s", len(bin_chunk), b"BIN\x00")

        return header + json_header + json_chunk + bin_header + bin_chunk


class VmdWriter():
    """
    既存の呼び出し方を維持したまま、VRMA(.vrma / GLB) を出力します。
    """

    FPS = 30.0

    # よくある VMD / MMD ボーン名 -> VRM Humanoid 名
    BONE_NAME_MAP = {
        "センター": "hips",
        "グルーブ": "hips",
        "下半身": "hips",
        "上半身": "spine",
        "上半身2": "chest",
        "上半身3": "upperChest",
        "首": "neck",
        "頭": "head",
        "顎": "jaw",
        "左目": "leftEye",
        "右目": "rightEye",

        "左肩": "leftShoulder",
        "左腕": "leftUpperArm",
        "左ひじ": "leftLowerArm",
        "左肘": "leftLowerArm",
        "左手首": "leftHand",

        "右肩": "rightShoulder",
        "右腕": "rightUpperArm",
        "右ひじ": "rightLowerArm",
        "右肘": "rightLowerArm",
        "右手首": "rightHand",

        "左足": "leftUpperLeg",
        "左ひざ": "leftLowerLeg",
        "左膝": "leftLowerLeg",
        "左足首": "leftFoot",
        "左つま先": "leftToes",

        "右足": "rightUpperLeg",
        "右ひざ": "rightLowerLeg",
        "右膝": "rightLowerLeg",
        "右足首": "rightFoot",
        "右つま先": "rightToes",
    }

    def __init__(self):
        pass

    def _normalize_bone_name(self, name: str) -> str:
        n = (name or "").strip()
        return self.BONE_NAME_MAP.get(n, n)

    def _group_frames(self, bone_frames):
        grouped = defaultdict(dict)
        for bf in bone_frames:
            bone_name = self._normalize_bone_name(bf.name)
            grouped[bone_name][int(bf.frame)] = bf  # 同一フレームは後勝ち
        return grouped

    def _serialize_showik_extras(self, showik_frames):
        result = []
        for sf in showik_frames:
            result.append({
                "frame": int(sf.frame),
                "show": int(sf.show),
                "ik": [
                    {"name": k.name, "onoff": int(k.onoff)}
                    for k in sf.ik
                ],
            })
        return result

    def write_vmd_file(self, filename, bone_frames, showik_frames):
        """
        VRMA (glTF GLB) を書き出します。
        呼び出し側の構造はそのままです。
        """
        builder = _GlbBuilder()

        grouped = self._group_frames(bone_frames)

        node_indices = {}
        nodes = []
        animations = []

        # ボーンごとに 1 ノードを作成
        for bone_name in sorted(grouped.keys()):
            node_indices[bone_name] = len(nodes)
            nodes.append({
                "name": bone_name,
            })

        # ボーンごとの animation channel を作成
        samplers = []
        channels = []

        for bone_name in sorted(grouped.keys()):
            frame_map = grouped[bone_name]
            frames = sorted(frame_map.keys())
            if not frames:
                continue

            times = [f / self.FPS for f in frames]
            translations = []
            rotations = []

            for f in frames:
                bf = frame_map[f]
                translations.extend([
                    float(bf.position.x()),
                    float(bf.position.y()),
                    float(bf.position.z()),
                ])
                v = bf.rotation.toVector4D()
                rotations.extend([
                    float(v.x()),
                    float(v.y()),
                    float(v.z()),
                    float(v.w()),
                ])

            time_accessor = builder.add_float_accessor(times, "SCALAR")
            translation_accessor = builder.add_float_accessor(translations, "VEC3")
            rotation_accessor = builder.add_float_accessor(rotations, "VEC4")

            sampler_translation = len(samplers)
            samplers.append({
                "input": time_accessor,
                "output": translation_accessor,
                "interpolation": "LINEAR",
            })
            channels.append({
                "sampler": sampler_translation,
                "target": {
                    "node": node_indices[bone_name],
                    "path": "translation",
                },
            })

            sampler_rotation = len(samplers)
            samplers.append({
                "input": time_accessor,
                "output": rotation_accessor,
                "interpolation": "LINEAR",
            })
            channels.append({
                "sampler": sampler_rotation,
                "target": {
                    "node": node_indices[bone_name],
                    "path": "rotation",
                },
            })

        if samplers:
            animations.append({
                "name": "Motion",
                "samplers": samplers,
                "channels": channels,
            })

        humanoid = {}
        for bone_name, node_index in node_indices.items():
            # VRM Humanoid の標準名だけを extension に載せる
            if bone_name in {
                "hips", "spine", "chest", "upperChest", "neck", "head", "jaw",
                "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
                "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
                "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
                "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
                "leftThumbMetacarpal", "leftThumbProximal", "leftThumbDistal",
                "leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal",
                "leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal",
                "leftRingProximal", "leftRingIntermediate", "leftRingDistal",
                "leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal",
                "rightThumbMetacarpal", "rightThumbProximal", "rightThumbDistal",
                "rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal",
                "rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal",
                "rightRingProximal", "rightRingIntermediate", "rightRingDistal",
                "rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal",
            }:
                humanoid[bone_name] = {"node": node_index}

        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "OpenAI VRMA Writer",
            },
            "scene": 0,
            "scenes": [
                {
                    "nodes": list(range(len(nodes))),
                }
            ],
            "nodes": nodes,
            "buffers": [
                {
                    "byteLength": len(builder._buffer),
                }
            ],
            "bufferViews": builder.buffer_views,
            "accessors": builder.accessors,
            "animations": animations,
            "extensionsUsed": ["VRMC_vrm_animation"],
            "extensionsRequired": ["VRMC_vrm_animation"],
            "extensions": {
                "VRMC_vrm_animation": {
                    "specVersion": "1.0",
                    "humanoid": {
                        "humanBones": humanoid,
                    },
                }
            },
        }

        # showik は VRMA の公式内容には含まれないため、互換用に extras に保存
        if showik_frames:
            gltf["extras"] = {
                "legacyShowIkFrames": self._serialize_showik_extras(showik_frames)
            }

        glb_bytes = builder.build_glb(gltf)

        with open(filename, "wb") as fout:
            fout.write(glb_bytes)
