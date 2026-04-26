# -*- coding: utf-8 -*-

import struct
import math
from PyQt6.QtGui import QQuaternion, QVector3D

class VmdBoneFrame():
    def __init__(self):
        self.name = ''
        self.frame = 0
        self.position = QVector3D(0, 0, 0)
        self.rotation = QQuaternion()

    def write(self, fout):
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
        fout.write(struct.pack('<L', self.frame))
        fout.write(struct.pack('b', self.show))
        fout.write(struct.pack('<L', len(self.ik)))
        for k in self.ik:
            name = k.name.encode('ms932')
            fout.write(name)
            fout.write(bytearray([0 for i in range(len(name), 20)]))
            fout.write(struct.pack('b', k.onoff))
        
class VmdWriter():
    def __init__(self):
        pass

    def write_vmd_file(self, filename, bone_frames, showik_frames):
        fout = open(filename, 'wb')
        fout.write(b'Vocaloid Motion Data 0002\x00\x00\x00\x00\x00')
        fout.write(b'Dummy Model Name    ')
        fout.write(struct.pack('<L', len(bone_frames)))
        for bf in bone_frames:
            bf.write(fout)
        fout.write(struct.pack('<L', 0))
        fout.write(struct.pack('<L', 0))
        fout.write(struct.pack('<L', 0))
        fout.write(struct.pack('<L', 0))
        fout.write(struct.pack('<L', len(showik_frames)))
        for sf in showik_frames:
            sf.write(fout)
        fout.close()
