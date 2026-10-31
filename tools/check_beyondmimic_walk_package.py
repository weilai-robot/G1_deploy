#!/usr/bin/env python3
"""Dependency-free integrity check of both native velocity-cyclic walking packages."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def check(package):
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    if manifest['format']!='beyondmimic_velocity_package_v1':
        raise ValueError('Not a velocity walking package')
    for name,key in [('policy.onnx','policy_sha256'),('gait.bin','gait_sha256'),('deploy.yaml','deploy_yaml_sha256')]:
        if hashlib.sha256((package/name).read_bytes()).hexdigest()!=manifest[key]:
            raise ValueError('Checksum mismatch: '+str(package/name))
    raw=(package/'gait.bin').read_bytes()
    magic,n,j,b,s,dt,fps,nominal,acceleration=struct.unpack('<8sIIIIffff',raw[:40])
    limits=struct.unpack('<6f',raw[40:64])
    support=struct.unpack('<'+'I'*s,raw[64:64+4*s])
    if (magic!=b'BMGCYC1\0' or j!=29 or b!=14 or n!=manifest['frames'] or
        fps!=manifest['fps'] or abs(dt-.02)>1e-6 or abs(fps*dt-1)>1e-5 or
        not math.isfinite(nominal) or nominal<=0 or not math.isfinite(acceleration) or acceleration<=0 or
        len(raw)!=64+4*s+n*156*4 or len(support)<2 or list(support)!=sorted(set(support)) or
        support[-1]>=n or not all(map(math.isfinite,limits)) or limits[0]!=0 or
        limits[1]<=0 or limits[2:4]!=(0,0) or not limits[4]<=0<=limits[5]):
        raise ValueError('Invalid cyclic gait header/support/limits')
    values=array.array('f');values.frombytes(raw[64+4*s:])
    if sys.byteorder!='little':values.byteswap()
    if not all(map(math.isfinite,values)):
        raise ValueError('Non-finite gait tensors')
    for frame in range(n):
        for body in range(b):
            k=frame*156+100+4*body
            if abs(math.sqrt(sum(x*x for x in values[k:k+4]))-1)>.01:
                raise ValueError('Invalid gait quaternion')
    if not manifest['normalizer_embedded'] or manifest['real_robot_tested']:
        raise ValueError('Unexpected normalization or robot-acceptance claim')
    return {'model_id':manifest['model_id'],'integrity':'passed','frames':n,'support_frames':support,
            'policy_sha256':manifest['policy_sha256'],'gait_sha256':manifest['gait_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('package',type=Path,nargs='?')
    a=p.parse_args()
    base=Path(__file__).resolve().parents[1]/'deploy/robots/g1_29dof/config/policy/beyondmimic'
    for package in ([a.package] if a.package else [base/'walk_baseline',base/'walk_contact']):
        print(json.dumps(check(package)))


if __name__=='__main__':main()
