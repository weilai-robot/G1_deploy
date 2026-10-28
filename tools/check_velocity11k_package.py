#!/usr/bin/env python3
"""Check frozen 11k model/config/header integrity, without third-party Python packages."""
import argparse
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT='8d4815ac2754bdf249d24e88c68ad1707679d09a2b2f56cb56916e86184178db'


def check(package):
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    if manifest['checkpoint_sha256']!=EXPECTED_CHECKPOINT or manifest['checkpoint_iter']!=10999:
        raise ValueError('This is not the verified 11k baseline package')
    for path,key in [(package/'exported/policy.onnx','policy_sha256'),
                     (package/'params/deploy.yaml','deploy_yaml_sha256'),
                     (REPO/'deploy/include/velocity11k/contract.h','cpp_contract_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[key]:
            raise ValueError(f'Checksum mismatch: {path}')
    contract=manifest['contract']
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=manifest['contract_sha256'] or contract['observation_dim']!=480 or contract['action_dim']!=29:
        raise ValueError('Invalid 11k contract')
    print('PASS: frozen model_10999; 480 observations / 29 actions; term-major history; model/config/C++ contract hashes')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',nargs='?',type=Path,default=REPO/'deploy/robots/g1_29dof/config/policy/velocity/11k')
    args=parser.parse_args()
    try: check(args.package.resolve())
    except (OSError,ValueError,KeyError) as error: raise SystemExit('FAILED: '+str(error))
