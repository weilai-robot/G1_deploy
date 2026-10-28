#!/usr/bin/env python3
"""Run compiled production inference against saved PyTorch fixtures; no robot or Python ML libraries."""
import argparse
import json
import math
import subprocess
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def check(binary,model,fixture):
    cases=json.loads(fixture.read_text(encoding='utf-8'))
    manifest=json.loads((REPO/'deploy/robots/g1_29dof/config/policy/velocity/11k/manifest.json').read_text(encoding='utf-8'))
    if cases['checkpoint_sha256']!=manifest['checkpoint_sha256'] or cases['contract_sha256']!=manifest['contract_sha256']:
        raise ValueError('Fixture provenance differs from the deployed contract')
    lines=[]
    for case in cases['cases']:
        s=case['snapshot']
        values=[int(case['reset']),*s['quat'],*s['omega'],*s['q'],*s['dq'],*s['command']]
        lines.append(' '.join(format(float(v),'.9g') for v in values))
    process=subprocess.run([str(binary),str(model)],input='\n'.join(lines)+'\n',capture_output=True,text=True,timeout=30)
    if process.returncode:raise ValueError(process.stderr.strip())
    outputs=process.stdout.splitlines()
    if len(outputs)!=3*len(cases['cases']):raise ValueError('Wrong number of C++ inference replies')
    maxima={'observation':0.,'action':0.,'target':0.}
    for i,case in enumerate(cases['cases']):
        for index,key,atol,rtol in [(0,'observation',2e-6,2e-6),(1,'action',2e-5,2e-5),(2,'target',2e-5,2e-5)]:
            actual=[float(v) for v in outputs[3*i+index].split()]
            expected=case[key]
            if len(actual)!=len(expected):raise ValueError('Runtime output shape mismatch')
            for a,b in zip(actual,expected):
                error=abs(a-b)
                if not math.isfinite(a) or error>atol+rtol*abs(b):
                    raise ValueError(f'Runtime mismatch: case={i} {key}, difference={error}')
                maxima[key]=max(maxima[key],error)
    version=subprocess.check_output([str(binary),'--version'],text=True).strip()
    print(json.dumps({'result':'PASS','cases':len(cases['cases']),'onnxruntime':version,'max_abs_difference':maxima,'robot_commands_sent':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=REPO/'deploy/robots/g1_29dof/build/velocity11k_offline')
    parser.add_argument('--model',type=Path,default=REPO/'deploy/robots/g1_29dof/config/policy/velocity/11k/exported/policy.onnx')
    parser.add_argument('--fixture',type=Path,default=REPO/'docs/velocity11k_runtime_fixture.json')
    args=parser.parse_args()
    try:check(args.binary.resolve(),args.model.resolve(),args.fixture.resolve())
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:raise SystemExit('FAILED: '+str(error))
