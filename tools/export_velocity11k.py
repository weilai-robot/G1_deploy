#!/usr/bin/env python3
"""Export the frozen 11k actor and generate its deployment contract; no Isaac required."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

CHECKPOINT_SHA256 = '8d4815ac2754bdf249d24e88c68ad1707679d09a2b2f56cb56916e86184178db'
TRAINING_COMMIT = '5421b4f'
REPO = Path(__file__).resolve().parents[1]
ASSET = 'source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keywords(call):
    return {item.arg: item.value for item in call.keywords}


def robot_contract(text):
    tree = ast.parse(text)
    robot = next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'UNITREE_G1_29DOF_CFG' for t in n.targets))
    kw = keywords(robot)
    init = ast.literal_eval(keywords(kw['init_state'])['joint_pos'])
    actuators = []
    for call in kw['actuators'].values:
        actuators.append({k: ast.literal_eval(v) for k, v in keywords(call).items()})
    return init, actuators, ast.literal_eval(kw['joint_sdk_names'])


def generate(training, checkpoint, package):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import yaml

    if sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Expected the verified TASK_20261020_069 model_10999, not a later trial')
    current = robot_contract((training / ASSET).read_text(encoding='utf-8'))
    historical = subprocess.check_output(['git', '-C', str(training), 'show', f'{TRAINING_COMMIT}:{ASSET}']).decode('utf-8')
    if current != robot_contract(historical):
        raise ValueError('Current robot contract differs from the historical 11k training commit')
    init, actuators, sdk_names = current
    # Order was verified against the Isaac articulation and the original deployment map.
    module = ast.parse((training/'scripts/rsl_rl/sim2sim_play.py').read_text(encoding='utf-8'))
    assignments = {n.targets[0].id: n.value for n in module.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    policy_names = ast.literal_eval(assignments['POLICY_JOINTS'])
    mapping = [sdk_names.index(name) for name in policy_names]
    assert mapping == ast.literal_eval(assignments['JOINT_IDS_MAP'])
    defaults, kp, kd, effort = [], [], [], []
    def match(value, name):
        if not isinstance(value, dict):
            return value
        found = [v for expr,v in value.items() if re.fullmatch(expr,name)]
        if len(found) != 1:
            raise ValueError(f'Non-unique motor parameter: {name}')
        return found[0]
    for name in policy_names:
        position = [v for expr,v in init.items() if re.fullmatch(expr,name)]
        assert len(position) <= 1
        defaults.append(position[0] if position else 0.)
        matches = [a for a in actuators if any(re.fullmatch(expr,name) for expr in a['joint_names_expr'])]
        if len(matches) != 1:
            raise ValueError(f'Non-unique actuator: {name}')
        actuator = matches[0]
        kp.append(match(actuator['stiffness'],name))
        kd.append(match(actuator['damping'],name))
        effort.append(match(actuator['effort_limit_sim'],name))
    xml_path = training/'resources/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml'
    xml = ET.fromstring(xml_path.read_text(encoding='utf-8'))
    limits = {n.attrib['name']: list(map(float,n.attrib['range'].split())) for n in xml.findall('.//joint') if 'range' in n.attrib}
    lower = [limits[n][0] for n in policy_names]
    upper = [limits[n][1] for n in policy_names]
    # These are the operator command bounds in the training task, not joystick multipliers.
    command_cfg = ast.parse((training/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py').read_text(encoding='utf-8'))
    commands_cls = next(n for n in command_cfg.body if isinstance(n,ast.ClassDef) and n.name=='CommandsCfg')
    command_call = next(n.value for n in commands_cls.body if isinstance(n,ast.Assign))
    ranges = {k: list(ast.literal_eval(v)) for k,v in keywords(keywords(command_call)['limit_ranges']).items()}
    # Actor input and output definitions are deliberately fixed to this checkpoint.
    contract = {'format':'g1_velocity_11k_v1','observation_dim':480,'action_dim':29,'history_length':5,
                'history_layout':'term_major_oldest_first','normalizer_embedded':False,
                'step_dt':0.02,'action_scale':0.25,'angular_velocity_scale':0.2,'joint_velocity_scale':0.05,
                'policy_joint_names':policy_names,'sdk_joint_names':sdk_names,'joint_ids_map':mapping,
                'default_joint_pos':defaults,'kp_policy':kp,'kd_policy':kd,'effort_limit_sim':effort,
                'joint_lower':lower,'joint_upper':upper,'command_ranges':ranges,
                'observation_terms':['base_ang_vel','projected_gravity','velocity_commands','joint_pos_rel','joint_vel_rel','last_action'],
                'observation_term_dims':[3,3,3,29,29,29]}
    canonical = json.dumps(contract,sort_keys=True,separators=(',',':')).encode()
    contract_sha = hashlib.sha256(canonical).hexdigest()
    package.mkdir(parents=True,exist_ok=True)
    (package/'params').mkdir(exist_ok=True)
    (package/'exported').mkdir(exist_ok=True)
    (package/'params/deploy.yaml').write_text(yaml.safe_dump(contract,sort_keys=False),encoding='utf-8')
    c = torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert c['iter'] == 10999
    actor = torch.nn.Sequential(torch.nn.Linear(480,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),
                                torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,29)).eval()
    actor.load_state_dict({k[6:]:v for k,v in c['model_state_dict'].items() if k.startswith('actor.')},strict=True)
    model_path = package/'exported/policy.onnx'
    torch.onnx.export(actor,torch.zeros(1,480),str(model_path),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
    model = onnx.load(str(model_path))
    onnx.helper.set_model_props(model,{'checkpoint_sha256':CHECKPOINT_SHA256,'contract_sha256':contract_sha,'checkpoint_iter':'10999','training_commit':TRAINING_COMMIT})
    onnx.checker.check_model(model)
    onnx.save(model,str(model_path))
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(10999)
    max_error=0.
    torch.set_num_threads(1)
    for obs in rng.normal(size=(128,1,480)).astype(np.float32):
        with torch.no_grad(): expected=actor(torch.from_numpy(obs)).numpy()
        actual=session.run(['actions'],{'obs':obs})[0]
        max_error=max(max_error,float(np.max(np.abs(actual-expected))))
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    # A generated, reviewable C++ contract avoids a second hand-maintained parameter list.
    def arr(kind,name,values):
        def fmt(v):
            return str(int(v)) if kind=='int' else f'{float(v):.9g}'+('f' if '.' in f'{float(v):.9g}' or 'e' in f'{float(v):.9g}' else '.0f')
        return f'inline constexpr std::array<{kind}, {len(values)}> {name}{{'+', '.join(fmt(v) for v in values)+'};\n'
    header = '// Generated by tools/export_velocity11k.py from the verified 11k training contract.\n#pragma once\n#include <array>\nnamespace velocity11k {\n'
    header += f'inline constexpr char CHECKPOINT_SHA256[]="{CHECKPOINT_SHA256}";\ninline constexpr char CONTRACT_SHA256[]="{contract_sha}";\n'
    header += 'inline constexpr int J=29, H=5, O=480;\ninline constexpr float DT=0.02f, ACTION_SCALE=0.25f, ANGULAR_SCALE=0.2f, JOINT_VEL_SCALE=0.05f;\n'
    for kind,name,values in [('int','SDK_MAP',mapping),('float','DEFAULT_Q',defaults),('float','KP',kp),('float','KD',kd),('float','EFFORT_SIM',effort),('float','LOWER',lower),('float','UPPER',upper),
                             ('float','COMMAND_LOW',[ranges[k][0] for k in ('lin_vel_x','lin_vel_y','ang_vel_z')]),('float','COMMAND_HIGH',[ranges[k][1] for k in ('lin_vel_x','lin_vel_y','ang_vel_z')])]:
        header += arr(kind,name,values)
    header += '} // namespace velocity11k\n'
    header_path=REPO/'deploy/include/velocity11k/contract.h'
    header_path.parent.mkdir(parents=True,exist_ok=True)
    header_path.write_text(header,encoding='utf-8')
    manifest={'checkpoint_sha256':CHECKPOINT_SHA256,'checkpoint_iter':c['iter'],'source_task':'TASK_20261020_069','training_commit':TRAINING_COMMIT,
              'contract_sha256':contract_sha,'policy_sha256':sha(model_path),'deploy_yaml_sha256':sha(package/'params/deploy.yaml'),
              'cpp_contract_sha256':sha(header_path),'mjcf_sha256':sha(xml_path),'torch_version':torch.__version__,
              'onnx_version':onnx.__version__,'onnxruntime_version':ort.__version__,
              'export_check':{'input_cases':128,'max_abs_action_difference':max_error,'atol':2e-5,'rtol':2e-5},
              'contract':contract}
    (package/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='contract'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,default=REPO.parent/'6_sim-train/G1/unitree_rl_lab_train')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--package',type=Path,default=REPO/'deploy/robots/g1_29dof/config/policy/velocity/11k')
    args=parser.parse_args()
    generate(args.training_root.resolve(),args.checkpoint or args.training_root/'checkpoints/model_10999_factorygait.pt',args.package)
