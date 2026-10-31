#!/usr/bin/env python3
"""Package an actual154 checkpoint for the observable C++ walking state."""
import argparse,copy,hashlib,importlib.util,json,struct
from pathlib import Path
import numpy as np
import torch
import yaml

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--runtime-source',required=True,type=Path,help='Matching training utils/runtime_policy.py')
    p.add_argument('--model-id',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fsm-output',required=True,type=Path)
    a=p.parse_args()
    if not a.model_id or any(not(c.isalnum() or c=='_') for c in a.model_id):raise ValueError('Invalid model label')
    if a.output.exists():raise FileExistsError('Never overwrite a frozen model package')
    root=Path(__file__).resolve().parents[1]
    cfgdir=root/'deploy/robots/g1_29dof/config'
    template=cfgdir/'policy/beyondmimic/walk_baseline'
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('diagnostic_synthetic'):raise ValueError('Synthetic interface fixtures are never deployable models')
    native=checkpoint['infos']['oma_runtime']
    if native['format']!='beyondmimic_g1_velocity_proprio_runtime_v1':raise ValueError('Expected new154 schema')
    spec=importlib.util.spec_from_file_location('runtime_policy',a.runtime_source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actor=module.RuntimeActor(checkpoint).eval()
    config=yaml.safe_load((template/'deploy.yaml').read_text(encoding='utf-8'))
    if native['joint_names']!=config['joint_names'] or native['tracked_body_names']!=config['body_names']:
        raise ValueError('Native/package articulation mismatch')
    for key in ['default_joint_pos','kp','kd','action_scale','effort_limit']:
        np.testing.assert_array_equal(np.array(config[key],np.float32),native['control'][key].numpy())
    a.output.mkdir(parents=True)
    torch.onnx.export(actor,torch.zeros(1,154),str(a.output/'policy.onnx'),
                      input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False)
    import onnxruntime as ort
    x=torch.randn((64,154),generator=torch.Generator().manual_seed(17))*actor.std+actor.mean
    with torch.inference_mode():expected=actor(x).numpy()
    session=ort.InferenceSession(str(a.output/'policy.onnx'),providers=['CPUExecutionProvider'])
    actual=np.concatenate([session.run(['actions'],{'obs':row[None]})[0] for row in x.numpy()])
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)
    motion=native['motion'];n=int(motion['cycle_frames'][0]);fps=float(motion['fps'][0])
    bodies=[native['body_names'].index(name) for name in config['body_names']]
    records=np.concatenate([motion['joint_pos'].numpy(),motion['joint_vel'].numpy(),
        motion['body_pos_w'][:,bodies].numpy().reshape(n,-1),motion['body_quat_w'][:,bodies].numpy().reshape(n,-1)],axis=1).astype('<f4')
    support=motion['support_frames'].numpy().astype('<u4')
    with (a.output/'gait.bin').open('wb') as f:
        f.write(struct.pack('<8sIIIIffff',b'BMGCYC1\0',n,29,14,len(support),native['policy_dt'],fps,
                float(motion['nominal_reference_speed_mps'][0]),native['phase_acceleration']))
        f.write(np.array(native['command_limits'],dtype='<f4').tobytes());f.write(support.tobytes());f.write(records.tobytes())
    config.update(format='beyondmimic_g1_154_v1',model_id=a.model_id,observation_dim=154,
        controller_mode='velocity_cyclic_proprio_v1',actor_observation_contract=native['actor_observation_contract'],
        armature=native['control']['armature'].tolist(),training_command_curriculum=native['training_command_curriculum'],
        cyclic_reference={'format':'BMGCYC1','frames':n,'support_frames':support.tolist(),
            'nominal_speed_mps':float(motion['nominal_reference_speed_mps'][0]),
            'command_limits':native['command_limits'],'phase_acceleration':native['phase_acceleration'],
            'initial_support_index':0})
    config['normalizer']['epsilon']=native['normalizer_epsilon']
    (a.output/'deploy.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8',newline='\n')
    manifest={'format':'beyondmimic_velocity_package_v1','model_id':a.model_id,'source_checkpoint':a.checkpoint.name,
        'checkpoint_sha256':sha(a.checkpoint),'source_commit':a.source_commit,
        'trained_reference_npz_sha256':native['reference_sha256'],'policy_sha256':sha(a.output/'policy.onnx'),
        'gait_sha256':sha(a.output/'gait.bin'),'deploy_yaml_sha256':sha(a.output/'deploy.yaml'),
        'frames':n,'fps':fps,'observation_dim':154,'actor_observation_contract':native['actor_observation_contract'],
        'normalizer_embedded':True,'normalizer_epsilon':native['normalizer_epsilon'],
        'onnx_parity':{'samples':64,'max_abs_error':float(np.abs(actual-expected).max())},
        'unsupported_commands':['backward','lateral','in_place_turn'],'real_robot_tested':False,
        'deployment_behavior_accepted':False,'global_position_guard':False,
        'state_source':'IMU/encoders/reference; no position or linear velocity estimator'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    fsm=yaml.safe_load((cfgdir/'beyondmimic_walk.yaml').read_text(encoding='utf-8'))
    settings=fsm['FSM']['BeyondMimicWalk']
    settings['policy_dir']=str(a.output.resolve().relative_to((root/'deploy/robots/g1_29dof').resolve())).replace('\\','/')
    settings.pop('max_anchor_error',None)
    ids=config['sdk_indices'];q=np.zeros(29);q[ids]=records[int(support[0]),:29]
    fsm['FSM']['FixStand']['qs'][1]=q.tolist()
    a.fsm_output.write_text(yaml.safe_dump(fsm,sort_keys=False),encoding='utf-8',newline='\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
