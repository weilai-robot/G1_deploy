#!/usr/bin/env python3
"""Import the two frozen loop artifacts; robot runtime needs neither Torch nor training code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import torch
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'deploy/robots/g1_29dof/config/policy/beyondmimic')
    a=p.parse_args()
    variants=[('baseline_exp003_model3998','walk_baseline'),('contact_exp004_model5997','walk_contact')]
    for source,name in variants:
        archive=a.archive/source
        original=json.loads((archive/'model_manifest.json').read_text(encoding='utf-8'))
        checkpoint=archive/original['checkpoint']
        if sha(checkpoint)!=original['checkpointSHA256']:
            raise ValueError('Checkpoint hash differs from frozen owner archive')
        native=torch.load(checkpoint,map_location='cpu',weights_only=True)['infos']['oma_runtime']
        if native['format']!='beyondmimic_g1_velocity_runtime_v1':
            raise ValueError('Not a velocity-command model')
        config=yaml.safe_load((archive/'offline_fixture/deploy.yaml').read_text(encoding='utf-8'))
        if config['joint_names']!=native['joint_names'] or config['body_names']!=native['tracked_body_names']:
            raise ValueError('Deployment articulation order differs from checkpoint')
        for key in ['default_joint_pos','kp','kd','action_scale','effort_limit']:
            np.testing.assert_array_equal(np.asarray(config[key],np.float32),native['control'][key].numpy())
        out=a.output/name
        out.mkdir(parents=True,exist_ok=True)
        expected=original['onnxExportVerification']['policy_sha256']
        if sha(archive/'policy.onnx')!=expected:
            raise ValueError('Frozen ONNX hash mismatch')
        shutil.copyfile(archive/'policy.onnx',out/'policy.onnx')
        motion=native['motion'];n=int(motion['cycle_frames'][0]);fps=float(motion['fps'][0])
        bodies=[native['body_names'].index(x) for x in config['body_names']]
        q=motion['joint_pos'].numpy();dq=motion['joint_vel'].numpy()
        pos=motion['body_pos_w'][:,bodies].numpy();quat=motion['body_quat_w'][:,bodies].numpy()
        records=np.concatenate([q,dq,pos.reshape(n,-1),quat.reshape(n,-1)],axis=1).astype('<f4')
        support=motion['support_frames'].numpy().astype('<u4')
        if records.shape!=(n,156) or not np.isfinite(records).all() or len(support)<2:
            raise ValueError('Invalid exact cyclic reference')
        nominal=float(motion['nominal_reference_speed_mps'][0]);dt=native['policy_dt']
        with (out/'gait.bin').open('wb') as f:
            f.write(struct.pack('<8sIIIIffff',b'BMGCYC1\0',n,29,14,len(support),dt,fps,nominal,native['phase_acceleration']))
            f.write(np.asarray(native['command_limits'],dtype='<f4').tobytes())
            f.write(support.tobytes());f.write(records.tobytes())
        config.update({'model_id':source,'controller_mode':'velocity_cyclic_v1','reference':'gait.bin',
                       'armature':native['control']['armature'].tolist(),
                       'cyclic_reference':{'format':'BMGCYC1','frames':n,'support_frames':support.tolist(),
                           'nominal_speed_mps':nominal,'command_limits':native['command_limits'],
                           'phase_acceleration':native['phase_acceleration'],'initial_support_index':0},
                       'training_command_curriculum':native['training_command_curriculum']})
        (out/'deploy.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8',newline='\n')
        first_q=records[int(support[0]),:29]
        sdk_q=np.zeros(29);sdk_kp=np.zeros(29);sdk_kd=np.zeros(29)
        sdk_q[config['sdk_indices']]=first_q
        sdk_kp[config['sdk_indices']]=config['kp'];sdk_kd[config['sdk_indices']]=config['kd']
        manifest={'format':'beyondmimic_velocity_package_v1','model_id':source,
                  'source_checkpoint':original['checkpoint'],'checkpoint_sha256':original['checkpointSHA256'],
                  'source_commit':original['sourceCommit'],'trained_reference_npz_sha256':native['reference_sha256'],
                  'policy_sha256':sha(out/'policy.onnx'),'gait_sha256':sha(out/'gait.bin'),
                  'deploy_yaml_sha256':sha(out/'deploy.yaml'),'frames':n,'fps':fps,
                  'normalizer_embedded':True,'normalizer_epsilon':native['normalizer_epsilon'],
                  'onnx_parity':original['onnxExportVerification'],
                  'unsupported_commands':['backward','lateral','in_place_turn'],
                  'reference_scope':'Infinite cyclic gait, command-root integration and support-aware braking; not finite reference playback',
                  'real_robot_tested':False}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
        # Retain the repository's existing safety limits; deadzone is an explicit input setting.
        fsm={'FSM':{'_':{'Passive':{'id':1},'FixStand':{'id':2},'BeyondMimicWalk':{'id':202,'type':'BeyondMimicWalk'}},
             'Passive':{'transitions':{'FixStand':'LT + up.on_pressed'},'mode':[1]*29,'kd':[3.0]*29},
             'FixStand':{'transitions':{'Passive':'LT + B.on_pressed','BeyondMimicWalk':'RB + X.on_pressed'},
                         'kp':sdk_kp.tolist(),'kd':sdk_kd.tolist(),'ts':[0,3],'qs':[[],sdk_q.tolist()]},
             'BeyondMimicWalk':{'transitions':{'Passive':'LT + B.on_pressed'},
                 'policy_dir':'config/policy/beyondmimic/'+name,'initial_support_index':0,
                 'command_deadzone':0.1,'minimum_speed_scale':native['training_command_curriculum']['speed_scale_range'][0],
                 'max_start_error':0.25,'max_tilt':0.85,'max_target_speed':8.0,'max_anchor_error':0.35,
                 'log_directory':'log/beyondmimic_walk'}}}
        (a.output.parents[1]/('beyondmimic_walk'+('_contact' if name=='walk_contact' else '')+'.yaml')).write_text(yaml.safe_dump(fsm,sort_keys=False),encoding='utf-8',newline='\n')
        print(json.dumps({'model':name,'checkpoint_sha256':manifest['checkpoint_sha256'],'gait_sha256':manifest['gait_sha256']}))


if __name__=='__main__':
    main()
