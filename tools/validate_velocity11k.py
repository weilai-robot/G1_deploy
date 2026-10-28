#!/usr/bin/env python3
"""Validate the production C++ 11k Core/Runner, optionally in a MuJoCo closed loop.

Requires numpy, torch and onnxruntime. --closed-loop additionally needs mujoco.
Does not use DDS or send robot commands.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[1]
PACKAGE=REPO/'deploy/robots/g1_29dof/config/policy/velocity/11k'


class Process:
    def __init__(self,binary,model):
        self.proc=subprocess.Popen([str(binary),str(model)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    def step(self,snapshot,reset=False):
        values=[int(reset),*snapshot['quat'],*snapshot['omega'],*snapshot['q'],*snapshot['dq'],*snapshot['command']]
        self.proc.stdin.write(' '.join(format(float(v),'.9g') for v in values)+'\n')
        self.proc.stdin.flush()
        result=[]
        for dimension in (480,29,29):
            line=self.proc.stdout.readline()
            if not line:
                raise RuntimeError('C++ offline process stopped: '+self.proc.stderr.read())
            array=np.fromstring(line,sep=' ',dtype=np.float32)
            if array.shape!=(dimension,) or not np.isfinite(array).all():
                raise RuntimeError('Malformed C++ offline output')
            result.append(array)
        return result
    def close(self):
        if not self.proc.stdin.closed: self.proc.stdin.close()
        self.proc.wait(timeout=10)
        error=self.proc.stderr.read()
        if self.proc.returncode: raise RuntimeError(error)


class Reference:
    """Independent NumPy implementation of the training observation contract."""
    def __init__(self,cfg):
        self.cfg=cfg; self.histories=None; self.previous=np.zeros(29)
    def observe(self,s,reset):
        mapping=np.array(self.cfg['joint_ids_map'])
        quat=np.array(s['quat'],dtype=np.float32).astype(float)
        quat/=np.linalg.norm(quat)
        w,x,y,z=quat
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                           [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                           [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        limits=self.cfg['command_ranges']
        low=[limits[k][0] for k in ['lin_vel_x','lin_vel_y','ang_vel_z']]
        high=[limits[k][1] for k in ['lin_vel_x','lin_vel_y','ang_vel_z']]
        if reset:self.previous=np.zeros(29)
        frames=[np.asarray(s['omega'],dtype=np.float32)*self.cfg['angular_velocity_scale'],rotation.T@np.array([0.,0.,-1.]),
                np.clip(np.array(s['command'],dtype=np.float32),low,high),
                np.array(s['q'],dtype=np.float32)[mapping]-self.cfg['default_joint_pos'],
                np.array(s['dq'],dtype=np.float32)[mapping]*self.cfg['joint_velocity_scale'],self.previous.copy()]
        if reset:self.histories=[np.tile(f,(self.cfg['history_length'],1)) for f in frames]
        else:
            for history,frame in zip(self.histories,frames):
                history[:-1]=history[1:]; history[-1]=frame
        return np.concatenate([h.flatten() for h in self.histories]).astype(np.float32)
    def targets(self,action):
        policy=np.clip(np.array(self.cfg['default_joint_pos'])+self.cfg['action_scale']*action,self.cfg['joint_lower'],self.cfg['joint_upper'])
        sdk=np.empty(29)
        sdk[self.cfg['joint_ids_map']]=policy
        self.previous=action.copy()
        return sdk


def actor(training,manifest):
    import torch
    checkpoint=training/'checkpoints/model_10999_factorygait.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('Training checkpoint provenance mismatch')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=torch.nn.Sequential(torch.nn.Linear(480,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),
                              torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,29)).eval()
    model.load_state_dict({k[6:]:v for k,v in saved['model_state_dict'].items() if k.startswith('actor.')})
    torch.set_num_threads(1)
    return model


def parity(binary,model_path,cfg,torch_actor):
    import torch
    process=Process(binary,model_path); ref=Reference(cfg)
    rng=np.random.default_rng(10999)
    default_sdk=np.empty(29); default_sdk[cfg['joint_ids_map']]=cfg['default_joint_pos']
    maxima={'observation':0.,'action':0.,'target':0.}
    for i in range(128):
        quat=rng.normal(size=4); quat/=np.linalg.norm(quat)
        s={'quat':quat,'omega':rng.normal(size=3),'q':default_sdk+rng.normal(0,.02,size=29),
           'dq':rng.normal(size=29),'command':rng.normal(size=3)}
        # Exercise re-entry reset and command clipping, including a repeated reset.
        reset=i in [0,32,33,64]
        obs,action,target=process.step(s,reset)
        expected_obs=ref.observe(s,reset)
        with torch.no_grad(): expected_action=torch_actor(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
        expected_target=ref.targets(action)
        np.testing.assert_allclose(obs,expected_obs,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(action,expected_action,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(target,expected_target,atol=2e-6,rtol=2e-6)
        for key,actual,expected in [('observation',obs,expected_obs),('action',action,expected_action),('target',target,expected_target)]:
            maxima[key]=max(maxima[key],float(np.max(np.abs(actual-expected))))
    process.close()
    # Production Runner rejects an existing incompatible model instead of silently loading it.
    old=REPO/'deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx'
    with tempfile.TemporaryDirectory(prefix='g1_11k_reject_') as directory:
        old_copy=Path(directory)/'old_policy.onnx'
        shutil.copyfile(old,old_copy)
        rejected=subprocess.run([str(binary),str(old_copy)],capture_output=True,text=True,timeout=10)
    if rejected.returncode==0 or not any(text in rejected.stderr for text in ['provenance','must have float']):
        raise AssertionError('Old model was not rejected by the model contract: '+rejected.stderr)
    return {'cases':128,'reset_cases':[0,32,33,64],'max_abs_difference':maxima,'old_model_rejected':True,
            'tolerance':{'obs_atol_rtol':2e-6,'target_atol_rtol':2e-6,'action_atol_rtol':2e-5}}


def rollout(binary,model_path,cfg,training,xml_path,output):
    import csv
    import mujoco
    mapping=np.array(cfg['joint_ids_map'])
    model=mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep=.005
    model.dof_armature[6:35]=.01
    model.sensor_noise[:]=0; model.sensor_cutoff[:]=0
    data=mujoco.MjData(model)
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in cfg['policy_joint_names']]
    qadr=model.jnt_qposadr[joints]; vadr=model.jnt_dofadr[joints]
    cadr=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in cfg['policy_joint_names']]
    assert min(cadr)>=0
    sensor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,'imu-pelvis-angular-velocity')
    assert sensor>=0
    gyro=model.sensor_adr[sensor]
    data.qpos[:]=0; data.qpos[2]=.8; data.qpos[3]=1; data.qpos[qadr]=cfg['default_joint_pos']
    data.qvel[:]=0; mujoco.mj_forward(model,data)
    # Same command sequence as the baseline audit; commands enter history before inference,
    # matching the production controller's current-state observation update.
    segments=[(4,[0,0,0]),(8,[.4,0,0]),(8,[.6,0,0]),(6,[.8,0,0]),(6,[0,.3,0]),(6,[.4,0,.2]),(4,[0,0,0])]
    ends=np.cumsum([s[0] for s in segments])
    process=Process(binary,model_path)
    fall_samples=0; rows=[]; parity_max=0.
    reference=Reference(cfg)
    for step in range(round(ends[-1]/cfg['step_dt'])):
        index=int(np.searchsorted(ends,step*cfg['step_dt'],side='right'))
        command=np.array(segments[index][1])
        q_sdk=np.empty(29); q_sdk[mapping]=data.qpos[qadr]
        dq_sdk=np.empty(29); dq_sdk[mapping]=data.qvel[vadr]
        s={'quat':data.qpos[3:7].copy(),'omega':data.sensordata[gyro:gyro+3].copy(),'q':q_sdk,'dq':dq_sdk,'command':command}
        obs,action,target_sdk=process.step(s,step==0)
        expected=reference.observe(s,step==0)
        np.testing.assert_allclose(obs,expected,atol=2e-6,rtol=2e-6)
        parity_max=max(parity_max,float(np.max(np.abs(obs-expected))))
        reference.targets(action)
        target=target_sdk[mapping]
        for _ in range(4):
            tau=np.clip(np.asarray(cfg['kp_policy'])*(target-data.qpos[qadr])-np.asarray(cfg['kd_policy'])*data.qvel[vadr],-np.asarray(cfg['effort_limit_sim']),np.asarray(cfg['effort_limit_sim']))
            data.ctrl[cadr]=tau
            mujoco.mj_step(model,data)
        w,x,y,z=data.qpos[3:7]
        roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)); pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        tilt_cos=-(1-2*(x*x+y*y))
        fallen=int(data.qpos[2]<.2 or tilt_cos>-np.cos(.8)); fall_samples+=fallen
        vx=np.cos(yaw)*data.qvel[0]+np.sin(yaw)*data.qvel[1]
        vy=-np.sin(yaw)*data.qvel[0]+np.cos(yaw)*data.qvel[1]
        row={'t':data.time,'cmd_vx':command[0],'cmd_vy':command[1],'cmd_wz':command[2],
             'base_z':data.qpos[2],'roll':roll,'pitch':pitch,'yaw':yaw,'vx_yaw':vx,'vy_yaw':vy,'wz':data.sensordata[gyro+2],'fall':fallen}
        for side in ['left','right']:
            name=f'{side}_shoulder_pitch_joint'
            row[name]=data.qpos[qadr[cfg['policy_joint_names'].index(name)]]
        rows.append(row)
    process.close()
    log=output.with_suffix('.rollout.csv')
    with log.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    segment_results=[]
    start=0.
    for index,(duration,command) in enumerate(segments):
        steady=[r for r in rows if r['t']>=start+1.5 and r['t']<start+duration]
        segment_results.append({'command':command,'steady_window_s':[start+1.5,start+duration],
                                'actual_velocity_mean':[float(np.mean([r[k] for r in steady])) for k in ['vx_yaw','vy_yaw','wz']],
                                'pitch_std_deg':float(np.degrees([r['pitch'] for r in steady]).std()),
                                'roll_std_deg':float(np.degrees([r['roll'] for r in steady]).std()),
                                'shoulder_p10_p90_deg':{side:float(np.degrees(np.ptp(np.percentile([r[f'{side}_shoulder_pitch_joint'] for r in steady],[10,90])))) for side in ['left','right']}})
        start+=duration
    return {'source':str(log),'source_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'mujoco_version':mujoco.__version__,
            'mjcf_sha256':hashlib.sha256(xml_path.read_bytes()).hexdigest(),'steps':len(rows),'fall_samples':fall_samples,
            'observation_parity_max_abs_difference':parity_max,'segments':segment_results,
            'schedule':'Current snapshot/command enters history before inference, as on the real controller.',
            'torque_model':'Explicit PD with training effort_limit_sim in MuJoCo; hardware uses SDK position PD and firmware torque limits.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-bin',type=Path,required=True)
    parser.add_argument('--training-root',type=Path,default=REPO.parent/'6_sim-train/G1/unitree_rl_lab_train')
    parser.add_argument('--model',type=Path,default=PACKAGE/'exported/policy.onnx')
    parser.add_argument('--closed-loop',action='store_true')
    parser.add_argument('--xml',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((PACKAGE/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(args.model.read_bytes()).hexdigest()!=manifest['policy_sha256']:raise ValueError('Model hash mismatch')
    model=actor(args.training_root,manifest)
    result={'checkpoint_sha256':manifest['checkpoint_sha256'],'policy_sha256':manifest['policy_sha256'],
            'contract_sha256':manifest['contract_sha256'],'platform':sys.platform,
            'offline_binary_sha256':hashlib.sha256(args.offline_bin.read_bytes()).hexdigest(),
            'production_core_sha256':hashlib.sha256((REPO/'deploy/include/velocity11k/core.h').read_bytes()).hexdigest(),
            'production_runner_sha256':hashlib.sha256((REPO/'deploy/include/velocity11k/runner.h').read_bytes()).hexdigest(),
            'cpp_onnxruntime_version':subprocess.check_output([str(args.offline_bin),'--version'],text=True).strip(),
            'parity':parity(args.offline_bin,args.model,manifest['contract'],model),
            'hardware_tested':False,'linux_dds_controller_built':False}
    if args.closed_loop:
        if not args.xml:parser.error('--closed-loop requires --xml')
        if hashlib.sha256(args.xml.read_bytes()).hexdigest()!=manifest['mjcf_sha256']:raise ValueError('MJCF provenance mismatch')
        result['rollout']=rollout(args.offline_bin,args.model,manifest['contract'],args.training_root,args.xml,args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
