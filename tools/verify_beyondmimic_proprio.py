#!/usr/bin/env python3
"""Deployment154 replay: actual shared C++ observations/limits, no root-position/velocity input."""
import argparse,csv,hashlib,importlib.util,json,subprocess
from pathlib import Path
import numpy as np
import torch
import mujoco
from verify_beyondmimic_walk import Probe,matrix

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['package','fixture','probe','checkpoint','runtime-dir','protocol','mjcf','output']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument("--contract-only",action="store_true",help="Interface checks only; not a walking acceptance")
    a=p.parse_args();torch.set_num_threads(1)
    cfg=json.loads((a.fixture/'config.json').read_text())
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    native=checkpoint['infos']['oma_runtime']
    if native['format']!='beyondmimic_g1_velocity_proprio_runtime_v1':raise ValueError('154 schema required')
    m=json.loads((a.package/'manifest.json').read_text())
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=m['checkpoint_sha256']:raise ValueError('Wrong checkpoint')
    spec=importlib.util.spec_from_file_location('runtime_policy',a.runtime_dir/'runtime_policy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actor=module.RuntimeActor(checkpoint).eval()
    protocol=json.loads(a.protocol.read_text());support=protocol['initial_support_index'];dt=native['policy_dt']
    model=mujoco.MjModel.from_xml_path(str(a.mjcf.resolve()));model.opt.timestep=native['physics_dt']
    data=mujoco.MjData(model);ids=np.array(cfg['sdk_indices'])
    joint_ids=np.array([model.joint(n).id for n in cfg['joint_names']])
    qp=model.jnt_qposadr[joint_ids];qv=model.jnt_dofadr[joint_ids]
    model.dof_armature[qv]=cfg['armature']
    sdk_qp=np.empty(29,int);sdk_qp[ids]=qp;sdk_qv=np.empty(29,int);sdk_qv[ids]=qv
    torso=model.body(cfg['anchor_body_name']).id
    def packet():return [*data.qpos[sdk_qp],*data.qvel[sdk_qv],*data.qpos[3:7],*data.qvel[3:6]]
    def initialise(cpp):
        ref=cpp.send('T')[9:];mujoco.mj_resetData(model,data)
        data.qpos[:3]=ref[58:61];data.qpos[3:7]=ref[100:104];data.qpos[qp]=ref[:29]
        mujoco.mj_forward(model,data);cpp.send('I',packet())
    def controller():return Probe(a.probe,a.package,a.fixture/'config.txt',support,proprio=True)
    a.output.mkdir(parents=True,exist_ok=True)
    report={'model_id':m['model_id'],'checkpoint_sha256':m['checkpoint_sha256'],'observation_dim':154,
            'actor_state_source':'C++ IMU/encoders/reference only; no truth position or linear velocity',
            'leg_odometry_used':False,'position_error_guard_used':False,'target_limits':'live C++ unchanged',
            'real_robot_tested':False,'observation_max_abs_error':0.,'observation_samples':0}
    cpp=controller();initialise(cpp);last=np.zeros(29);rng=np.random.default_rng(17)
    for _ in range(64):
        ref=cpp.send('T')[9:]
        data.qpos[qp]=ref[:29]+rng.normal(0,.025,29);data.qvel[qv]=rng.normal(0,.1,29)
        data.qvel[3:6]=rng.normal(0,.05,3);mujoco.mj_forward(model,data)
        obs=cpp.send('O',[0,*packet()])
        if obs.shape!=(154,):raise AssertionError('Unexpected actor packet size')
        r=matrix(data.xquat[torso])
        expected=np.concatenate([ref[:58],(r.T@matrix(ref[128:132]))[:,:2].reshape(-1),
            data.qvel[3:6],data.qpos[qp]-np.array(cfg['default_joint_pos']),data.qvel[qv],last])
        np.testing.assert_allclose(obs,expected,rtol=1e-4,atol=3e-5)
        report['observation_max_abs_error']=max(report['observation_max_abs_error'],float(np.abs(obs-expected).max()))
        # Actual counterfactual sensor-equivalence test: world translation/velocity
        # change in MuJoCo, while the robot's available proprioceptive sensors stay fixed.
        saved_p=data.qpos[:3].copy();saved_v=data.qvel[:3].copy()
        data.qpos[:3]=rng.normal(0,100,3);data.qvel[:3]=rng.normal(0,10,3);mujoco.mj_forward(model,data)
        np.testing.assert_array_equal(cpp.send('O',[0,*packet()]),obs)
        data.qpos[:3]=saved_p;data.qvel[:3]=saved_v;mujoco.mj_forward(model,data)
        with torch.inference_mode():last=actor(torch.tensor(obs[None],dtype=torch.float32))[0].numpy()
        cpp.send('A',last);cpp.send('N');report['observation_samples']+=1
    cpp.close();report['world_position_velocity_counterfactual']='passed'
    bad=subprocess.run([str(a.probe),str(a.package/'gait.bin'),str(a.fixture/'config.txt'),str(support),'proprio'],
        input='I '+' '.join(map(str,packet()))+'\nO 1 '+' '.join(map(str,packet()))+' 0 0 0 0 0 0\n',
        capture_output=True,text=True,timeout=10)
    if bad.returncode==0:raise AssertionError('154 mode accepted truth-state actor packet')
    report['truth_packet_rejection']='passed'
    if a.contract_only:
        report['scope']='interface only; no learned behavior claim'
        (a.output/'contract-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return
    cpp=controller();initialise(cpp);completed=0;reason=None;limited=0
    max_tilt=0.;min_height=float(data.qpos[2])
    header=['time_s','stage','command_vx','command_wz','phase','phase_rate','actual_x','actual_y','actual_z',
        'actual_vx_b','actual_vy_b','actual_vz_b','tilt','actual_yaw','target_yaw','actual_wz','limited_joints']
    with (a.output/'states.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header)
        for stage in protocol['stages']:
            ref=cpp.send('C',[*stage['velocity'],int(stage.get('walk_enabled',True))])
            for _ in range(round(stage['duration_s']/dt)):
                r=matrix(data.qpos[3:7]);tilt=float(np.arccos(np.clip(r[2,2],-1,1)))
                if tilt>protocol['abort_tilt_rad']:reason='pelvis_tilt_guard';break
                if data.qpos[2]<protocol['abort_pelvis_height_m']:reason='pelvis_height_abort';break
                if ((data.qpos[qp]<cfg['joint_lower'])|(data.qpos[qp]>cfg['joint_upper'])).any():
                    report['joint_range_violations']=[{'joint':cfg['joint_names'][i],'position_rad':float(data.qpos[qp][i]),'lower_rad':cfg['joint_lower'][i],'upper_rad':cfg['joint_upper'][i]} for i in range(29) if data.qpos[qp][i]<cfg['joint_lower'][i] or data.qpos[qp][i]>cfg['joint_upper'][i]]
                    reason='measured_joint_range_guard';break
                obs=cpp.send('O',[0,*packet()])
                with torch.inference_mode():action=actor(torch.tensor(obs[None],dtype=torch.float32))[0].numpy()
                target=cpp.send('A',action);limited+=target[-1]>0
                writer.writerow([completed*dt,stage['name'],stage['velocity'][0],stage['velocity'][2],*ref[:2],
                    *data.qpos[:3],*(r.T@data.qvel[:3]),tilt,np.arctan2(r[1,0],r[0,0]),ref[5],
                    (r@data.qvel[3:6])[2],int(target[-1])])
                targets=target[:29][ids]
                for _ in range(native['decimation']):
                    torque=np.clip(np.array(cfg['kp'])*(targets-data.qpos[qp])-np.array(cfg['kd'])*data.qvel[qv],
                                   -np.array(cfg['effort_limit']),cfg['effort_limit'])
                    data.qfrc_applied[:]=0;data.qfrc_applied[qv]=torque;mujoco.mj_step(model,data)
                ref=cpp.send('N');completed+=1
                final_tilt=float(np.arccos(np.clip(matrix(data.qpos[3:7])[2,2],-1,1)))
                max_tilt=max(max_tilt,final_tilt);min_height=min(min_height,float(data.qpos[2]))
                if not np.isfinite(data.qpos).all():reason='non_finite_physics';break
                if final_tilt>protocol['abort_tilt_rad']:reason='pelvis_tilt_guard';break
                if data.qpos[2]<protocol['abort_pelvis_height_m']:reason='pelvis_height_abort';break
            if reason:break
    cpp.close()
    planned=sum(round(s['duration_s']/dt) for s in protocol['stages'])
    report.update(planned_steps=planned,completed_steps=completed,completed_seconds=completed*dt,
        completed_protocol=completed==planned and reason is None,termination_reason=reason,
        limited_steps=int(limited),maximum_tilt_rad=max_tilt,minimum_pelvis_height_m=min_height,
        behavior_acceptance=False,scope='deployment-interface physical replay; full speed/contact/video gates remain required')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['completed_protocol']:raise SystemExit(2)
if __name__=='__main__':main()
