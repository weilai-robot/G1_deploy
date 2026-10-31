#!/usr/bin/env python3
"""Offline-only checks of the live C++ cyclic planner / observation / target limits.

prepare: YAML -> a numeric fixture (probe has no YAML/ORT/DDS dependency).
check: frozen native Torch actor, independently checked ONNX export, MuJoCo feedback.
No robot communication. No environment installation.
"""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types


def prepare(package,output):
    import yaml
    cfg=yaml.safe_load((package/'deploy.yaml').read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    numbers=[cfg['step_dt'],*cfg['sdk_indices']]
    for key in ['default_joint_pos','kp','kd','action_scale','joint_lower','joint_upper','effort_limit']:
        numbers.extend(cfg[key])
    for key in ['torso','left_foot','right_foot']:
        chain=cfg['kinematics'][key];numbers.append(len(chain))
        for seg in chain:numbers.extend([seg['motor_index'],*seg['xyz'],*seg['rpy'],*seg['axis']])
    (output/'config.txt').write_text(' '.join(map(str,numbers))+'\n',encoding='utf-8')


class Probe:
    def __init__(self,binary,package,fixture,support):
        self.p=subprocess.Popen([str(binary),str(package/'gait.bin'),str(fixture),str(support)],
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    def send(self,command,values=()):
        self.p.stdin.write(command+' '+' '.join(map(str,values))+'\n');self.p.stdin.flush()
        line=self.p.stdout.readline()
        if not line:raise RuntimeError('C++ probe: '+self.p.stderr.read())
        if line.strip()=='OK':return None
        import numpy as np
        return np.fromstring(line,sep=' ')
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write('Q\n');self.p.stdin.flush()
        self.p.communicate(timeout=10)
        if self.p.returncode:raise RuntimeError('Probe exit '+str(self.p.returncode))


def matrix(q):
    import numpy as np
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True,type=Path)
    p.add_argument('--prepare',type=Path)
    p.add_argument('--fixture',type=Path)
    p.add_argument('--probe',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--runtime-dir',type=Path)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--mjcf',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--onnx-parity',action='store_true')
    p.add_argument('--truth-diagnostic',action='store_true')
    p.add_argument('--raw-targets-diagnostic',action='store_true')
    a=p.parse_args()
    if a.prepare:
        prepare(a.package,a.prepare);return
    import numpy as np
    import torch
    torch.set_num_threads(1)
    namespace=types.ModuleType('frozen_walk');namespace.__path__=[str(a.runtime_dir)]
    sys.modules['frozen_walk']=namespace
    RuntimeActor=importlib.import_module('frozen_walk.runtime_policy').RuntimeActor
    Engine=importlib.import_module('frozen_walk.velocity_gait_reference').VelocityGaitReference
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    native=checkpoint['infos']['oma_runtime'];actor=RuntimeActor(checkpoint).eval()
    manifest=json.loads((a.package/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match packaged ONNX')
    if a.onnx_parity:
        import onnxruntime as ort
        session=ort.InferenceSession(str(a.package/'policy.onnx'),providers=['CPUExecutionProvider'])
        x=torch.randn((64,160),generator=torch.Generator().manual_seed(17))*actor.std+actor.mean
        with torch.inference_mode():expected=actor(x).numpy()
        actual=np.concatenate([session.run(['actions'],{'obs':v[None]})[0] for v in x.numpy()])
        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)
        print(json.dumps({'model_id':manifest['model_id'],'samples':64,'max_abs_action_error':float(np.abs(actual-expected).max())}))
        return
    import mujoco
    cfg=json.loads((a.fixture/'config.json').read_text(encoding='utf-8'))
    if cfg['model_id']!=manifest['model_id']:raise ValueError('Fixture/model mismatch')
    protocol=json.loads(a.protocol.read_text(encoding='utf-8'))
    support=protocol['initial_support_index'];dt=native['policy_dt']
    a.output.mkdir(parents=True,exist_ok=True)
    engine=Engine(native['motion'],1,native['command_limits'],acceleration=native['phase_acceleration'])
    engine.reset(torch.tensor([0]),torch.tensor([support]),torch.zeros(1,2),torch.zeros(1))
    body_ids=[native['body_names'].index(n) for n in cfg['body_names']]
    report={'model_id':manifest['model_id'],'checkpoint_sha256':manifest['checkpoint_sha256'],
            'policy_sha256':manifest['policy_sha256'],'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
            'state_source':'truth diagnostic' if a.truth_diagnostic else 'C++ leg odometry',
            'inference':'frozen Torch actor; packaged ONNX parity checked separately',
            'target_source':'raw diagnostic' if a.raw_targets_diagnostic else 'live C++ joint/rate/PD-envelope limits',
            'mujoco_version':mujoco.__version__,'real_robot_tested':False,
            'reference_max_abs_error':0.,'reference_phase_max_abs_error':0.,'observation_max_abs_error':0.,
            'reference_steps':0,'observation_samples':0}
    # Exhaustive frozen protocol reference oracle: wraps, speed changes, turns, stops, restart.
    cpp=Probe(a.probe,a.package,a.fixture/'config.txt',support)
    for stage in protocol['stages']:
        enabled=stage.get('walk_enabled',True)
        engine.request(torch.tensor([stage['velocity']]),torch.tensor([enabled]))
        current=cpp.send('C',[*stage['velocity'],int(enabled)])
        for _ in range(round(stage['duration_s']/dt)):
            expected=np.concatenate([engine.joint_pos[0].numpy(),engine.joint_vel[0].numpy(),
                engine.body_pos_w[0,body_ids].numpy().reshape(-1),engine.body_quat_w[0,body_ids].numpy().reshape(-1)])
            phase_error=abs(current[0]-float(engine.clock.phase[0]))
            error=float(np.max(np.abs(current[9:]-expected)))
            report['reference_max_abs_error']=max(report['reference_max_abs_error'],error)
            report['reference_phase_max_abs_error']=max(report['reference_phase_max_abs_error'],phase_error)
            # Explicit float32 cross-language tolerance, not measured robot capability.
            np.testing.assert_allclose(current[9:],expected,rtol=2e-4,atol=2e-4)
            np.testing.assert_allclose(current[3:5],engine.root.position[0].numpy(),rtol=2e-4,atol=2e-4)
            assert current[2]==int(engine.clock.cycles[0])
            report['reference_steps']+=1
            engine.step(dt);current=cpp.send('N')
    cpp.close()
    # Invalid commands and sensor data must fail before action publication.
    zero=[0.]*65
    rejection={'bad_imu':'I '+' '.join(map(str,zero))+'\n',
               'unsupported_lateral':'C 0 .1 0 1\n','in_place_turn':'C 0 0 .1 1\n'}
    for name,packet in rejection.items():
        result=subprocess.run([str(a.probe),str(a.package/'gait.bin'),str(a.fixture/'config.txt'),str(support)],
                              input=packet,text=True,capture_output=True,timeout=10)
        if result.returncode==0:raise AssertionError('Did not reject '+name)
        report[name+'_rejection']='passed'
    cpp=Probe(a.probe,a.package,a.fixture/'config.txt',support)
    nominal=native['command_limits'][0][1]
    for axes,expected in [([0,0,0],[0,0,0]),([-1,0,1],[0,0,0]),([0,1,1],[0,0,0]),
                          ([1,0,0],[nominal,0,0]),([1,0,-1],[nominal,0,.2]),
                          ([1,0,1],[nominal,0,-.2])]:
        np.testing.assert_allclose(cpp.send('S',axes),expected,rtol=1e-6,atol=1e-7)
    cpp.close();report['joystick_direction_and_unsupported_axes']='passed'
    with tempfile.TemporaryDirectory(prefix='walking-probe-rejection-') as temporary:
        tmp=Path(temporary);(tmp/'truncated.bin').write_bytes((a.package/'gait.bin').read_bytes()[:64])
        config_text=(a.fixture/'config.txt').read_text().split()
        config_text[1:30]=['0']*29
        (tmp/'bad_config.txt').write_text(' '.join(config_text))
        for name,gait,config,index in [
            ('truncated_gait',tmp/'truncated.bin',a.fixture/'config.txt',str(support)),
            ('duplicate_sdk_indices',a.package/'gait.bin',tmp/'bad_config.txt',str(support)),
            ('invalid_initial_support',a.package/'gait.bin',a.fixture/'config.txt','10000')]:
            result=subprocess.run([str(a.probe),str(gait),str(config),index],input='',text=True,capture_output=True,timeout=10)
            if result.returncode==0:raise AssertionError('Did not reject '+name)
            report[name+'_rejection']='passed'
    # Ground-truth FK is an independent observation oracle; production uses kinematic odometry.
    model=mujoco.MjModel.from_xml_path(str(a.mjcf.resolve()));model.opt.timestep=native['physics_dt']
    data=mujoco.MjData(model);ids=np.array(cfg['sdk_indices'])
    joint_ids=np.array([model.joint(n).id for n in cfg['joint_names']])
    qp=model.jnt_qposadr[joint_ids];qv=model.jnt_dofadr[joint_ids]
    model.dof_armature[qv]=cfg['armature']
    sdk_qp=np.empty(29,int);sdk_qp[ids]=qp;sdk_qv=np.empty(29,int);sdk_qv[ids]=qv
    torso=model.body(cfg['anchor_body_name']).id
    def initialise(cpp):
        ref=cpp.send('T')[9:]
        mujoco.mj_resetData(model,data)
        data.qpos[:3]=ref[58:61];data.qpos[3:7]=ref[100:104];data.qpos[qp]=ref[:29]
        mujoco.mj_forward(model,data);cpp.send('I',packet())
    def packet():
        return [*data.qpos[sdk_qp],*data.qvel[sdk_qv],*data.qpos[3:7],*data.qvel[3:6]]
    cpp=Probe(a.probe,a.package,a.fixture/'config.txt',support);initialise(cpp)
    last=np.zeros(29);rng=np.random.default_rng(123)
    cpp.send('C',[native['command_limits'][0][1],0,0,1])
    for sample in range(64):
        ref=cpp.send('T')[9:]
        data.qpos[qp]=ref[:29]+rng.normal(0,.025,29)
        data.qvel[:]=rng.normal(0,.05,model.nv);mujoco.mj_forward(model,data)
        rot=matrix(data.qpos[3:7]);torso_r=matrix(data.xquat[torso])
        obs=cpp.send('O',[1,*packet(),*data.qpos[:3],*(rot.T@data.qvel[:3])])[:160]
        expected=np.concatenate([ref[:58],torso_r.T@(ref[79:82]-data.xpos[torso]),
             (torso_r.T@matrix(ref[128:132]))[:,:2].reshape(-1),rot.T@data.qvel[:3],data.qvel[3:6],
             data.qpos[qp]-np.asarray(cfg['default_joint_pos']),data.qvel[qv],last])
        np.testing.assert_allclose(obs,expected,rtol=1e-4,atol=3e-5)
        report['observation_max_abs_error']=max(report['observation_max_abs_error'],float(np.abs(obs-expected).max()))
        report['observation_samples']+=1
        with torch.inference_mode():last=actor(torch.tensor(obs[None],dtype=torch.float32))[0].numpy()
        cpp.send('A',last);cpp.send('N')
    cpp.close()
    cpp=Probe(a.probe,a.package,a.fixture/'config.txt',support);initialise(cpp)
    completed=0;reason=None;limited_steps=0;max_drift=0.;max_tilt=0.;min_height=float(data.qpos[2])
    header=['time_s','stage','phase','phase_rate','reference_x','reference_y','actual_x','actual_y','actual_z',
            'actual_vx_b','actual_vy_b','actual_vz_b','tilt','anchor_error','estimated_x','estimated_y','estimated_z',
            'position_estimation_error','limited_joints','reference_wz','actual_wz','actual_yaw','target_yaw']
    with (a.output/'states.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(header)
        for stage in protocol['stages']:
            current=cpp.send('C',[*stage['velocity'],int(stage.get('walk_enabled',True))])
            for _ in range(round(stage['duration_s']/dt)):
                rot=matrix(data.qpos[3:7]);truth_v=rot.T@data.qvel[:3]
                values=[int(a.truth_diagnostic),*packet()]
                if a.truth_diagnostic:values.extend([*data.qpos[:3],*truth_v])
                result=cpp.send('O',values);obs=result[:160]
                with torch.inference_mode():action=actor(torch.tensor(obs[None],dtype=torch.float32))[0].numpy()
                target=cpp.send('A',action);limited_steps+=target[-1]>0
                anchor=float(np.linalg.norm(obs[58:61]));drift=float(np.linalg.norm(result[160:163]-data.qpos[:3]))
                tilt=float(np.arccos(np.clip(rot[2,2],-1,1)))
                max_drift=max(max_drift,drift);max_tilt=max(max_tilt,tilt);min_height=min(min_height,float(data.qpos[2]))
                writer.writerow([completed*dt,stage['name'],*current[:2],*current[3:5],*data.qpos[:3],
                    *truth_v,tilt,anchor,*result[160:163],drift,int(target[-1]),stage['velocity'][2],
                    (rot@data.qvel[3:6])[2],float(np.arctan2(rot[1,0],rot[0,0])),current[5]])
                if tilt>protocol['abort_tilt_rad']:reason='pelvis_tilt_guard';break
                if data.qpos[2]<protocol['abort_pelvis_height_m']:reason='pelvis_height_abort';break
                if anchor>.35:reason='live_anchor_error_guard';break
                targets=target[:29][ids]
                if a.raw_targets_diagnostic:targets=np.array(cfg['default_joint_pos'])+np.array(cfg['action_scale'])*action
                for _ in range(native['decimation']):
                    torque=np.clip(np.array(cfg['kp'])*(targets-data.qpos[qp])-np.array(cfg['kd'])*data.qvel[qv],
                                   -np.array(cfg['effort_limit']),cfg['effort_limit'])
                    data.qfrc_applied[:]=0;data.qfrc_applied[qv]=torque;mujoco.mj_step(model,data)
                current=cpp.send('N');completed+=1
                if not np.isfinite(data.qpos).all():reason='non_finite_physics';break
            if reason:break
    cpp.close()
    planned=sum(round(s['duration_s']/dt) for s in protocol['stages'])
    report.update(planned_steps=planned,completed_steps=completed,completed_seconds=completed*dt,
                  completed_protocol=completed==planned and reason is None,termination_reason=reason,
                  limited_steps=int(limited_steps),maximum_position_estimation_error_m=max_drift,
                  maximum_tilt_rad=max_tilt,minimum_pelvis_height_m=min_height)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['completed_protocol']:sys.exit(2)


if __name__=='__main__':main()
