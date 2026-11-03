#!/usr/bin/env python3
"""Import the owner-selected154 pair from its checksummed archive; robot needs no Torch."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
import torch,yaml

FILES=('policy.onnx','gait.bin','deploy.yaml','manifest.json')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def inside(base,rel):
 p=(base/rel).resolve()
 if not p.is_relative_to(base.resolve()):raise ValueError('Archive path escapes selected archive')
 return p

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
 p.add_argument('--replace',action='store_true',help='Explicitly replace the two existing walking slots after staging both packages')
 a=p.parse_args();root=Path(__file__).resolve().parents[1];robot=root/'deploy/robots/g1_29dof';cfg=robot/'config';base=cfg/'policy/beyondmimic'
 archive=a.archive.resolve();pair=json.loads((archive/'pair.json').read_text(encoding='utf8'))
 if pair.get('format')!='beyondmimic_owner_selected_proprio_pair_v1':raise ValueError('Expected owner-selected154 pair archive')
 entries=pair['models'];slots={'walk_baseline':'beyondmimic_walk.yaml','walk_contact':'beyondmimic_walk_contact.yaml'}
 if len(entries)!=2 or {e['package']for e in entries}!=set(slots):raise ValueError('Exactly stable/fluid walking slots required')
 if not a.replace and any((base/e['package']).exists()for e in entries):raise FileExistsError('Existing pair: require explicit --replace')
 # Validate BOTH actual sources before creating or replacing a deployment artifact.
 for e in entries:
  f=inside(archive,e['checkpoint']);runtime=inside(archive,e['runtime_source'])
  if sha(f)!=e['checkpoint_sha256']:raise ValueError('Selected checkpoint SHA mismatch')
  c=torch.load(f,map_location='cpu',weights_only=True);n=c['infos']['oma_runtime']
  if c.get('diagnostic_synthetic') or n['format']!='beyondmimic_g1_velocity_proprio_runtime_v1':raise ValueError('Actual154 checkpoint required')
  if not all(torch.isfinite(v).all()for v in c['model_state_dict'].values()):raise ValueError('Nonfinite selected actor')
  wanted=n['actor_observation_contract']['implementation_sha256']['utils/runtime_policy.py']
  if hashlib.sha256(runtime.read_text(encoding='utf8').encode()).hexdigest()!=wanted:raise ValueError('Archived runtime source differs from actual checkpoint')
  if not n.get('deployment_target_contract',{}).get('applied_to_native_action'):raise ValueError('Native deployed-target training contract required')
  if e['fsm']!=slots[e['package']]:raise ValueError('Selected model slot/config mismatch')
 base.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='._selected_pair_',dir=base)as tmp:
  stage=Path(tmp).resolve()
  if not stage.is_relative_to(base.resolve()):raise ValueError('Staging path escapes policy directory')
  index={'format':'beyondmimic_active_walk_pair_v1','real_robot_tested':False,'overall_goal_accepted':False,'models':[]}
  for e in entries:
   out=stage/e['package'];fsm=stage/e['fsm']
   subprocess.run([sys.executable,str(root/'tools/export_beyondmimic_proprio.py'),'--checkpoint',str(inside(archive,e['checkpoint'])),'--runtime-source',str(inside(archive,e['runtime_source'])),'--model-id',e['model_id'],'--source-commit',e['source_commit'],'--output',str(out),'--fsm-output',str(fsm)],check=True,capture_output=True,text=True)
   config=yaml.safe_load(fsm.read_text(encoding='utf8'));config['FSM']['BeyondMimicWalk']['policy_dir']='config/policy/beyondmimic/'+e['package'];fsm.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8',newline='\n')
   m=json.loads((out/'manifest.json').read_text());summary=json.loads(inside(archive,e['summary']).read_text(encoding='utf8'))
   m.update(source_task_id=e['task_id'],source_experiment=e['experiment'],variant_role=e['label'],owner_characterization=e['owner_characterization'],owner_selected_for_comparison=True,offline_functional_pass_count=summary['quantitativePassed'],offline_functional_total=summary['total'],offline_contact_gates_all_passed=summary['qualityAllPassed'])
   (out/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
   index['models'].append({'package':e['package'],'model_id':e['model_id'],'label':e['label'],'owner_characterization':e['owner_characterization'],'checkpoint':e['checkpoint'].split('/')[-1],'checkpoint_sha256':e['checkpoint_sha256'],'source_task_id':e['task_id'],'source_experiment':e['experiment'],'source_commit':e['source_commit'],'fsm_config':'config/'+e['fsm'],'fsm_sha256':sha(fsm),'manifest_sha256':sha(out/'manifest.json'),'observation_dim':154})
  # Keep overwritten files outside Git; all sources were validated before this point.
  backup=archive/'deployment_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');backup.mkdir(parents=True)
  for e in entries:
   old=base/e['package']
   if old.exists():
    dest=backup/e['package'];dest.mkdir()
    for name in FILES:shutil.copyfile(old/name,dest/name)
   if (cfg/e['fsm']).exists():shutil.copyfile(cfg/e['fsm'],backup/e['fsm'])
  for e in entries:
   out=base/e['package'];out.mkdir(exist_ok=True)
   for name in FILES:os.replace(stage/e['package']/name,out/name)
   os.replace(stage/e['fsm'],cfg/e['fsm'])
  (base/'walk_models.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
  print(json.dumps({'replaced':[e['model_id']for e in entries],'backup':str(backup),'robot_commands_sent':False},ensure_ascii=False))

if __name__=='__main__':main()
