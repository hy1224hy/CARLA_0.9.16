"""Matched final-policy replay: fixed case plus explicitly selected L2 success."""
import json
from pathlib import Path

import torch
from PIL import Image,ImageDraw
from highway_hybrid.config import HybridConfig
from highway_hybrid.environment import HybridEnv
from highway_hybrid.model import HybridActorCritic
from highway_hybrid.runner import load_checkpoint,source_digest

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'l1_l2_latest_gifs'


def rollout(level,method,case,kind):
    name=f'L{level}_{method}_s43'
    folder=ROOT/f'l{level}_latest_500k'/'audit_A'/name
    state=load_checkpoint(folder/'checkpoint.pt')
    assert state['train_steps']==500000 and state['source_sha256']==source_digest()
    cfg=HybridConfig.from_dict(state['config'])
    model=HybridActorCritic(cfg.hidden,cfg.state_dim)
    model.load_state_dict(state['models']['A']);model.eval()
    env=HybridEnv(cfg,render_mode='rgb_array');frames=[]
    try:
        obs,_=env.reset(seed=case,options={'family':'adjacent'})
        initial=dict(attacker=env.attacker.position.tolist(),target=env.target.position.tolist(),speed=float(env.target.speed))
        with torch.no_grad():
            while True:
                acceleration,lane=model.deterministic(torch.as_tensor(obs['state']),torch.as_tensor(obs['lane_mask']))
                obs,reward,term,trunc,info=env.step(dict(acceleration=acceleration.numpy(),lane=int(lane)))
                frame=Image.new('RGB',(900,300),'#18212b');frame.paste(Image.fromarray(env.render()),(0,45))
                draw=ImageDraw.Draw(frame)
                draw.text((12,5),f'L{level} | {"PPO" if method=="independent" else "Mut-PPO"} | seed43 / A / latest500k | case={case}',fill='white')
                draw.text((12,23),f'BLUE attacker / RED IDM target | {kind} | deterministic',fill='white')
                draw.text((12,284),f'cuts={info["cut_count"]} delay={info["red_delay_m"]:.1f}m block={info["blocking_time"]:.1f}s collision={info["collision"]} grade={info["grade_violation"]}'+(' END' if term or trunc else ''),fill='white')
                frames.append(frame)
                if term or trunc:break
        expected=next(json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines() if json.loads(line)['case_seed']==case)
        assert info['success']==expected['success'] and info['cut_count']==expected['cut_count'] and info['collision']==expected['collision']
        result=dict(level=level,method=method,train_seed=43,team='A',steps=500000,case_seed=case,initial=initial,
            success=info['success'],cut_count=info['cut_count'],collision=info['collision'],grade_violation=info['grade_violation'],
            blocking_time=info['blocking_time'],red_delay_m=info['red_delay_m'],frames=len(frames))
    finally:env.close()
    return frames,result


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    source=ROOT/'l2_latest_500k'/'audit_A'/'L2_mutual_s43'/'episodes.jsonl'
    success=sorted(json.loads(line)['case_seed'] for line in source.read_text().splitlines() if json.loads(line)['attack_success'] and json.loads(line)['family']=='adjacent')
    assert success
    choices=[(4000001,'fixed_case'),(success[0],'L2_MutPPO_success_example')]
    manifest=[]
    for case,kind in choices:
        videos=[];results=[]
        for level,method in [(1,'independent'),(1,'mutual'),(2,'independent'),(2,'mutual')]:
            frames,result=rollout(level,method,case,kind);videos.append(frames);results.append(result)
        assert all(r['initial']==results[0]['initial'] for r in results)
        combined=[]
        for i in range(max(map(len,videos))):
            frame=Image.new('RGB',(1800,600))
            for k,video in enumerate(videos):
                tile=video[min(i,len(video)-1)].copy()
                if i>=len(video):ImageDraw.Draw(tile).text((700,23),'TERMINATED: FRAME HELD',fill='#ffcc33')
                frame.paste(tile,((k%2)*900,(k//2)*300))
            combined.append(frame)
        path=OUT/f'L1_L2_s43_A_{kind}_{case}.gif'
        combined[0].save(path,save_all=True,append_images=combined[1:],duration=200,loop=0)
        combined[len(combined)//2].save(path.with_suffix('.png'))
        manifest.append(dict(gif=path.name,selection=kind,results=results))
        print(json.dumps(manifest[-1]),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
