"""Native environment replay and physical trace, using fixed scenario IDs."""
import json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from highway_eval.evaluate import plot_trace
from .config import StudyConfig
from .env import StudyEnv
from .runtime import load,check,ActorCritic


def replay(run,output,checkpoint='latest',team='A',case_seed=4000003,family='adjacent'):
    state=load(Path(run)/(checkpoint+'.pt'));check(state)
    cfg=StudyConfig.from_dict(state['config']);torch.set_num_threads(cfg.threads)
    model=ActorCritic(cfg.hidden,cfg.state_dim)
    chosen=state['selection']['team'] if checkpoint=='selected' else team
    model.load_state_dict(state['model'] if checkpoint=='selected' else state['models'][team]);model.eval()
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    frames=[];trace=[];env=StudyEnv(cfg)
    try:
        obs,_=env.reset(seed=case_seed,options={'family':family})
        with torch.no_grad():
            while True:
                acc,lane=model.deterministic(torch.as_tensor(obs['state']),torch.as_tensor(obs['lane_mask']))
                obs,_,term,trunc,info=env.step(dict(acceleration=acc.numpy(),lane=int(lane)))
                frame=Image.new('RGB',(900,290),'#18212b');frame.paste(Image.fromarray(env.render()),(0,40))
                draw=ImageDraw.Draw(frame)
                draw.text((10,5),f'{cfg.run_id} / {checkpoint} / {chosen} / case {case_seed} {family}',fill='white')
                draw.text((10,23),'BLUE attacker / RED target | deterministic | fixed-case replay',fill='white')
                draw.text((10,276),f'cuts={info["cut_count"]} block={info["blocking_time"]:.1f}s collision={info["collision"]} grade={info["grade_violation"]}',fill='white')
                frames.append(frame);a,v=env.attacker,env.target;gap=float(a.position[0]-v.position[0]-5)
                trace.append(dict(time=info['time'],attacker_x=float(a.position[0]),attacker_y=float(a.position[1]),
                    target_x=float(v.position[0]),target_y=float(v.position[1]),attacker_speed=float(a.speed),target_speed=float(v.speed),
                    reference_speed=float(env.reference.vehicles[1].speed),thw=gap/max(float(v.speed),2) if gap>0 else 100.))
                if term or trunc:break
    finally:env.close()
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=round(1000/cfg.policy_frequency),loop=0)
    output.with_suffix('.json').write_text(json.dumps(dict(case_seed=case_seed,family=family,team=chosen,success=info['success'],trace=trace),indent=2),encoding='utf-8')
    plot_trace(trace,output.with_suffix('.png'),cfg.run_id)
    return dict(gif=str(output),success=info['success'],collision=info['collision'])
