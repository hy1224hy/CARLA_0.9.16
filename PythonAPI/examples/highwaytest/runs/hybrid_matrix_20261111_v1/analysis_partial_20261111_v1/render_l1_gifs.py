"""Render frozen L1 audit checkpoints on matched development scenarios."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from highway_hybrid.config import HybridConfig
from highway_hybrid.environment import HybridEnv
from highway_hybrid.model import HybridActorCritic
from highway_hybrid.runner import load_checkpoint, source_digest


def main():
    root = Path(__file__).resolve().parent
    output = root / 'l1_gifs'
    output.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    results = []
    for family, seed in [('follow', 4000000), ('adjacent', 4000001)]:
        videos = []
        for method in ['independent', 'mutual']:
            name = f'L1_{method}_s44'
            state = load_checkpoint(root / 'development_audit' / name / 'checkpoint.pt')
            assert state['source_sha256'] == source_digest(), 'Frozen source mismatch'
            cfg = HybridConfig.from_dict(state['config'])
            model = HybridActorCritic(cfg.hidden, cfg.state_dim)
            model.load_state_dict(state['model'])
            model.eval()
            env = HybridEnv(cfg, render_mode='rgb_array')
            frames = []
            try:
                obs, _ = env.reset(seed=seed, options={'family': family})
                with torch.no_grad():
                    while True:
                        acceleration, lane = model.deterministic(torch.as_tensor(obs['state']), torch.as_tensor(obs['lane_mask']))
                        obs, reward, term, trunc, info = env.step({'acceleration': acceleration.numpy(), 'lane': int(lane)})
                        frame = Image.new('RGB', (900, 300), '#18212b')
                        frame.paste(Image.fromarray(env.render()), (0, 40))
                        draw = ImageDraw.Draw(frame)
                        draw.text((12, 7), f'{name} | {family} seed={seed} | deterministic selected.pt', fill='white')
                        draw.text((12, 23), 'BLUE = learned attacker     RED = IDM target     speeds: m/s', fill='white')
                        draw.text((12, 281), f'blocking={info["blocking_time"]:.1f}s  collision={info["collision"]}  grade_violation={info["grade_violation"]}', fill='white')
                        frames.append(frame)
                        if term or trunc:
                            break
                results.append({'run': name, 'family': family, 'seed': seed, 'success': info['success'], 'collision': info['collision'], 'grade_violation': info['grade_violation'], 'blocking_time': info['blocking_time'], 'frames': len(frames)})
            finally:
                env.close()
            videos.append(frames)
        combined = []
        for i in range(max(map(len, videos))):
            frame = Image.new('RGB', (900, 600))
            for row, video in enumerate(videos):
                frame.paste(video[min(i, len(video)-1)], (0, row*300))
            combined.append(frame)
        path = output / f'L1_s44_{family}_comparison.gif'
        combined[0].save(path, save_all=True, append_images=combined[1:], duration=200, loop=0)
        print(path, flush=True)
    (output / 'manifest.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
