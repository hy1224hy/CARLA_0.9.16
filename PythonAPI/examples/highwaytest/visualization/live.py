"""Publish current training frames; never step/reset an environment or sample RNG."""
import json
import math
from pathlib import Path
import time

HTML = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Highway training preview</title><style>
body{background:#172029;color:#e6edf3;font:16px system-ui;margin:30px auto;max-width:1050px;padding:16px}
img{width:100%;border:1px solid #45505b;border-radius:10px}pre{white-space:pre-wrap;color:#aac8dc}
</style><h1>Highway · actual training rollout</h1>
<p>Red 0/1: fixed IDM · Blue 2/3: current sampled policy</p>
<img id="frame" alt="Waiting for the first training frame"><pre id="status">Loading…</pre>
<p>Frames pause during PPO updates and validation. No extra simulation is run by this page.</p>
<script>
async function refresh(){try{
const response=await fetch('status.json',{cache:'no-store'});
if(!response.ok)throw Error('Waiting for trainer');
const s=await response.json();
document.getElementById('status').textContent=JSON.stringify(s,null,2)+
'\\nFrame age: '+Math.max(0,Date.now()/1000-s.frame_time).toFixed(1)+'s';
if(s.frame_time!==window.lastFrame){document.getElementById('frame').src='frame.png?t='+s.frame_time;window.lastFrame=s.frame_time;}
}catch(e){document.getElementById('status').textContent=String(e);}
setTimeout(refresh,200);}refresh();
</script></html>"""


def publish(temp, target):
    # Windows readers/virus scanners can briefly hold an existing PNG. Preview
    # contention must not abort a scientific run; a later frame can replace it.
    for _ in range(5):
        try:
            temp.replace(target)
            return True
        except PermissionError:
            time.sleep(0.01)
    return False


class LivePreview:
    def __init__(self, output, *, speed=0.0):
        if not math.isfinite(speed) or speed < 0 or 0 < speed < 0.1:
            raise ValueError("preview speed must be 0 (unpaced) or >=0.1")
        self.directory = Path(output) / "preview"
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "index.html").write_text(HTML, encoding="utf-8")
        self.speed = speed
        self.last_frame = self.last_step = 0.0
        self.status = {}
        with (self.directory / "sessions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"started_at": time.time(), "speed": speed,
                                     "source": "actual training rollout, stochastic policy"}) + "\n")

    def context(self, team, round_id, train_steps):
        self.status.update(team=team, round=round_id, train_steps=train_steps, phase="sampling")
        self.counter = train_steps
        self.last_step = 0.0

    def _status(self):
        temp = self.directory / "status.tmp"
        temp.write_text(json.dumps(self.status, indent=2, allow_nan=False), encoding="utf-8")
        publish(temp, self.directory / "status.json")

    def phase(self, name):
        self.status["phase"] = name
        self._status()

    def step(self, env, info, action, reward):
        now = time.perf_counter()
        if self.speed and self.last_step:
            time.sleep(max(0, 1 / env.task.policy_frequency / self.speed - (now - self.last_step)))
        self.last_step = time.perf_counter()
        self.counter += 1
        if self.last_step - self.last_frame < (0.1 if self.speed else 0.5) and not env.ended:
            return
        from PIL import Image
        temp = self.directory / "frame.tmp"
        with Image.fromarray(env.render()) as frame:
            frame.save(temp, format="PNG")
        if not publish(temp, self.directory / "frame.png"):
            return
        self.last_frame = self.last_step
        self.status.update(frame_time=time.time(), phase="sampling", train_steps=self.counter,
                           episode_step=info["episode_steps"], family=info["family"],
                           collision=info["collision"], offroad=info["offroad"], success=info["success"],
                           blue_progress_m=info["blue_progress_m"], red_delay_m=info["red_delay_m"],
                           action=action.tolist(), step_reward=reward.tolist(), speed=self.speed)
        self._status()


def serve(output, port=8000):
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    directory = Path(output) / "preview"
    if not (directory / "index.html").exists():
        raise ValueError("start train --preview first, then serve the same run directory")

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(directory))) as server:
        print(f"Preview: http://127.0.0.1:{port} | {directory}", flush=True)
        server.serve_forever()
