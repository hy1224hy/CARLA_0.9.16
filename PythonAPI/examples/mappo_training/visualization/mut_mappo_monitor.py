"""Runtime-only MutMAPPO monitoring; never owns a simulation tick or policy RNG."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


DEFAULTS = {"mode": "off", "host": "127.0.0.1", "port": 8080,
            "width": 960, "fps": 5.0, "overhead_height": 35.0, "smoothing_seconds": .5}


def add_monitor_arguments(parser):
    for key, value in DEFAULTS.items():
        kwargs = {"default": value, "type": type(value)}
        if key == "mode":
            kwargs["choices"] = ("off", "web")
        parser.add_argument("--monitor-" + key.replace("_", "-"), **kwargs)


def options_from_args(args):
    return validate_options({key: getattr(args, "monitor_"+key, default) for key, default in DEFAULTS.items()})


def validate_options(values):
    if set(values) - set(DEFAULTS):
        raise ValueError("unknown MutMAPPO monitor option")
    result = dict(DEFAULTS, **values)
    if (result["mode"] not in {"off", "web"} or type(result["port"]) is not int
            or not 1 <= result["port"] <= 65535 or not isinstance(result["host"], str)
            or not result["host"] or type(result["width"]) is not int or result["width"] < 160):
        raise ValueError("invalid MutMAPPO monitor configuration")
    for key in ("fps", "overhead_height", "smoothing_seconds"):
        value = result[key]
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or (key != "smoothing_seconds" and value == 0)):
            raise ValueError(f"invalid MutMAPPO monitor {key}")
    return result


def monitor_cli_arguments(args):
    options = options_from_args(args)
    if options["mode"] == "off":
        return []
    return [part for key, value in options.items()
            for part in ("--monitor-"+key.replace("_", "-"), str(value))]


class MutMAPPOMonitor:
    def __init__(self, config, directory, *, purpose="training", planned_rounds=None):
        self.options = validate_options(config.get("_monitor_options", {}))
        self.directory = Path(directory).resolve()
        self.purpose, self.web = purpose, None
        self.fields = {"monitor_kind": "mut_mappo", "purpose": purpose,
                       "run_id": self.directory.name, "method": config.get("method", "off"),
                       "round": None, "team": None, "condition": None,
                       "operation": "train" if purpose == "training" else "evaluation",
                       "planned_updates": planned_rounds,
                       "red_source_update": config.get("_red_payload", {}).get("update"),
                       "roles": {"0": "冻结红方", "1": "冻结红方", "2": "学习蓝方", "3": "学习蓝方"}}
        self.episodes, self._warning = 0, False

    def start(self):
        if self.options["mode"] == "off":
            return
        from visualization.web_training_monitor import WebTrainingMonitor
        root = Path(__file__).resolve().parents[1]
        self.web = WebTrainingMonitor(
            host=self.options["host"], port=self.options["port"], width=self.options["width"],
            fps=self.options["fps"], view="overhead", overhead_height=self.options["overhead_height"],
            smoothing_seconds=self.options["smoothing_seconds"], output_dir=str(self.directory/"monitor"),
            service_dir=str(root/"logs"/f"web_monitor_{self.options['port']}"))
        try:
            self.web.start()
            self.phase("starting", "正在初始化 MutMAPPO " + self.purpose)
            print(f"MutMAPPO 网页监控已启动: {self.web.url}", flush=True)
        except BaseException:
            self.web.close()
            self.web = None
            raise

    def call(self, callback, *args, **kwargs):
        if self.web is None:
            return
        try:
            return getattr(self.web, callback)(*args, **kwargs)
        except Exception as error:
            if not self._warning:
                print(f"[Mut monitor] {callback}: {error}; experiment continues", flush=True)
                self._warning = True

    def context(self, **fields):
        self.fields.update(fields)

    def phase(self, phase, message, **fields):
        self.context(**fields)
        self.call("set_phase", phase, **dict(self.fields, message=message))

    def begin(self, runtime, case):
        self.episodes += 1
        self.context(team=runtime.team, condition=runtime.condition,
                     scenario=case["case_key"], episode=self.episodes, step=0,
                     update=self.fields.get("round"))
        self.call("begin_episode", **dict(self.fields, message="正在切换四车场景，等待新画面"))

    def ready(self):
        operation = self.fields["operation"]
        phase = "validating" if operation == "gate" else "evaluating" if operation == "evaluation" else "rollout"
        message = "教师门控评估" if operation == "gate" else "独立配对评估" if operation == "evaluation" else "正在采样训练"
        self.phase(phase, message)

    def before_step(self, env):
        self.call("prepare_camera", env)

    def after_step(self, env, step, rewards, info):
        self.context(step=step, world_frame=getattr(env, "_last_world_frame", None),
                     episode_end_reason=info.get("episode_end_reason"))
        self.call("submit", {}, dict(self.fields, rewards={str(k):float(v) for k,v in rewards.items()}))

    def detach(self):
        # Always before the owning environment tears down sensors/world state.
        self.call("stop_camera")

    def finish(self, phase, *, result=None, message=""):
        if self.web is None:
            return
        saved = {"purpose": self.purpose, "output_directory": str(self.directory),
                 "qualification": "运行结束不代表驾驶或对抗效果合格"}
        if self.purpose == "training":
            saved.update(safe_round=None, checkpoint=None)
            try:
                manifest = json.loads((self.directory/"checkpoint_manifest.json").read_text(encoding="utf-8"))
                checkpoint = (self.directory/manifest["checkpoint"]).resolve()
                checkpoint.relative_to(self.directory)
                digest = hashlib.sha256()
                with checkpoint.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024*1024), b""):
                        digest.update(block)
                if digest.hexdigest() == manifest["sha256"]:
                    saved.update(safe_round=manifest["completed_round"], checkpoint=str(checkpoint))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        saved.update(result or {})
        try:
            self.call("finish", phase, result=saved, message=message)
        finally:
            self.call("close")
            self.web = None
