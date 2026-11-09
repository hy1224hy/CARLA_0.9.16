"""Spectator-only companion: never ticks, resets, spawns, or controls vehicles."""
import argparse
import json
import time
from pathlib import Path


def main():
    import carla
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status-file',type=Path,required=True)
    args=parser.parse_args()
    client=carla.Client('127.0.0.1',2000)
    client.set_timeout(5.)
    print('TOPDOWN_OBSERVER_STARTED; WORLD_TICKS_ADDED=0',flush=True)
    while True:
        status=json.loads(args.status_file.read_text(encoding='utf-8'))
        if status.get('status') in ('COMPLETED','FAILED','STOPPED'):
            break
        try:
            world=client.get_world()
            vehicles=list(world.get_actors().filter('vehicle.*'))
            locations=[v.get_location() for v in vehicles if v.is_alive]
            if locations:
                xs=[p.x for p in locations]; ys=[p.y for p in locations]
                span=max(max(xs)-min(xs),max(ys)-min(ys),20.)
                world.get_spectator().set_transform(carla.Transform(
                    carla.Location(x=(min(xs)+max(xs))/2,y=(min(ys)+max(ys))/2,
                                   z=min(90.,max(30.,span*1.2))),
                    carla.Rotation(pitch=-90.,yaw=0.,roll=0.)))
        except RuntimeError:
            pass  # Actor cleanup/map transitions are expected for this observer.
        time.sleep(.2)
    print('TOPDOWN_OBSERVER_FINISHED; WORLD_TICKS_ADDED=0',flush=True)


if __name__=='__main__': main()
