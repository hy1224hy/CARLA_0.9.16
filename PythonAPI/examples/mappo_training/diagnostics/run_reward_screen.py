"""Serial reward-screen adapter holding the shared CARLA endpoint lease."""
import argparse
from utils.mut_mappo_session import CarlaOwner
from evaluation.evaluate_reward_fork import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[62001])
    args = parser.parse_args()
    with CarlaOwner(run_dir=args.output, purpose='frozen base reward behavior screening'):
        evaluate(args.checkpoints, args.output, args.seeds, 'deterministic', False)


if __name__ == '__main__':
    main()
