"""Versioned behavior pilot; run from PythonAPI/examples."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--config',required=True);t.add_argument('--output',required=True)
    t.add_argument('--resume');t.add_argument('--max-rounds',type=int)
    e=sub.add_parser('evaluate');e.add_argument('--runs',required=True);e.add_argument('--output',required=True)
    e.add_argument('--split',choices=['development','test'],default='development')
    e.add_argument('--cases',type=int,default=40);e.add_argument('--checkpoint',choices=['selected','latest'],default='latest')
    e.add_argument('--team',choices=['A','B'],default='A')
    args=parser.parse_args()
    from .runner import train,evaluate_suite
    from .config import HybridConfig
    if args.command=='train':
        cfg=HybridConfig.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
        result=train(cfg,args.output,args.resume,args.max_rounds)
    else:result=evaluate_suite(args.runs,args.output,args.split,args.cases,args.checkpoint,args.team)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
