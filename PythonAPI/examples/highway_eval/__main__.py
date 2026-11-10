import argparse
import json
from .evaluate import evaluate_suite


def main():
    parser=argparse.ArgumentParser(description='Independent frozen-model Highway behavioral evaluation')
    parser.add_argument('--runs',required=True,help='Training root, one run directory, or one checkpoint file')
    parser.add_argument('--output',required=True,help='New output directory; never overwrite')
    parser.add_argument('--split',choices=('development','test'),default='development')
    parser.add_argument('--cases',type=int,default=100,help='Even case count, balanced follow/adjacent')
    parser.add_argument('--checkpoint',choices=('selected','latest'),default='selected')
    parser.add_argument('--team',choices=('A','B'),default='A',help='Replica for latest.pt; selected.pt records its own replica')
    args=parser.parse_args()
    result=evaluate_suite(args.runs,args.output,args.split,args.cases,args.checkpoint,args.team)
    print(json.dumps([dict(run=r['run'],team=r['team'],steps=r['checkpoint_train_steps'],
                          overall=r['overall']) for r in result],indent=2))


if __name__=='__main__':main()
