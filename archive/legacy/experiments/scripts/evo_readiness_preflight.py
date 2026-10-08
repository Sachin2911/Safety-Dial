#!/usr/bin/env python3
"""Read-only structural/budget/launch review; never creates a run or issues queries."""
import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from helpers.evoReadinessProtocol import launch_blockers,protocol_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/evo/stage5_readiness_draft.yaml')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config=yaml.safe_load(args.protocol.read_text())
    plan=protocol_plan(config)
    blockers=launch_blockers(config,ROOT)
    result=dict(protocol=str(args.protocol),structurally_valid=True,execution_ready=not blockers,
        launch_blockers=blockers,counts=plan['counts'],candidate_batch_shapes=plan['batch_shapes'],
        experiment_run=False,model_queries=0,real_simulator_steps=0)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
