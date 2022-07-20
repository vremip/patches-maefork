import argparse
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

import main_pretrain as trainer


def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=torch.cuda.device_count(), type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    return parser.parse_args()


def main(rank, args):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank % torch.cuda.device_count())
    trainer.main(args)


if __name__ == '__main__':

    args = parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    world_size = args.ngpus * args.nodes
    os.environ['WORLD_SIZE'] = str(world_size)

    mp.spawn(
        main,
        args=(args,),
        nprocs=world_size
    )