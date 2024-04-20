import argparse
import random
import os
from pathlib import Path
import sys

sys.path.append(Path.cwd().as_posix())

from coach.checkpoint import CoachCheckpointer
from coach.config import CfgNode
from coach.engine import DefaultTrainer, default_argument_parser, default_setup, launch

# Just put it here to make the import order correct.
import project

def setup_cfg(args: argparse.Namespace) -> CfgNode:
    """
    Create configs from default settings, file, and command-line arguments.
    """
    cfg = CfgNode(CfgNode.load_yaml_with_base(args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args: argparse.ArgumentParser):
    cfg = setup_cfg(args)

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        CoachCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        result = DefaultTrainer.test(cfg, model)

        return result

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()

    # Set the url of the distributed backend to the node
    if args.num_machines == 1:
        args.dist_url = "tcp://127.0.0.1:{}".format(random.randint(11111, 60000))
    else:
        if args.dist_url == "host":
            args.dist_url = "tcp://{}:12345".format(os.environ["SLURM_JOB_NODELIST"])
        elif not args.dist_url.startswith("tcp"):
            tmp = os.popen(
                "echo $(scontrol show job {} | grep BatchHost)".format(
                    args.dist_url
                )
            ).read()
            tmp = tmp[tmp.find("=") + 1: -1]
            args.dist_url = "tcp://{}:12345".format(tmp)
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
