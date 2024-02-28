import argparse
import multiprocessing as mp

from coach.config import CfgNode, get_cfg
from coach.utils import setup_logger
from .neural_renderer import NeuralRenderer

def setup_cfg(args: argparse.Namespace) -> CfgNode:
    """
    Create configs from default settings, file, and command-line arguments.
    """
    cfg = get_cfg()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coach demo for builtin configurations")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=[], nargs=argparse.REMAINDER
    )

def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="Coach")
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = Demo(cfg)

if __name__ == "__main__":
    main()
