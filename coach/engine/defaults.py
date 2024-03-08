import argparse
import logging
import os
import sys
from typing import Any
import weakref

from omegaconf import OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.optim
import torch.utils.data as torch_data

from coach.config import CfgNode
from coach.checkpoint import CoachCheckpointer
from coach.data import build_coach_train_loader, build_coach_test_loader
from coach.modeling import build_model
from coach.solver import build_optimizer, build_scheduler
from coach.utils import comm
from coach.utils.collect_env import collect_env_info
from coach.utils.env import seed_all_rng
from coach.utils.events import EventWriter, CommonMetricPrinter, JSONWriter, TensorboardXWriter
from coach.utils.file_io import PathManagerSingleton
from coach.utils.logger import setup_logger

from .base_trainer import TrainerBase
from .amp_trainer import AMPTrainer
from .simple_trainer import SimpleTrainer

__all__ = [
    "create_ddp_model",
    "default_argument_parser",
    "default_setup",
    "default_writers",
    "DefaultTrainer",
]

def _try_get_key(cfg: CfgNode, *keys, default=None) -> Any:
    """
    Try to get the value of the key from the config. Otherwise, return the default value.
    
    Args:
        `cfg` (CfgNode): the config.
        `*keys` (str): the keys to try.
        `default` (Any): the default value to return if the key is not found.

    Returns:
        (Any): the value of the key if found, otherwise the default value.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    not_found = object()
    for key in keys:
        value = OmegaConf.select(cfg, key, default=not_found)
        if value is not not_found:
            return value
    return default

def _highlight(code: str, file_name: str) -> str:
    """
    Add syntax highlighting to the code according to the file name.

    Args:
        `code` (str): the code to highlight.
        `file_name` (str): the file name.

    Returns:
        (str): the highlighted code.
    """
    try:
        import pygments
    except ImportError:
        return code
    
    from pygments.formatters import Terminal256Formatter
    from pygments.lexers import PythonLexer, YamlLexer

    lexer = PythonLexer() if file_name.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code

def default_argument_parser() -> argparse.ArgumentParser:
    """
    A default argument parser for Coach, which supports users to add customized arguments if necessary.
    The default arguments are as follows:
    - `--config-file`: the path to the config file.
    - `--resume`: whether to resume from the output directory if exists.
    - `--eval-only`: perform evaluation only.
    - `--num-gpus`: number of gpus *per machine*.
    - `--num-machines`: total number of machines.
    - `--machine-rank`: the rank of this machine (unique per machine).
    - `--dist-url`: initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details.
    """
    parser = argparse.ArgumentParser(
        epilog="""
Examples:

Run on single machine:
    $ {0} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {0} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {0} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {0} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""".format(sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true", help="whether to resume from the output directory if exists")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", default=1, type=int, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", default=1, type=int, help="total number of machines")
    parser.add_argument("--machine-rank", default=0, type=int, help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 3 * 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See https://pytorch.org/docs/stable/distributed.html for details."
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None, help="Modify config options at the end of the command. For Yacs configs, use space-separated `PATH.KEY VALUE` pairs.")

    return parser

def default_setup(cfg: CfgNode, args: argparse.Namespace) -> None:
    """
    Perform some basic common setups at the beginning of a job, including:
    - Setting up logger
    - Setting up the random number generator
    - Backing up the config to output directory

    Args:
        `cfg` (CfgNode): the config.
        `args` (argparse.Namespace): the command line arguments.
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "train.output_dir")
    if comm.is_main_process() and output_dir is not None:
        PathManagerSingleton.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))

    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManagerSingleton.open(args.config_file, "r").read(), args.config_file)
            )
        )

    if comm.is_main_process() and output_dir is not None:
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with PathManagerSingleton.open(path, "w") as f:
                f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # Make sure that the seed is different for different processes.
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False)

def default_writers(output_dir: str, max_iter: int = None) -> list[EventWriter]:
    """
    A list of `EventWriter` to be used. By default it contains
    - `CommonMetricPrinter`
    - `JSONWriter`
    - `TensorboardXWriter`
    """
    PathManagerSingleton.mkdirs(output_dir)
    return [
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]

def create_ddp_model(model: torch.nn.Module, *, fp16_compression: bool = False, **kwargs) -> torch.nn.Module:
    """
    Create a DDP model if there are multiple processes.
    
    Args:
        `model` (torch.nn.Module): the model to be wrapped.
        `fp16_compression` (bool): whether to use fp16 compression.

    Returns:
        (torch.nn.Module): the DDP model.
    """
    if comm.get_world_size() == 1:
        return model

    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)

    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hook
        # It casts GradBucket tensor to `torch.float16` format and then divides it by the process group size.
        ddp.register_comm_hook(state=None, hook=comm_hook.fp16_compress_hook)
    return ddp

class DefaultTrainer(TrainerBase):
    """
    The default trainer for Coach, which includes all the necessary components for training.
    Also, it provides some useful methods for building the model, optimizer, scheduler, etc.
    The trainer is built upon `SimpleTrainer` or `AMPTrainer` according to the config.

    Args:
        `cfg` (CfgNode): the config.

    Attributes:
        `_trainer`
        `scheduler`
        `checkpointer`
    """

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__()
        logger = logging.getLogger("Coach")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        # Usually, the model is constructed before the optimizer.
        model = DefaultTrainer.build_model(cfg)
        optimizer = DefaultTrainer.build_optimizer(cfg, model)
        data_loader = DefaultTrainer.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = DefaultTrainer.build_scheduler(cfg, optimizer)
        self.checkpointer = CoachCheckpointer(
            self._trainer.model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self)
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

    def resume_or_load(self, resume: bool = True) -> None:
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished,
            # thus we start from it + 1
            self.start_iter = self.iteration + 1

    def train(self) -> None:
        super().train(self.start_iter, self.max_iter)

    def test(self) -> None:
        raise NotImplementedError()

    def step(self):
        self._trainer.iteration = self.iteration
        self._trainer.step()

    def state_dict(self) -> dict:
        result = super().state_dict()
        result["_trainer"] = self._trainer.state_dict()
        return result

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])

    def build_writers(self) -> list[EventWriter]:
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    @classmethod
    def build_model(cls, cfg: CfgNode) -> torch.nn.Module:
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
        return build_optimizer(cfg, model)

    @classmethod
    def build_scheduler(cls, cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return build_scheduler(cfg, optimizer)
    
    @classmethod
    def build_train_loader(cls, cfg: CfgNode) -> torch_data.DataLoader:
        return build_coach_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode) -> torch_data.DataLoader:
        return build_coach_test_loader(cfg)
