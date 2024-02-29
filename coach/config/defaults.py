from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# Benchmark different cudnn algorithms.
_C.CUDNN_BENCHMARK = False

# A negative value means using random seed depends on the system.
# A positive value means using the given value as seed.
_C.SEED = -1

# The version number of the config.
_C.VERSION = 1

_C.OUTPUT_DIR = ""

# -----------------------------------------------------------------------------
# Dataloader
# -----------------------------------------------------------------------------

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.IMAGE_BATCH_SIZE = 1
_C.DATALOADER.RAY_BATCH_SIZE = 4096

_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

_C.DATASETS = CN()
_C.DATASETS.TRAIN = []

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NAME = ""
_C.MODEL.WEIGHTS = ""

_C.MODEL.SAMPLER = CN()
_C.MODEL.SAMPLER.NAME = ""
_C.MODEL.SAMPLER.PROJECTION = "perspective"

_C.MODEL.SCENE = CN()
_C.MODEL.SCENE.NAME = ""
_C.MODEL.SCENE.MLP_WIDTH = 256
_C.MODEL.SCENE.MLP_DEPTH = 8
_C.MODEL.SCENE.SKIP_CONNECTION = False

_C.MODEL.SCENE.POSITIONAL_ENCODER = CN()
_C.MODEL.SCENE.POSITIONAL_ENCODER.NAME = ""
_C.MODEL.SCENE.POSITIONAL_ENCODER.N_FREQUENCIES = 10

_C.MODEL.SCENE.DIRECTIONAL_ENCODER = CN()
_C.MODEL.SCENE.DIRECTIONAL_ENCODER.NAME = ""
_C.MODEL.SCENE.DIRECTIONAL_ENCODER.N_FREQUENCIES = 4

_C.MODEL.RENDERER = CN()
_C.MODEL.RENDERER.NAME = ""

_C.MODEL.CRITERION = CN()
_C.MODEL.CRITERION.NAME = ""
_C.MODEL.CRITERION.TYPE = ""

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

_C.SOLVER = CN()

_C.SOLVER.AMP = CN()
_C.SOLVER.AMP.ENABLED = False

_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.BASE_LR_END = 0.5

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.WEIGHT_DECAY_NORM = None

_C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = False

_C.SOLVER.LR_SCHEDULER_NAME = ""
_C.SOLVER.STEPS = ()
_C.SOLVER.MAX_ITER = 0
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.RESCALE_INTERVAL = False
_C.SOLVER.WARMUP_FACTOR = 1e-3
_C.SOLVER.WARMUP_ITERS = 0
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.LOG_PERIOD = 20
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

_C.TEST = CN()
_C.TEST.EVAL_PERIOD = 5000
