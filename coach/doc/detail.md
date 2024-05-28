# Coach Detail

## Training Pipeline

`tools/train_net.py` 是训练的入口。`__main__` 解析了来自 [engine/defaults.py](#enginedefaultspy) 的参数，设置了分布式节点的通信地址。随后由来自 [engine/launch](#enginelaunchpy) `launch` 函数进入了 `main` 函数。`setup_cfg` 函数读取了默认配置，解析了配置文件并且与命令行参数合并，进行来自 [engine/defaults.py](#enginedefaultspy) 的基本的初始化设定。

创建了来自 [engine/defaults.py](#enginedefaultspy) 的 `DefaultTrainer`。该 Trainer 继承自 [engine/base_trainer.py](#enginebase_trainerpy) 的 `TrainerBase`。

## `checkpoint`

### `checkpoint/checkpointer.py`

类 `Checkpointer`。检查点管理器。负责保存和加载本地的模型、优化器、学习率调度器、训练状态。会告知模型参数的不匹配情况。

## `config`

### `config/config.py`

函数 `configurable`。作为一个装饰器，将函数或者 `__init__` 方法装饰为可配置的函数。装饰后的函数可以接受一个 `CfgNode` 类型的参数 `cfg`，并且可以通过 `cfg` 获取配置参数。部分参数会被转发到配置函数中，而不是直接传递给被装饰的函数。具体使用方法见其注释。

### `config/defaults.py`

包含并且定义了所有默认的配置参数。

## `data`

### `data/build.py`

## `engine`

### `engine/base_trainer.py`

类 `TrainerBase`。训练器的基类。包含了钩子，部分的训练状态。

类 `TrainerBase` 函数 `train`。训练的主要流程。

类 `TrainerBase` 函数 `before_train`、`after_train`、`before_step`、`after_step`、`after_backward`。使用钩子函数，对训练过程进行监控。

类 `TrainerBase` 函数 `is_loop_completed`。判断训练是否结束。

类 `TrainerBase` 函数 `state_dict`、`load_state_dict`。保存和加载训练状态。

### `engine/defaults.py`

函数 `default_argument_parser`。解析命令行参数，返回可编辑的 `argparse.ArgumentParser`。默认的参数有：
- config-file: 配置文件路径
- resume: 是否从上次训练的断点继续训练
- eval-only: 是否只进行评估
- num-gpus: 每个节点的 GPU 数量
- num-machines: 节点总数量
- machine-rank: 当前节点的编号
- dist-url: 训练时的分布式节点的通信地址
- opts: 用 Key-Value 的形式修改配置文件中的参数

函数 `default_setup`。基本的初始化设定。包括：
- 设置日志文件，由主进程创建输出文件夹，和其他进程一起将日志输出到该文件夹
- 收集进程信息、环境信息、命令行参数信息
- 设置随机数种子

函数 `default_writers`。创建输入日志文件夹，创建文本形式、JSON 形式、TensorBoard 形式的日志文件。

函数 `create_ddp_model`。创建分布式模型。如果只有一个节点，则直接返回单机模型。如果有多个节点，则创建分布式模型，使用 `DistributedDataParallel` 封装模型。

类 `DefaultTrainer`。设置日志文件，创建模型、优化器、数据加载器、学习率调度器、检查点管理器。

### `engine/hooks.py`

类 `IterationTimer`。迭代计时器。在训练过程中，记录每次迭代的时间。

类 `PeriodicWriter`。周期性的指标打印器。在训练过程中，每隔一段时间打印一次指标。

类 `PeriodicCheckpointer`。周期性的检查点管理器。在训练过程中，每隔一段时间保存一次检查点。

### `engine/launch.py`

函数 `launch`。分布式训练的入口。如果进程数为 1，则直接调用 `main` 函数。如果机器数为 1，则自动设置端口。对于多机器多进程，在每个机器上启动和 `num_gpu` 个新的进程，并且调用 `_distributed_worker` 函数。

函数 `_distributed_worker`。初始化多机通信，组成局部通信，同步所有进程以防止某个进程初始化失败。随后，每个进程调用 `main` 函数。

## `modeling`

#### `modeling/architecture/build.py`

函数 `build_model`。工厂模式，注册所有模型。根据配置文件中的 `MODEL.NAME` 参数，返回对应的构建好的模型。

## `Solver`

### `Solver/build.py`

函数 `build_optimizer`。构建优化器。

函数 `build_lr_scheduler`。构建学习率调度器。

### `Solver/scheduler.py`

类 `WarmupParamScheduler`。预热学习率调度器。在预热阶段，学习率从很小的值线性增长到基础学习率。

类 `LRMultiplier`。每一步，学习率都会乘以调度器的输出。

## `utils`

### `utils/collect_env.py`

函数 `collect_env_info`。收集尽可能多的环境信息，包括但不限于：
- 系统信息
- Python, CUDA, PyTorch, TorchVision, numpy, pillow, fvcore 等的版本
- CUDA, GPU 的信息

### `utils/comm.py`

函数 `get_world_size`。返回分布式节点的总数量。

函数 `get_rank`。返回当前进程的编号。

函数 `is_main_process`。判断当前进程是否为主进程。

函数 `create_local_process_group`。把同一机器的进程编为一个通信组，用于局部通信。

函数 `get_local_process_group`。获得局部通信组。

函数 `get_local_size`。获得局部通信组的大小。

函数 `get_local_rank`。获得当前进程在局部通信组的编号。

函数 `synchronize`。阻塞并同步所有进程。

函数 `all_gather`。收集所有进程的数据，并广播给所有进程。

函数 `gather`。收集所有进程的数据，并广播给指定进程。

### `utils/env.py`

函数 `seed_all_rng`。设置随机数种子。如果给定的种子为 `None`，则使用当前时间戳和进程编号作为种子。

函数 `setup_environment`。收集自定义的环境变量，导入其中的自定义模块。

### `utils/events.py`

类 `EventStorage`。事件存储器。用于记录训练过程中的标量值和生成的图像。

类 `EventStorage`，函数 `latest_with_smoothing_hint`。获取最后一个事件的值。如果指定了平滑方法，返回滑动窗口的中位数。

类 `CommonMetricPrinter`。通用的指标打印器。在训练过程中，每隔一段时间打印一次指标。

类 `JSONWriter`。JSON 格式的日志文件。在训练过程中，每隔一段时间将指标写入日志文件。

类 `TensorboardXWriter`。TensorBoard 格式的日志文件。在训练过程中，每隔一段时间将指标写入日志文件。

### `utils/file_io`

对象 `PathManagerSingleton`。单例模式的文件管理器。负责项目内部的所有文件的读写。

类 `CoachHandler`。负责项目远端资源的获取。

### `utils/logger.py`

类 `_ColoredFormatter`。继承自 `logging.Formatter`。用于设置日志输出的颜色。

函数 `setup_logger`。创建项目的日志文件夹，设置日志文件的格式和输出方式。对于主进程，将日志输出到 `stdout`，并且设置日志的颜色；对于其他进程，将日志输出到日志文件夹。

函数 `_cached_log_stream`。cache 了日志文件的输出流，当不同进程都需要输出日志时，只需要使用同一个输出流。

函数 `log_api_usage`。记录 API 的使用情况。
