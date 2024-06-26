# Coach

训练器相关的代码来自于 [detectron2](https://github.com/facebookresearch/detectron2.git)。

## 环境配置

```bash
conda create -n coach python=3.11
conda activate coach
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ipykernel matplotlib tensorboard
pip install fvcore omegaconf
```

## 使用方法

```bash
# NeRF with blender dataset
python tools/train.py --config-file project/nerf/config/blender.yaml
# VAE with MNIST dataset
python tools/train.py --config-file project/vae/config/mnist.yaml
```

在环境中可以设定环境变量。可选的有

```bash
export COACH_ENV_MODULE = <PATH_TO_MODULE> # 指定额外导入的模块
export COACH_DISABLE_OPENCV = False # 禁用 OpenCV
```
