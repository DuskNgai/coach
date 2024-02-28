# Coach

训练器相关的代码来自于 [detectron2](https://github.com/facebookresearch/detectron2.git)。NeRF 相关的代码来自于各种 NeRF 项目。

## 环境配置

```bash
conda create -n coach python=3.11
conda activate coach
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install ipykernel
pip install fvcore omegaconf tensorboard
```

## 使用方法

```bash
python tools/train_net.py --config-file nerf/config/base.yaml
```

在环境中可以设定环境变量。可选的有

```bash
export COACH_ENV_MODULE = <PATH_TO_MODULE> # 指定额外导入的模块
export COACH_DISABLE_OPENCV = False # 禁用 OpenCV
```
