# NeRF PyTorch

## `dataset`



## `modeling`

### `modeling/architecture/nerf.py`

类 `NeRF`。主要功能为重建的 NeRF 模型。包含了四个子模块：
1. 从输入的图像和相机参数中采样 3D 点的模块。
2. 将 3D 点映射成特征的模块。
3. 将特征映射成颜色等属性的模块。
4. 损失计算模块。

### `modeling/backbone/vanilla.py`

类 `VanillaNeRF`。**TODO：完成基本训练骨架后实现。**
