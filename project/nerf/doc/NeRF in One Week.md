# NeRF in One Week (Python)

我做 NeRF in One Week 的初衷是想做一个属于自己和实验室的*模块化 NeRF*。为了最好的性能，模块化 NeRF 必须要用 C++/CUDA 编程。不过在动手写 CUDA 之前，我想到可以用之前的 Python/Jupyter 版的 NeRF 作为 CUDA 版的模板。为了更好地理解 NeRF，以及给实验室新人一个比较友善的学习体验，我把我对于 NeRF 代码的理解写成一个教程以供参考。

对于看 NeRF in One Week 的你，这里我们要求你：
1. 有 Python 基础，至少对于接下来出现的代码不会感到疑惑。
2. 有 Pytorch 基础，至少看得懂官网教程的代码。
3. 看过并且理解 NeRF 论文，能把论文和代码对应起来。

## 1 前瞻

NeRF 本质是 Neural Volume Rendering。对于 Volume Rendering，我们通常用光追的方法渲染，这是一个简单的[光追教程](https://raytracing.github.io/)。对于 NeRF，它可以分为图形学部分和神经网络部分。图形学部分主要包括相机模型，成像模型；而神经网络部分主要包括场景的表达。因此我们可以把[光追教程](https://raytracing.github.io/)中对于场景部分的代码替换为神经网络即可。不过这样还是过于凌乱，在稍加整理之后，我们可以得到以下的代码结构：
![]()

我们会根据以上的代码结构图对于 NeRF in One Week 的代码进行讲解。

## 2 输入输出模块

这是整个工程中最简单的部分，且对于 NeRF 的理解没有帮助，可以选择跳过。写这部分的目的只是为以后的 CUDA 版本明确所需的小模块。

### 2.1 BaseIO

所有输入输出类的基类，规定了输入输出类的

1. 从外部输入的数据类型。
2. 从外部输入的函数。
3. 输出到外部的数据类型。
4. 输出到外部的函数。

### 2.2 输入输出类

现有的输入输出类包括了：

1. `ImageIO`
2. `JsonIO`
3. `VideoIO`
4. `VolumeIO`

## 3 数据集模块

数据集存储的主要是送入网络的输入，而送入网络的输入是光线上的采样点，即三维坐标 $\mathbf{x}$ 和二维方向 $\mathbf{d}$ （实际上是归一化的三维向量）。由于 NeRF 也算一种 Image-based Rendering，而且我们需要通过已知的相机参数和图片信息生成光线，因此相机类和图片数据集类是必不可少的两个小模块。

### 3.1 相机

图形学中的相机有两种，分别是透视（perspective）投影相机和正交（orthographic）投影相机。

## 4 模型模块

### 4.1 编码模块

编码器的作用是将低维的坐标输入映射到高维编码空间。由于神经网络倾向于学习低频的信息，编码操作使得网络能够区分空间上相邻的两个点。

#### 4.1.1 Positional Encoder

Positional Encoding 是用一系列的 $\sin$ 和 $\cos$ 函数来编码低维输入，可以用来编码位置向量和方向向量。具体的数学表达形式为：
$$
(x,y,z)\mapsto(\\
\sin(2^0x),\sin(2^1x),\dots,\sin(2^Lx),\\
\cos(2^0x),\cos(2^1x),\dots,\cos(2^Lx),\\
\sin(2^0y),\dots,\cos(2^0y),\dots,\\
\sin(2^0z),\dots,\cos(2^0z),\dots\\
)
$$
其中 $(x,y,z)$ 既可以表示坐标，也可以表示方向。在 NeRF 原文中，对于坐标的编码频率为 $L=10$，对于方向的编码频率为 $L=4$。

#### 4.1.2 Spherical Harmonics Encoder

Spherical Harmonics Encoder 是用一系列的球谐函数来编码 3 维输入向量，且只用来编码方向向量。具体的数学表达形式为：
$$
(x,y,z)\mapsto(Y_0^0(x,y,z),Y_1^{-1}(x,y,z),\dots,Y_n^{n}(x,y,z))
$$
$n$ 阶球谐一共有有 $n^2$ 个球谐函数。具体的球谐函数表可以参考 [球谐函数表](https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Spherical_harmonics)。这里大概提一下如何计算球谐函数的归一化系数。我们这里将球谐函数的 $\phi$ 分量用三角函数表示，即：
$$
Y^m_l(\theta,\phi)=P^m_l(\cos\theta)\cdot\begin{cases}\cos{m\phi}&m>0\\1&m=0\\\sin{m\phi}&m<0\end{cases}
$$
因此归一化系数可以通过求 $Y^m_l(\theta,\phi)$ 的二范数模长得到，这里假设 $m>0$，即：
$$
\begin{align*}
|Y^m_l(\theta,\phi)|&=\int^{\pi/2}_{-\pi/2}\int^{2\pi}_0[Y^m_l(\theta,\phi)]^2\mathrm{d}\theta\mathrm{d}\phi\\
&=\int^{\pi/2}_{-\pi/2}\int^{2\pi}_0[P^m_l(\cos\theta)\cos{m\phi}]^2\sin\theta\mathrm{d}\theta\mathrm{d}\phi\\
&=\int^{\pi/2}_{-\pi/2}[P^m_l(\cos\theta)]^2\sin\theta\mathrm{d}\theta\int^{2\pi}_0[\cos{m\phi}]^2\mathrm{d}\phi\\
\end{align*}
$$
然后将 $Y^m_l(\theta,\phi)$ 化为直角坐标系下的表示 $Y^m_l(x,y,z)$，并且除以归一化系数，就得到了代码中用到的编码方式。

### 4.2 网络模块

#### 4.2.1 Network

Network 严格遵循了 NeRF 的网络设置。考虑到不同的网络的配置文件会大不相同，因此没有实现一个全局的 `make_networks` 函数，而是一个类附带一个 `make_networks` 函数。

#### 4.2.2 Setting

设置部分包含了网络训练所需的其他组分，包括：损失函数，激活函数，优化器和度量。这些组分都可以写在配置文件中，用 `make_` 开头的函数就可以解析配置文件，得到所需的组分。各组分可以任意拓展。

**Losses**: 当前支持的 Loss 类型有：`SmoothL1Loss`, `MSELoss`, `HuberLoss`.

**Activations**: 当前支持的 Activations 类型有：`None`(`Identity`), `Exponential`， `ReLU`， `Sigmoid`， `Softplus`.

**Optimizers**: 当前支持的 Optimizers 类型有：`Adam`, `SGD`.

**Metrics**: 当前支持的 Metrics 类型有：`PSNR`.

#### 4.2.3 Rendering

Rendering 在深度学习中的本质是 testing。

