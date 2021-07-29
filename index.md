<center> <h1> CVPR2021 NAS Track1 第三名技术方案：如何提升超网络一致性？ </h1></center>

Author: 关超宇 from Meta_Learners, Tsinghua University

本库为CVPR 2021 Lightweight NAS Challenge Track1 第三名（并列）的解决方案。接下来，我们将详细讲解本方案中的技术部分。

## 问题分解
针对于one-shot神经网络通道数搜索问题，我们建立了一套统一的解决框架，其中包含四个关键部分（如图1）——**超网络构建**、**架构和数据采样**、**超网络参数优化**以及**子网络评估**。超网络构建部分负责使用参数共享机制来构建统一的超网络；数据和架构采样部分负责从搜索空间和训练样本空间中采样每一轮优化所需要的架构和数据；超网络参数优化负责利用采样出的架构和样本针对超网络进行参数更新；而子网络评估旨在利用训练好的超网络来评估空间中每个网络的效果。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/0ae11dfebe0544b587c0318bf8ba80bad5f58c4ca3144c29b94b7745231c5a07" width="600"/></center>
<center>图1 one-shot神经网络通道数搜索框架</center>

## 超网络构建
针对赛题所给的ResNet20搜索空间，每个架构包含三种不同的操作——$3\times 3$卷积操作（conv）、批量归一化操作（batch norm）以及全连接操作（fc）。对于通道数搜索这一任务，我们要考虑的是如何共享不同输入输出维度的相同操作。
对于卷积操作，针对输入输出维度的考虑，一共有四种可以考虑的参数共享方式：
- 完全独立（Independent）：对于同一层的不同输入输出维度的卷积操作，我们使用完全独立的参数。即每个卷积操作的拥有独立的参数$P_{in,out}\in \mathbb{R}^{d_{in}\times d_{out}\times 3\times 3}$
- 输入共享（Front）：对于拥有相同输入维度的卷积操作，我们使用同一组参数$P_{in}\in \mathbb{R}^{d_{in}\times d_{outmax}\times 3\times 3}$进行参数共享。每个卷积操作的参数为$P_{in,out}=P_{in} [:,:d_{out} ]\in \mathbb{R}^{d_{in}\times d_{out}\times 3\times 3}$
- 输出共享（End）：和输入共享的逻辑相同，只不过针对所有输出维度一致的卷积操作使用相同的参数进行共享：$P_{out}\in \mathbb{R}^{d_{inmax}\times d_{out}\times 3\times 3}$。每个卷积操作的参数为$P_{in,out}=P_{out} [:d_{in} ]\in \mathbb{R}^{d_{in}\times d_{out}\times 3\times 3}$.
- 全共享（Full）：同一层的所有卷积操作共享相同的一组参数$P\in\mathbb{R}^{d_{in}\times d_{out}\times 3\times 3}$。每个卷积操作的参数为$P_{in,out}=P[:d_{in},:d_{out} ]∈\mathbb{R}^{d_{in}\times d_{out}\times 3\times 3}$.

同样的，对于批量归一化操作，按照其前面所连接的卷积层输入输出维度来区分，仍然有四种不同的权重共享方式：完全独立、输入共享、输出共享、全共享。而对于全连接层，由于只出现在模型的最后一层，其输出维度永远为$100$（cifar-100数据集的类别数），所以，我们只需要考虑两种共享方式：
- 完全独立（Independent）：每个输入不一样的FC层使用完全独立的参数。
- 全共享（Full）：所有FC层使用同一组参数。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/5d31adbea5524238ba3d492178c3f51daac33179a24f434fab90ec2f9a0d5656" width="600"></center>

<center>图2 不同种类权重共享示意图</center>

## 数据和架构采样
对于数据采样，为了保持和子网络训练方式一致，我们采用随机采样的方式从训练样本集中随机抽取大小为$128$的数据样本。而对于架构采样，我们探究了如下两种采样策略：
1. 均匀采样<sup>[1]</sup>：从每层的通道数选择中均匀地采样架构。
2. 公平采样<sup>[2]</sup>：从每层的通道数选择中不放回地抽取每个架构，如果通道数候选集已取完，则重置候选集。这样做的目的是每经过$L$次采样（$L$为通道数可选个数），保证每个通道数选择都能够精准地被采样$1$次。

## 超网络参数优化
对于参数优化策略，我们探究了如下三种方式：
1. 朴素优化<sup>[1]</sup>：直接使用交叉熵损失来进行参数优化。
2. 固定教师的知识蒸馏<sup>[3]</sup>：使用训练好的空间最大网络来指导子模型，使用KL散度损失优化。
3. 动态教师的知识蒸馏<sup>[4]</sup>：教师和学生同时训练，教师使用交叉熵损失，学生使用KL散度损失。同时，我们遵循相关文献中的建议，使用三明治法则来进行参数更新。

## 子网络评估
针对每一个子网络，我们首先将超网络中对应的参数拷贝入子网络。之后，我们测量它在测试集上的交叉熵损失，并以此来作为子网络的效果评估代理。在每个数据批次上，对于批量归一化操作中的动态均值和方差，我们直接使用当前数据批次的均值和方差作为估算，来进一步提升超网络的一致性。

## 实验与结论
我们从最基本的模块设定出发，来探究模块中不同策略对超网络一致性的影响。我们的基线模块设定为：全共享超网络（包括卷积、批量归一化、全连接）+公平采样+朴素优化。
### 超网络构建
首先，我们探究超网络的不同构建方式对超网络一致性的影响：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/05659bc8cc944e4689d3e61048e0fb7667e0f8e6c00c46a28eaf050534277905" width="500"></center>
<center>表1 超网络构建分析对比实验</center>

从表中我们可以看到，卷积层使用全共享机制以外的其他方式，都会使得超网络的一致性降低。而批量归一化操作和全连接操作却有着相反的结论。最终，全共享卷积、输出维度共享的批量归一化以及完全独立的全连接达到了最好的一致性指标，相比于基线模块设定提升了将近$0.03$的肯德尔系数。

### 数据架构采样
之后，我们对比了均匀采样和公平采样两种架构采样方式。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/52548c7e41724378bc2d52868ffcfd4f9c14b56771204aeab9e0f0057ae4e7d0" width="300"></center>
<center>表2 数据架构采样分析对比实验</center>

我们发现公平采样的方式得到的网络一致性要更稳定一些，相对于均匀采样也有着一定的提升。

### 超网络参数优化
而对于超网络参数优化方式，我们首先对比了固定教师的知识蒸馏和朴素优化之间的优劣，因为这两种方式之间唯一的不同在于所用的损失函数形式。从对比表格中我们发现，在神经网络通道数搜索问题中，朴素优化的方式要优于固定教师的知识蒸馏。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/fabb5a7f1a844e219f5f3f2211ff3a96c9778de2d98144bf958c21d48b447f36" width="300"></center>
<center>表3 朴素优化与固定教师的知识蒸馏</center>
而对于动态教师，我们探究了同一更新轮次中不同架构数对一致性的影响：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/26f13f3b0dcb4e458dff9c8b58c3dbe941a139bf2e7a4f6591aa224773434d6f" width="500"></center>
<center>图3 朴素优化与动态教师的知识蒸馏</center>


其中same batch代表对同一更新轮次中所有架构在同一个数据批次上求取损失。从图中我们可以看到，动态教师的蒸馏策略有着显著的超网络一致性提升。而对于same batch策略，关闭它效果有明显下降。

### 将所有最佳实践组合
经过上述方法上的探索，我们最终选择框架中表现最好的各个策略组成最后的超网络构建与训练最佳实践。并测试了其和原有的一些基线模型的效果对比：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/5663f4902bf1497a8b493e3bc11120e8ed916102d8264fad9d903dc060d76fd6" width="500"></center>
<center>表4 最佳实践组合与已有baseline对比</center>

可以看到，我们的最佳算法相比于原有的最佳极限模型提升了将近$0.06$。而在最终的比赛榜单中，我们也以$0.8324$的成绩并列第三。


- [1]	Guo, Zichao, et al. “Single Path One-Shot Neural Architecture Search with Uniform Sampling.” European Conference on Computer Vision, 2019, pp. 544–560.
- [2]	Chu, Xiangxiang, et al. “FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search.” ArXiv Preprint ArXiv:1907.01845, 2019.
- [3]	Cai, Han, et al. “Once for All: Train One Network and Specialize It for Efficient Deployment.” ICLR 2020 : Eighth International Conference on Learning Representations, 2020.
- [4]	Yu, Jiahui, and Thomas Huang. “AutoSlim: Towards One-Shot Architecture Search for Channel Numbers.” ArXiv Preprint ArXiv:1903.11728, 2019.


请遵循以下toturial来使用本库代码

## CVPR2021 NAS track1 3rd Solution Illustration
Brought to you by **Meta_Learners**, **THUMNLab**, **Tsinghua University**

**Team Member**: Chaoyu Guan (关超宇) (**Leader**), Yijian Qin (秦一鉴), Zhikun Wei (卫志坤), Zeyang Zhang (张泽阳), Zizhao Zhang (张紫昭), Xin Wang (王鑫)
### Table of content
- [Train](#I)
- [Evaluate](#II)
- [Tuning](#III)

### <a id="I">Part I. Train </a>
Before training, we need to prepare dataset under `./data` folder. Please first download the cifar100 dataset from [competition website](https://aistudio.baidu.com/aistudio/datasetdetail/76994) and move it to `./data/cifar-100-python.tar.gz`, and download the submit 5w archs from [competition website](https://aistudio.baidu.com/aistudio/datasetdetail/73326) to `./data/Track1_final_archs.json`

Then, we need to import the necessary modules.


```python
import json
import pickle
from tqdm import tqdm

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as opt
from paddle.optimizer.lr import CosineAnnealingDecay, LinearWarmup

from supernet.utils import seed_global, Dataset, str2arch, get_param
from supernet.sample import Generator, strict_fair_sample, uniform_sample
from supernet.super.supernet import Supernet
from supernet.super import super_bn, super_conv, super_fc

# BUG: there are some bugs when seeding paddlepaddle. Even the seed is set for paddlepaddle, the training procedure still has randomness.
# you can run this notebook several times and you will get different supernet performance even when paddle.seed(0) is called inside seed_global.
seed_global(0)
```

Then, we can build a supernet using shared conv, bn and fc module, and load the necessary data for training.


```python
supernet = Supernet(super_bn.BestBN, super_conv.BestConv, super_fc.BestFC)
dataloader = Dataset().get_loader(batch_size=128, mode='train', num_workers=4)
```

Next, we train the supernet following paper ["Universally Slimmable Networks and Improved Training Techniques"](https://arxiv.org/abs/1903.05134). The basic idea is that, for one data batch, we randomly sample $n - 2$ architectures, together with two fixed architectures: The biggest architecture and the smallest architecture. For the biggest architecture, we use ground truth label of data batch for training. For the other architectures, we use knowledge distillation and use the biggest architecture as teacher to guide the training.

The idea behind is that the biggest and smallest architectures represent the performance upper and lower bound of the space. Optimizing both will result in better overall performances.

To get a better rank, we use basically the same training hyperparameters with each sub-architecture training procedure. And use channel-wise fair sampling method borrowed from ["FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search"](https://arxiv.org/abs/1907.01845) to sample the $n - 2$ architectures. The basic idea is that, for every $m\times k$ architectures sampled, every channel in certain layer appears exactly $m$ times, where $k$ is the number of choices that layer can choose.

**NOTE: The following cell will run about 10h to train the supernet on 1 Tesla V100.**


```python
# Hyper Parameters
EPOCH = 300
LR = 0.1
WEIGHT_DECAY = 0.1     # we find that using weight decay 0 will result in better ranks. See chapter III. Tuning
GRAD_CLIP = 5          # we find that using gradient clip will result in slightly better ranks. See chapter III. tuning
SPACES = (
    [[4, 8, 12, 16]] * 7
    + [[4, 8, 12, 16, 20, 24, 28, 32]] * 6
    + [[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]] * 7
)
N = 18                 # We choose n=18 instead of a small n in paper, which we find is more suitable in this competition. See chapter III. tuning
SKIP_CONNECT = True    # We find that turning on skip connection will result in better ranks. See chapter III. tuning

# build an architecture generator sampled according to channel-wise fair sampling
generator = Generator(strict_fair_sample)
max_arch = [max(x) for x in SPACES]
min_arch = [min(x) for x in SPACES]

# build the optimizer
optimizer = opt.Momentum(
    LinearWarmup(CosineAnnealingDecay(LR, EPOCH), 2000, 0.0, LR),
    momentum=0.9,
    parameters=supernet.parameters(),
    weight_decay=WEIGHT_DECAY,
    grad_clip=None if GRAD_CLIP <= 0 else nn.ClipGradByGlobalNorm(GRAD_CLIP),
)
optimizer.clear_grad()

for e in tqdm(range(EPOCH), desc="total"):
    for data in tqdm(dataloader, desc=f"{e} epoch"):
        # sample archs
        archs = [generator() for _ in range(N - 2)] + [min_arch]
        
        # use ground truth label to train the biggest model.
        logit_big = supernet(data[0], max_arch, SKIP_CONNECT)
        loss = F.cross_entropy(logit_big, data[1]) / N
        loss.backward()

        # use output of big model to knowledge distill other models
        with paddle.no_grad():
            distribution = F.softmax(logit_big).detach()
        for arch in archs:
            logit = supernet(data[0], arch, SKIP_CONNECT)
            loss = F.cross_entropy(logit, distribution, soft_label=True) / N
            loss.backward()

        # update parameters of supernet
        optimizer.step()
        optimizer.clear_grad()

        # update scheduler
        if optimizer._learning_rate.last_epoch < optimizer._learning_rate.warmup_steps:
            optimizer._learning_rate.step()
        
        # save models for this epoch
        paddle.save(supernet.state_dict(), f'./saved_models/{e}-model.pdparams')

    # update scheduler
    if optimizer._learning_rate.last_epoch >= optimizer._learning_rate.warmup_steps:
        optimizer._learning_rate.step()
```

After training, we can evaluate all the trained supernets derived under `./saved_models` folder.

### <a id="II">Part II. Evaluate </a>
For evaluation, we simply extract the parameters (without running batch norm statistics) from supernet to form the parameters of every single model. The batch norm statistics are calculated on the fly using the data batch statistics from test set. We find that using this technique will result in better ranks. See [III. tuning](#III) for more details.

We also find that using negative cross entropy loss of single model on test dataset will result in better ranks with ground truth accuracy of model.

**NOTE: the cell below will cost about 8 hours running on one Tesla V100.**


```python
SKIP_CONNECT = True

with paddle.no_grad():
    # build model, load best parameters
    supernet = Supernet(super_bn.BestBN, super_conv.BestConv, super_fc.BestFC)

    '''
    # use the line below to generate the 0.97122 result
    # supernet.set_state_dict(paddle.load('./saved_models/model_97122.pdparams'))
    
    # use the line below to generate the 0.97131 result
    # supernet.set_state_dict(paddle.load('./saved_models/model_97131.pdparams'))
    
    # the 97131 and 97122 is two models reruned following the notebook instruction above
    '''
    supernet.set_state_dict(paddle.load('./saved_models/255-model.pdparams'))

    supernet.eval()
    dataloader = Dataset().get_loader(batch_size=512, mode='test', num_workers=4)

    architecture = json.load(open('./data/Track1_final_archs.json', 'r'))
    for key in tqdm(architecture, desc="all archs"):
        arch = architecture[key]["arch"]
        arch = str2arch(arch)

        loss = 0.
        for data in tqdm(dataloader, desc="test dataset"):
            logit = supernet.inference(data[0], arch, SKIP_CONNECT)
            loss += float(F.cross_entropy(logit, data[1], reduction='sum'))
        architecture[key]["acc"] = - loss

# normalize architecture accs
max_acc = max([architecture[key]['acc'] for key in architecture])
min_acc = min([architecture[key]['acc'] for key in architecture])

for key in architecture:
    architecture[key]['acc'] = (architecture[key]['acc'] - min_acc) / (max_acc - min_acc)

# save the results
json.dump(architecture, open("./saved_models/result_5w.json", "w"))
```

The derived `./saved_models/result_5w.json` is the file we submit online.

### <a id="III">Part III. tuning</a>

In this chapter, we will introduce what we have explored during this competition, and share you the findings.

NOTE: since most of our results are derived using different version of codes, backends and structures, we __only provide the universal findings__ and __how to run these comparison experiments__. We do not report the exact rank scores since most of them are not runned under our newest version of codes.

#### III.a Influence of different sharing strategies

The first thing we've tested is how to share each parts of supernet. We can choose to share fc layer, share bn layer and share conv layer.
- For fc layer, there are two choices:
    - when not shared, every different last channel choice will use different fc layer. [__IndependentFC__]
    - when shared, the fc layer with small channel number will be derived from the biggest fc layer by taking the smallest $c$ channels, where c is the small channel number. [__FullFC__]
- For conv layer, there are 4 ways to do the share:
    - Independent. For one layer, every choice of [in_c, out_c] will use independent kernel with shape [in_c, out_c]. [__IndependentConv__]
    - Front share. For one layer, the conv op with the same input channel number will share the same kernel, wich the small kernel extracted from the shared kernel by choosing neurons with small ids. [__FrontShareConv__]
    - End share. For one layer, the conv op with the same output channel number will share the same kernel, wich the small kernel extracted from the shared kernel by choosing neurons with small ids. [__EndShareConv__]
    - Full share. Every op in one layer is a slice of the shared kernel by choosing the small ids. [__FullConv__]
- For bn layer, we can also have 4 ways to do the share, similar to the situation in conv layer.
    - Independent. Every choice of [in_c, out_c] will use independent bn ops of shape [out_c]. [__IndependentBN__]
    - Front share. For one layer, the bn op with the same conv input channel number will be shared. [__FrontShareBN__]
    - End share. For one layer, the bn op with the same conv output channel number will be shared. [__EndShareBN__]
    - Full share. Use the same bn for every op choices. [__FullBN__]

There are so many cross combinations, so we __do not iterate__ the whole space. Instead, we __start from the fully shared supernet (FullConv, FullBN, FullFC)__, and test whether the rank will improve when we change the share type of bn, conv and fc. Specially, we also test the basic combinations to use both _xxxConv_ and _xxxBN_, since they share the same logic when sharing.

In total, we test the following combinations:
1. FullConv, FullBN, FullFC
2. IndependentConv, FullBN, FullFC
3. FrontShareConv, FullBN, FullFC
4. EndShareConv, FullBN, FullFC
5. FullConv, IndependentBN, FullFC
6. FullConv, FrontShareBN, FullFC
7. FullConv, EndShareBN, FullFC
8. FullConv, FullBN, IndependentFC
9. FrontShareConv, FrontShareBN, FullFC
10. EndShareConv, EndShareBN, FullFC
11. IndependentConv, IndependentBN, FullFC

You can use the following command to run every combination by replacing xxxconv xxbn xxfc to the corresponding classes.


```python
!python -m supernet.scripts.train --convclass xxxconv --bnclass xxbn --fcclass xxfc
```

Conclusion: __FullConv, EndShareBN, IndependentFC supernet performs best__.

#### III.b sample strategy

The second exploration is how to sample architectures to train the supernet. We mainly explore 3 strategies.

- uniform sample. Sample uniformly.
- fair sample. Sample fairly following ["FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search"](https://arxiv.org/abs/1907.01845)
- sample according to architecture parameter size. Architectures with bigger parameter size will have higher probability to be sampled.

Using following commands to explore sample strategy.


```python
# first, generate the archs for `sampling according to architecture parameter size`
archs = uniform_sample(1000000)
arch_param = [get_param(a)[1] / 1000000 for a in archs]
sums = sum(arch_param)
arch_prob = [x / sums for x in arch_param]

pickle.dump(archs, open('./data/arch_1m.bin', 'wb'))
pickle.dump(arch_prob, open('./data/arch_1m_prob.bin', 'wb'))

!python -m supernet.scripts.train --sample uniform
!python -m supernet.scripts.train --sample fair
!python -m supernet.scripts.train --sample fixed --train_arch ./data/arch_1m.bin --train_arch_prob ./data/arch_1m_prob.bin
```

Conclusion: fair sample performs slightly better among all results.

#### III.c Sandwich rule, knowledge distillation
To test whether the sandwich rule and knowledge distillation make effects, we also explore the use of both. You can derive the supernet by running the following codes.


```python
# use sandwich rule and knowledge distillation
!python -m supernet.scripts.train --n 6 --kd --sandwich
# use sandwich rule only
!python -m supernet.scripts.train --n 6 --sandwich
# do not use any
!python -m supernet.scripts.train --n 6

# use small n to rerun
!python -m supernet.scripts.train --n 3 --kd --sandwich
!python -m supernet.scripts.train --n 3 --sandwich
!python -m supernet.scripts.train --n 3
```

Conclusion: sandwich rule will only be effective when knowledge distillation is on and when $n$ is large(r than $5$). Sandwich rule + knowledge distillation can result in higher kendall score, while do not use both will result in higher pearson score.

#### III.d Hyper parameters
We also tune the lr, weight decay, gradient clip, training epoch, etc. You can use the following command to run.


```python
!python -m supernet.scripts.train --weight_decay xxx --lr xxx --epoch xxx --clip xxx
```

Conclusion: weight decay should be 0, gradient clip = 5 is better. lr and epoch should remain the same with training ground truth: lr=0.1, epoch=300

#### III.e N
The main last hyper parameter during training we explore is N.


```python
!python -m supernet.scripts.train --n xxx --kd --sandwich
```

Conclusion: n=18 is the best

#### III.f Evaluation

We explore the evaluation methods. There are mainly two fields to explore.
- The metric to use
    - negative cross entropy loss [loss]
    - top1 accuracy [acc]
- The bn statistics to use
    - those in supernet during training [0]
    - calibrate them on training dataset [1]
    - calibrate them on test dataset [2]
    - calculate on the fly during testing [3]

You can evaluate them using following commands:



```python
!python -m supernet.scripts.evaluate --path path/to/pdparams --path_to_arch path/to/arch --metric loss or acc --bn_mode 0 or 1 or 2 or 3
```

Conclusion: loss is better than acc, while calibrate bn performs similar with calculate on the fly. Directly using supernet statistics performs worst. For speed reason, we use 3 at last.

#### III.g Others

We've also test some other strategies, which is not so systematic or important, so we omit them in this notebook, you can refer to `./supernet/scripts/train` for more details.


```python

```
