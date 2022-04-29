# hfai.nn实验



## hfai介绍

我们只需要在代码中加入：

```python
model = hfai.nn.to_hfai(model)
```

就可以使用到hfai提供的算子优化。



## 背景

除了HFReduce，Hfai.nn也是一个强大的模型训练加速工具。hfai.nn提供了对当前主流深度学习模型中常用的MultiHeadAttention，LayerNorm，LSTM等结构中算子的深度优化，能够大幅加速模型运算中使用到这些算子部分的速度。我们自然希望探索在Alphafold中引入hfai.nn算子库来能够对模型训练产生多大的增益效果。



## Alphafold GPU开销分析

由于hfai.nn的算子加速取决于模型中各类型运算占总GPU开销的比重情况，因此我们首先尝试使用torch.profiler工具对Alphafold模型使用标准torch算子时的GPU运算时间进行一些分析。在一次耗时为12.6s的迭代中，主要开销情况如下：

| Name                                                    | CUDA Time | CUDA % |
| ------------------------------------------------------- | --------- | ------ |
| aten::native_layer_norm                                 | 2.105s    | 16.81% |
| void at::native::(anonymous namespace)::RowwiseMomen... | 1.866s    | 14.90% |
| aten::mm                                                | 1.716s    | 13.70% |
| aten::native_layer_norm_backward                        | 1.642s    | 13.11% |
| atten::add_                                             | 1.454s    | 11.61% |

仅前5类运算占了Alphafold训练开销的70%，而其中LayerNorm与Attention中的矩阵运算又是最主要的耗时来源：仅LayerNorm运算就占了总耗时的30%左右。Alphafold相对一般的BERT等Transformer类模型更为复杂，使用了自己实现的Attention，因此在注意力运算上无法获得hfia.nn的加速。但同样占据耗时大头的LayerNorm却能够使用到hfai算子加速，因此可以预期hfai.nn能给alphafold带来较好的加速效果。

为了探究具体加速效果，我们首先尝试对Alphafold中LayerNorm的使用情况进行一些分析。一般来说输入Tensor的形状往往会对算子加速的效果产生较大的影响，因此在这里我们首先尝试对输入模型中的LayerNorm层的不同的Tensor形状进行一些理论上的分析。在一个典型Transformer类模型中，LayerNorm的输入形状往往是[BatchSize, SeqLen, EmbDim]。在Alphafold中由于其模型结构与简单的Transformer模型有较大差异，输入的张量形状会更为特殊一些。模型中LayerNorm层的不同Input Shape对应的出现频率可见下表：（第一维为BatchSize，训练时在单个GPU上大小为1）

| Input Shape      | Run%   | Relative Perf% | Torch Run Time(s) | Hfai Run Time(s) |
| ---------------- | ------ | -------------- | ----------------- | ---------------- |
| [1,256,256,128]  | 59.96% | 364.07%        | 0.003             | 0.0008           |
| [1,132,256,256]  | 13.2%  | 267.02%        | 0.0017            | 0.0007           |
| [1,1,256,256,64] | 8.03%  | 379.80%        | 0.0028            | 0.0007           |
| [1,128,256,256]  | 7.92%  | 276.52%        | 0.0017            | 0.0006           |
| [1,256,132,256]  | 4.4%   | 276.13%        | 0.0017            | 0.0006           |
| [1,256,128,256]  | 2.64%  | 272.92%        | 0.0017            | 0.0006           |
| [1,256,384]      | 1.87%  | 80.50%         | 0.0002            | 0.0003           |
| [1,1024,256,64]  | 1.32%  | 471.77%        | 0.0111            | 0.0024           |
| [1,256,1024,64]  | 0.44%  | 474.66%        | 0.0111            | 0.0024           |
| [1,4,256,256,64] | 0.11%  | 474.41%        | 0.0111            | 0.0024           |
| [1,256,256]      | 0.11%  | 93.81%         | 0.0003            | 0.0003           |

可见在大多数输入时，Hfai.nn提供的LayerNorm算子都能取得相比torch.nn数倍的性能提升，只在极少数形状下性能可能会稍差于torch。从上表可以计算得，使用hfai.nn在Alphafold中预期可以在LayerNorm的GPU开销上取得350%的加速。从前面的分析中可见Alphafold中的LayerNorm使用频率很高，且在模型的不同部分都有使用，因此可以预计对模型整体也能有较明显的加速效果。



## Hfai实际训练加速

在理论分析了hfai.nn能给Alphafold带来的加速幅度后，我们也希望能了解在实际训练时Alphafold能够从hfai.nn中获得的收益。因此我们在与Alphafold预训练时相同设定的真实场景下使用128卡进行了并行训练，分别测试了使用torch算子和hfai算子进行Alphafold训练时的训练耗时情况，结果如下表：

| 算子     | GPU数 | 迭代次数 | 平均时长 |
| -------- | ----- | -------- | -------- |
| torch.nn | 128   | 200      | 11.26s   |
| hfai.nn  | 128   | 200      | 8.64s    |

可见在Alphafold训练时只需要添加一行代码引入hfai.nn，就可以将模型整体训练时单次迭代的时长从11.26秒减少到8.64秒，足足能够获得30%的训练性能提升。值得注意的是，由于Alphafold中自定义了Attention实现没有使用标准的nn.MultiHeadAttention，能够获得hfai.nn算子加速的其实只有LayerNorm。由此可见在常见的标准Transformer类模型中，使用hfai.nn将更加容易获得大幅的性能提升。

