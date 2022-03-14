# Alphafold Model Profile

DDP Backend: NCCL
总GPU：128卡

## 基本情况

本实验目的在于使用Pytorch Profile对模型的训练耗时进行更细粒度的分析。

从Pytorch Profile的结果来看，在一次长为12.6秒的迭代中，Allreduce的耗时仅为0.4秒左右，占总耗时不到5%。对于使用HFreduce的结果由于Pytorch Profile并不兼容，因此仍然无法获知这一部分的开销。但从一般DDP的结果依然可见这一部分开销占总耗时的占比非常之小，因此很难从总迭代时间中感受到HFReduce带来的优化效果。

对于使用HFReduce的训练，我们尝试了统计Sync_grad()过程的开销，但发现在每次迭代中不同进程上此函数的开销差异很大，推测是由于提前于其他进程完成迭代后的等待时间也被包含在内。此时Sync_grad函数耗时的平均值大约也在0.4秒，并不能如实反映其耗时，准备尝试取每个batch中耗时最小的进程的梯度同步时间作为HFreduce下的此部分开销再与DDP结果进行对比。

另外Pytorch Profile也提供了对模型各类计算的开销统计。由于产出的profile log大小过大无法被tensorboard打开，从结果中粗略可见layer_norm和attention乘法是算子中开销最大的，说明Alphafold模型确实有很大的从hfai算子优化中获得提升的空间。

## Profile结果

结果仅包含了CUDA运算时间统计，并且排除了占比小于1%的运算。

```
-------------------------------------------------------   ------------  ------------  ------------  ------------ 
                                                   Name      Self CUDA   Self CUDA %    CUDA total  CUDA time avg
-------------------------------------------------------   ------------  ------------  ------------  ------------ 
                                aten::native_layer_norm         2.105s        16.81%        2.224s     638.068us 
void at::native::(anonymous namespace)::RowwiseMomen...         1.866s        14.90%        1.866s     535.426us 
                                               aten::mm         1.716s        13.70%        1.716s      96.875us 
                       aten::native_layer_norm_backward         1.642s        13.11%        1.665s       2.372ms 
void at::native::unrolled_elementwise_kernel<at::nat...         1.468s        11.72%        1.468s     126.801us 
                                             aten::add_         1.454s        11.61%        1.454s      57.029us 
void at::native::(anonymous namespace)::GammaBetaBac...         1.404s        11.21%        1.404s       2.056ms 
                                              aten::bmm         1.079s         8.62%        1.294s     272.094us 
                                            aten::copy_      959.786ms         7.66%     960.038ms      44.927us 
void at::native::unrolled_elementwise_kernel<at::nat...      938.584ms         7.49%     938.584ms      59.213us 
                                              aten::mul      807.203ms         6.45%     807.203ms      30.193us 
void at::native::vectorized_elementwise_kernel<4, at...      614.885ms         4.91%     614.885ms      22.070us 
                                             aten::mul_      563.038ms         4.50%     563.038ms      31.781us 
void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_1...      502.353ms         4.01%     502.353ms      79.286us 
void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_1...      498.088ms         3.98%     498.088ms     532.145us 
ncclKernel_AllReduce_RING_LL_Sum_float(ncclWorkElem)...      494.207ms         3.95%     494.207ms      30.888ms 
void at::native::unrolled_elementwise_kernel<at::nat...      460.159ms         3.67%     460.159ms      24.643us 
                                         aten::_softmax      450.283ms         3.60%     450.283ms     389.182us 
void (anonymous namespace)::softmax_warp_forward<flo...      449.987ms         3.59%     449.987ms     397.515us 
void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_2...      403.262ms         3.22%     403.262ms     188.528us 
void at::native::vectorized_elementwise_kernel<4, at...      390.407ms         3.12%     390.407ms      11.523us 
                                              aten::add      385.774ms         3.08%     385.774ms      25.333us 
void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_6...      314.098ms         2.51%     314.098ms     275.525us 
void at::native::vectorized_elementwise_kernel<4, at...      307.775ms         2.46%     307.775ms     121.171us 
void at::native::(anonymous namespace)::LayerNormFor...      238.816ms         1.91%     238.816ms      68.527us 
void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_2...      181.126ms         1.45%     181.126ms      75.469us 
void at::native::(anonymous namespace)::ComputeInter...      179.908ms         1.44%     179.908ms     257.011us 
                                              aten::sum      176.345ms         1.41%     176.372ms      54.285us 
                                          aten::sigmoid      146.771ms         1.17%     146.771ms      50.264us 
void at::native::vectorized_elementwise_kernel<4, at...      146.771ms         1.17%     146.771ms      50.264us 
void at::native::reduce_kernel<128, 4, at::native::R...      140.724ms         1.12%     140.724ms      96.718us 
                           aten::_softmax_backward_data      135.478ms         1.08%     271.527ms       1.165ms 
void (anonymous namespace)::softmax_warp_backward<fl...      135.424ms         1.08%     135.424ms     593.965us 
void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_1...      132.605ms         1.06%     132.605ms     179.196us 
```