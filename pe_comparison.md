# 位置机制的统一训练对比

在相同模型骨架、数据和训练预算下，比较语言模型及图像分类中的位置机制。语言模型与图像分类结果分别见实验 1 和实验 2。表格按各实验指定主指标从差到好排列，最佳观测值位于最下方；这不代表跨任务综合排名或统计显著差异。各指标仅在同组全部方法具有完整正式结果后展示，未完成的指标组不设占位列，完成后自动补入原表。

## 效率测量的共用配置

- 硬件与软件：H20 96 GB、PyTorch 2.8.0+cu128。同硬件训练基准使用同一台机器的 4 卡 DDP，推理使用单卡；各方法依次测量以避免互相争用。结果针对本仓库实现，不等同论文作者的其他 kernel 实现。
- 训练计时：预热 20 步后计时 100 次完整更新，包含前向、反向、DDP 通信和优化器更新。输入提前放入 GPU，数据读取、评测及保存不计入；每次同步 CUDA，取最慢 rank 耗时，再报告均值和吞吐。梯度累积为 1，activation checkpointing 关闭。
- 指标与单位：参数量 M=10⁶，显存 GiB=2³⁰ bytes；耗时和显存越低越好，吞吐越高越好。训练显存取四卡中最大的 peak allocated，推理显存为单卡 peak allocated，不等同 nvidia-smi 占用。
- 结果口径：效果列使用最终训练权重；同硬件效率列使用各实验声明的权重和固定输入形状，不更新正式训练权重。云训练列包含实际运行开销，与同硬件基准分别解释；单次计时不用于推断跨机器或跨运行的稳定排序。
- 原始测量：中位数、P95、reserved memory、环境与源码哈希保存在[语言模型基准](/cfs/common_users/zhijian.qiao/outputs/convpe-reproduction/20261031/efficiency/lm)及[图像基准](/cfs/common_users/zhijian.qiao/outputs/convpe-reproduction/20261031/efficiency/vision)中。

## 实验 1：FineWeb-Edu 语言模型

**目的：**

比较位置机制对语言模型预测、常规任务和长度外推的影响。

**配置：**

- 训练集：FineWeb-Edu sample/10BT 前四个 parquet；[fineweb-4parquet.bin](/cfs/common_users/zhijian.qiao/code/convpe/data/fineweb-4parquet.bin) 含 3,001,607,756 个 uint16 GPT-2 token，无额外文档分隔符。序列长度 1024；保留完整 block，最后一个全局 batch 按同一规则补齐。
- 测试集：lm-eval 的 WikiText、LAMBADA、HellaSwag、XStoryCloze、PIQA、ARC-Easy、WinoGrande、BoolQ、TruthfulQA 九项任务，分别使用 62 篇、5153、10042、1511、1838、2376、1267、3270、817 条样本。七个方法共用完整离线数据及文件哈希，见 [datasets.json](/cfs/common_users/zhijian.qiao/outputs/convpe-reproduction/20261031/eval-offline/datasets.json)。另含 RULER 十二项任务，每任务每长度 500 条固定样本。训练 bin 开头 1,048,576 tokens 的 loss 仅用于复现原论文训练数据诊断。
- 指标：表格以最大上下文 8192 的 WikiText word-PPL 为排序主指标，从高到低排列。WikiText/LAMBADA perplexity 越低越好；各任务 accuracy/accuracy-normalized 越高越好，逐任务报告，不混合不同尺度。RULER 使用相同任务样本及评分规则。
- 初始化与训练范围：seed 42，从头全量训练；七个方法共用初始骨架参数。12 层、宽度 768、12 heads、head dim 128、MQA、共享 K=V、query 归一化、attention sink，词表 50304。
- 资源与预算：Composite 和 DAPE 使用 2 节点 × 8 H20、每卡 1 条序列；其余方法使用 1 节点 × 8 H20、每卡 2 条序列。均为 DDP，不启用分片；GBS 16，183,204 步，梯度累积 1，activation checkpointing 关闭。Composite 和 DAPE 从已有 checkpoint 续训，保留优化器状态及学习率进度；相邻两 rank 合并有效 token 分母，保持原每两条序列的 loss 归一规则。
- 权重共享与复现边界：当前 Transformers 4.57.3 下，输入词嵌入和输出矩阵未实际共享；RoPE 为 163.470M、ConvPE 为 166.764M，均比共享版本多 38.633M 参数。临时显式绑定后参数量分别为 124.836M/128.131M，与论文计数一致，但实际训练 checkpoint 已形成两份不同权重及独立优化器状态，不能通过改统计口径消除差异；依据见 [parameter-sharing-audit.json](/cfs/common_users/zhijian.qiao/outputs/convpe-reproduction/20261031/efficiency/lm/parameter-sharing-audit.json)。本组效果与效率均针对未共享版本，不能作为论文原规模的严格复现。
- 优化器：排除名称含 emb/head 的二维参数用 Muon，LR 0.003、momentum 0.95、weight decay 0.01；其他参数用 Adam，LR 0.0003、betas=(0.9,0.999)、epsilon=1e-8、weight decay=0。Muon 使用固定 PyTorch 2.10 官方源码兼容实现。
- 调度与精度：cosine，warmup 500 updates；BF16 autocast，attention score FP32；全局梯度范数裁剪 4.0。全方法同一调度，保留第十步、每 2000 步及末步 checkpoint。
- 评测配置：独立评测环境固定 lm-eval 0.4.9.2、Transformers 4.57.1，保持 checkpoint 的未共享词嵌入；BF16。九项常规任务沿用各任务默认模板和评分，实际均为 zero-shot，batch 4、最大上下文 8192。评测 random/numpy/torch/fewshot seeds 分别为 0/1234/1234/1234。WikiText 报告 word-perplexity；训练数据 loss 使用 1024/2048/4096/8192，独立于测试集指标。单次 seed 42 训练不用于推断跨种子稳定性。
- RULER 配置：标称长度 1024/2048/4096，生成器内部 seed 42，七模型共用 18,000 条固定样本；batch 1、最大上下文 5120、无 KV cache 贪心生成，沿用各任务输出预算与官方评分。4096 为相对作者脚本启用的 1024/2048 的额外长度。SQuAD 数据生成修复文档数减至零导致的死循环，保留问题、答案、样本数及评分；部分官方 FWE 和 2048 长度的 niah_single_2 样本超过标称预算，按原样保留，实际输入加输出预算均不超过 5120。样本、长度范围及修正见 [manifest.json](/cfs/common_users/zhijian.qiao/outputs/convpe-reproduction/20261031/ruler-data/manifest.json)。
- 同硬件训练效率：GBS 16，序列长度 1024，BF16；读取表中各方法中途 checkpoint 及 Adam/Muon 状态，固定恢复的学习率。使用同一组 16 个 FineWeb block；每卡四条序列分成两组，各组按有效标签求均值后再平均，保持原每两条序列的 loss 语义。采用控制变量中说明的 Composite/DAPE 优化实现；权重步数和哈希见 [manifest.json](/cfs/common_users/zhijian.qiao/outputs/convpe-reproduction/20261031/efficiency/lm/manifest.json)。
- 同硬件推理效率：使用与训练效率基准对应的中途权重，单卡 BF16、batch 1。Prefill 为长度 1024 的完整前向，预热 10 次、计时 30 次；生成使用长度 512 的输入并生成 32 tokens，预热 1 次、计时 5 次。无 KV cache，每步重算完整前缀，不能与有缓存的 decode 吞吐直接比较；推理显存取 prefill 与生成阶段的合计峰值。
- 云训练窗口：表中固定的约 2000 步窗口均包含一次 checkpoint 保存，采样时刻与来源见 [training.json](/cfs/common_users/zhijian.qiao/outputs/convpe-reproduction/20261031/efficiency/cloud/training.json)。节点数不同，不能以该列直接排名算法效率；同硬件基准列用于比较固定资源下的开销。

**控制变量：**

- **位置机制**；公共骨架与训练配置保持一致。
  - **RoPE**：作者旋转维度 64，base 10000，保留共享 KV 和输出逆旋转。
  - **ConvPE**：w128d16r0h1g1，沿用作者 TileLang 参考的窗口索引方向。
  - **NoPE**：关闭显式位置机制，保留因果 mask。
  - **ALiBi**：官方每 head 固定线性距离斜率。
  - **CABLE**：cable6 的上下文距离累积与 query scale。
  - **DAPE**：DAPE V2–Kerple，卷积 1×3，隐藏宽度 32，卷积前下三角输入；采用融合输入整理与 channels-last 布局，保留数学公式和参数。浮点归约顺序可能改变，不保证位级一致。
  - **Composite**：ACL 2021 主方案 qk+convfixed+convq，位置半径 8；移植到因果 MQA 骨架。反向计算采用按距离桶归约，保留数学公式；浮点归约顺序与原索引实现不同，不保证位级一致。

<!-- LM_RESULTS_START -->
<table>
<thead>
<tr><th rowspan="2">方法</th><th rowspan="2">参数 M</th><th rowspan="2">GBS</th><th rowspan="2">训练 Step</th><th rowspan="2">效率权重 Step</th><th colspan="1">最终权重效果</th><th colspan="3">同硬件训练：4 卡 H20</th><th colspan="4">同硬件推理：单卡 H20</th><th colspan="4">云训练采样窗口</th><th data-convpe-autofill="group" colspan="3">完整评测</th></tr>
<tr><th>WikiText word-PPL ↓（最大上下文 8192）</th><th>ms/更新 ↓</th><th>输入 ktoken/s ↑</th><th>显存 GiB ↓</th><th>B1 prefill ms ↓</th><th>输入 ktoken/s ↑</th><th>无缓存生成 token/s ↑</th><th>显存 GiB ↓</th><th>节点×卡</th><th>采样 Step</th><th>ms/更新 ↓</th><th>输入 ktoken/s ↑</th><th data-convpe-autofill="task">常规任务（逐项）</th><th data-convpe-autofill="ruler">RULER（逐项×长度）</th><th data-convpe-autofill="train_bin">训练数据诊断 PPL</th></tr>
</thead>
<tbody>
<tr><td>NoPE <a href="#ref-6">[6]</a></td><td>163.470</td><td>16</td><td>183204</td><td>60000</td><td>5290.01</td><td>141.30</td><td>115.95</td><td>10.77</td><td>11.31</td><td>90.54</td><td>139.15</td><td>1.22</td><td>1×8</td><td>58200–60200</td><td>91.73</td><td>178.61</td><td data-convpe-autofill="task"><details><summary>查看逐项指标</summary>lambada_openai: perplexity=189.5702, acc=0.1962, acc_stderr=0.0055, perplexity_stderr=9.0394<br>hellaswag: acc=0.2847, acc_norm=0.2983, acc_norm_stderr=0.0046, acc_stderr=0.0045<br>xstorycloze_en: acc=0.5427, acc_stderr=0.0128<br>piqa: acc=0.6143, acc_norm=0.6034, acc_norm_stderr=0.0114, acc_stderr=0.0114<br>arc_easy: acc=0.5084, acc_norm=0.4444, acc_norm_stderr=0.0102, acc_stderr=0.0103<br>winogrande: acc=0.5312, acc_stderr=0.0140<br>boolq: acc=0.5495, acc_stderr=0.0087<br>truthfulqa_mc1: acc=0.2570, acc_stderr=0.0153</details></td><td data-convpe-autofill="ruler"><details><summary>查看逐项指标</summary>niah_multikey_1_1024: 0.1860<br>niah_multikey_1_2048: 0.0000<br>niah_multikey_1_4096: 0.0000<br>niah_multikey_2_1024: 0.0020<br>niah_multikey_2_2048: 0.0000<br>niah_multikey_2_4096: 0.0000<br>niah_multikey_3_1024: 0.0000<br>niah_multikey_3_2048: 0.0000<br>niah_multikey_3_4096: 0.0000<br>niah_multiquery_1024: 0.1515<br>niah_multiquery_2048: 0.0000<br>niah_multiquery_4096: 0.0000<br>niah_multivalue_1024: 0.1160<br>niah_multivalue_2048: 0.0000<br>niah_multivalue_4096: 0.0000<br>niah_single_1_1024: 0.8120<br>niah_single_1_2048: 0.0000<br>niah_single_1_4096: 0.0000<br>niah_single_2_1024: 0.8260<br>niah_single_2_2048: 0.0000<br>niah_single_2_4096: 0.0000<br>niah_single_3_1024: 0.1560<br>niah_single_3_2048: 0.0000<br>niah_single_3_4096: 0.0000<br>ruler_cwe_1024: 0.0026<br>ruler_cwe_2048: 0.0002<br>ruler_cwe_4096: 0.0000<br>ruler_fwe_1024: 0.0000<br>ruler_fwe_2048: 0.0000<br>ruler_fwe_4096: 0.0000<br>ruler_qa_squad_1024: 0.0573<br>ruler_qa_squad_2048: 0.0180<br>ruler_qa_squad_4096: 0.0000<br>ruler_vt_1024: 0.0080<br>ruler_vt_2048: 0.0000<br>ruler_vt_4096: 0.0000</details></td><td data-convpe-autofill="train_bin"><details><summary>查看逐项指标</summary>1024: 24.3408<br>2048: 173.7087<br>4096: 1389.2511<br>8192: 3618.9891</details></td></tr>
<tr><td>Composite <a href="#ref-4">[4]</a></td><td>163.498</td><td>16</td><td>183204</td><td>26000</td><td>1015.42</td><td>180.02</td><td>91.01</td><td>11.01</td><td>15.00</td><td>68.26</td><td>106.66</td><td>1.23</td><td>2×8</td><td>24100–26100</td><td>94.90</td><td>172.64</td><td data-convpe-autofill="task"><details><summary>查看逐项指标</summary>lambada_openai: perplexity=148.5015, acc=0.2183, acc_stderr=0.0058, perplexity_stderr=6.8595<br>hellaswag: acc=0.2947, acc_norm=0.3201, acc_norm_stderr=0.0047, acc_stderr=0.0045<br>xstorycloze_en: acc=0.5539, acc_stderr=0.0128<br>piqa: acc=0.6197, acc_norm=0.6164, acc_norm_stderr=0.0113, acc_stderr=0.0113<br>arc_easy: acc=0.5269, acc_norm=0.4731, acc_norm_stderr=0.0102, acc_stderr=0.0102<br>winogrande: acc=0.5170, acc_stderr=0.0140<br>boolq: acc=0.6098, acc_stderr=0.0085<br>truthfulqa_mc1: acc=0.2411, acc_stderr=0.0150</details></td><td data-convpe-autofill="ruler"><details><summary>查看逐项指标</summary>niah_multikey_1_1024: 0.1960<br>niah_multikey_1_2048: 0.0380<br>niah_multikey_1_4096: 0.0000<br>niah_multikey_2_1024: 0.0200<br>niah_multikey_2_2048: 0.0000<br>niah_multikey_2_4096: 0.0000<br>niah_multikey_3_1024: 0.0020<br>niah_multikey_3_2048: 0.0000<br>niah_multikey_3_4096: 0.0000<br>niah_multiquery_1024: 0.1040<br>niah_multiquery_2048: 0.0005<br>niah_multiquery_4096: 0.0000<br>niah_multivalue_1024: 0.0705<br>niah_multivalue_2048: 0.0020<br>niah_multivalue_4096: 0.0000<br>niah_single_1_1024: 0.8760<br>niah_single_1_2048: 0.0000<br>niah_single_1_4096: 0.0000<br>niah_single_2_1024: 0.8980<br>niah_single_2_2048: 0.0000<br>niah_single_2_4096: 0.0000<br>niah_single_3_1024: 0.8540<br>niah_single_3_2048: 0.0000<br>niah_single_3_4096: 0.0000<br>ruler_cwe_1024: 0.0024<br>ruler_cwe_2048: 0.0012<br>ruler_cwe_4096: 0.0002<br>ruler_fwe_1024: 0.0007<br>ruler_fwe_2048: 0.0007<br>ruler_fwe_4096: 0.0000<br>ruler_qa_squad_1024: 0.0828<br>ruler_qa_squad_2048: 0.0627<br>ruler_qa_squad_4096: 0.0100<br>ruler_vt_1024: 0.0100<br>ruler_vt_2048: 0.0000<br>ruler_vt_4096: 0.0000</details></td><td data-convpe-autofill="train_bin"><details><summary>查看逐项指标</summary>1024: 21.3446<br>2048: 38.1464<br>4096: 246.1553<br>8192: 984.3409</details></td></tr>
<tr><td>RoPE <a href="#ref-3">[3]</a></td><td>163.470</td><td>16</td><td>183204</td><td>56000</td><td>224.20</td><td>149.03</td><td>109.94</td><td>10.77</td><td>13.01</td><td>78.73</td><td>94.81</td><td>1.23</td><td>1×8</td><td>55200–57200</td><td>96.99</td><td>168.92</td><td data-convpe-autofill="task"><details><summary>查看逐项指标</summary>lambada_openai: perplexity=155.9269, acc=0.2239, acc_stderr=0.0058, perplexity_stderr=7.3541<br>hellaswag: acc=0.2879, acc_norm=0.3114, acc_norm_stderr=0.0046, acc_stderr=0.0045<br>xstorycloze_en: acc=0.5467, acc_stderr=0.0128<br>piqa: acc=0.6349, acc_norm=0.6246, acc_norm_stderr=0.0113, acc_stderr=0.0112<br>arc_easy: acc=0.5025, acc_norm=0.4407, acc_norm_stderr=0.0102, acc_stderr=0.0103<br>winogrande: acc=0.5130, acc_stderr=0.0140<br>boolq: acc=0.6037, acc_stderr=0.0086<br>truthfulqa_mc1: acc=0.2448, acc_stderr=0.0151</details></td><td data-convpe-autofill="ruler"><details><summary>查看逐项指标</summary>niah_multikey_1_1024: 0.1740<br>niah_multikey_1_2048: 0.1800<br>niah_multikey_1_4096: 0.0000<br>niah_multikey_2_1024: 0.0000<br>niah_multikey_2_2048: 0.0000<br>niah_multikey_2_4096: 0.0000<br>niah_multikey_3_1024: 0.0000<br>niah_multikey_3_2048: 0.0000<br>niah_multikey_3_4096: 0.0000<br>niah_multiquery_1024: 0.0685<br>niah_multiquery_2048: 0.0115<br>niah_multiquery_4096: 0.0000<br>niah_multivalue_1024: 0.1490<br>niah_multivalue_2048: 0.0460<br>niah_multivalue_4096: 0.0000<br>niah_single_1_1024: 0.4260<br>niah_single_1_2048: 0.0080<br>niah_single_1_4096: 0.0000<br>niah_single_2_1024: 0.5380<br>niah_single_2_2048: 0.0020<br>niah_single_2_4096: 0.0000<br>niah_single_3_1024: 0.0040<br>niah_single_3_2048: 0.0080<br>niah_single_3_4096: 0.0000<br>ruler_cwe_1024: 0.0028<br>ruler_cwe_2048: 0.0018<br>ruler_cwe_4096: 0.0028<br>ruler_fwe_1024: 0.0040<br>ruler_fwe_2048: 0.0000<br>ruler_fwe_4096: 0.0093<br>ruler_qa_squad_1024: 0.0633<br>ruler_qa_squad_2048: 0.0587<br>ruler_qa_squad_4096: 0.0187<br>ruler_vt_1024: 0.0060<br>ruler_vt_2048: 0.0000<br>ruler_vt_4096: 0.0000</details></td><td data-convpe-autofill="train_bin"><details><summary>查看逐项指标</summary>1024: 22.8908<br>2048: 24.1881<br>4096: 55.0092<br>8192: 207.5764</details></td></tr>
<tr><td>ConvPE <a href="#ref-11">[11]</a></td><td>166.764</td><td>16</td><td>183204</td><td>52000</td><td>48.31</td><td>156.00</td><td>105.02</td><td>11.21</td><td>12.79</td><td>80.07</td><td>102.91</td><td>1.24</td><td>1×8</td><td>51000–53000</td><td>104.31</td><td>157.08</td><td data-convpe-autofill="task"><details><summary>查看逐项指标</summary>lambada_openai: perplexity=96.3570, acc=0.2614, acc_stderr=0.0061, perplexity_stderr=4.3086<br>hellaswag: acc=0.2980, acc_norm=0.3336, acc_norm_stderr=0.0047, acc_stderr=0.0046<br>xstorycloze_en: acc=0.5606, acc_stderr=0.0128<br>piqa: acc=0.6360, acc_norm=0.6202, acc_norm_stderr=0.0113, acc_stderr=0.0112<br>arc_easy: acc=0.5417, acc_norm=0.4912, acc_norm_stderr=0.0103, acc_stderr=0.0102<br>winogrande: acc=0.4957, acc_stderr=0.0141<br>boolq: acc=0.5786, acc_stderr=0.0086<br>truthfulqa_mc1: acc=0.2313, acc_stderr=0.0148</details></td><td data-convpe-autofill="ruler"><details><summary>查看逐项指标</summary>niah_multikey_1_1024: 0.1540<br>niah_multikey_1_2048: 0.1480<br>niah_multikey_1_4096: 0.0020<br>niah_multikey_2_1024: 0.0080<br>niah_multikey_2_2048: 0.0000<br>niah_multikey_2_4096: 0.0000<br>niah_multikey_3_1024: 0.0000<br>niah_multikey_3_2048: 0.0000<br>niah_multikey_3_4096: 0.0000<br>niah_multiquery_1024: 0.1645<br>niah_multiquery_2048: 0.1405<br>niah_multiquery_4096: 0.0000<br>niah_multivalue_1024: 0.1540<br>niah_multivalue_2048: 0.1280<br>niah_multivalue_4096: 0.0000<br>niah_single_1_1024: 0.9060<br>niah_single_1_2048: 0.7940<br>niah_single_1_4096: 0.3460<br>niah_single_2_1024: 0.9120<br>niah_single_2_2048: 0.0620<br>niah_single_2_4096: 0.0000<br>niah_single_3_1024: 0.7560<br>niah_single_3_2048: 0.3340<br>niah_single_3_4096: 0.0000<br>ruler_cwe_1024: 0.0034<br>ruler_cwe_2048: 0.0006<br>ruler_cwe_4096: 0.0040<br>ruler_fwe_1024: 0.0193<br>ruler_fwe_2048: 0.0373<br>ruler_fwe_4096: 0.0113<br>ruler_qa_squad_1024: 0.0808<br>ruler_qa_squad_2048: 0.0767<br>ruler_qa_squad_4096: 0.0553<br>ruler_vt_1024: 0.0320<br>ruler_vt_2048: 0.0144<br>ruler_vt_4096: 0.0000</details></td><td data-convpe-autofill="train_bin"><details><summary>查看逐项指标</summary>1024: 20.5222<br>2048: 19.9464<br>4096: 20.6651<br>8192: 29.3290</details></td></tr>
<tr><td>ALiBi <a href="#ref-5">[5]</a></td><td>163.470</td><td>16</td><td>183204</td><td>58000</td><td>43.46</td><td>145.67</td><td>112.47</td><td>10.77</td><td>12.92</td><td>79.25</td><td>134.20</td><td>1.22</td><td>1×8</td><td>56300–58300</td><td>94.56</td><td>173.27</td><td data-convpe-autofill="task"><details><summary>查看逐项指标</summary>lambada_openai: perplexity=141.4947, acc=0.2290, acc_stderr=0.0059, perplexity_stderr=6.6721<br>hellaswag: acc=0.2949, acc_norm=0.3216, acc_norm_stderr=0.0047, acc_stderr=0.0046<br>xstorycloze_en: acc=0.5546, acc_stderr=0.0128<br>piqa: acc=0.6322, acc_norm=0.6289, acc_norm_stderr=0.0113, acc_stderr=0.0113<br>arc_easy: acc=0.5286, acc_norm=0.4752, acc_norm_stderr=0.0102, acc_stderr=0.0102<br>winogrande: acc=0.5099, acc_stderr=0.0140<br>boolq: acc=0.5492, acc_stderr=0.0087<br>truthfulqa_mc1: acc=0.2436, acc_stderr=0.0150</details></td><td data-convpe-autofill="ruler"><details><summary>查看逐项指标</summary>niah_multikey_1_1024: 0.1200<br>niah_multikey_1_2048: 0.1140<br>niah_multikey_1_4096: 0.0220<br>niah_multikey_2_1024: 0.0000<br>niah_multikey_2_2048: 0.0000<br>niah_multikey_2_4096: 0.0000<br>niah_multikey_3_1024: 0.0000<br>niah_multikey_3_2048: 0.0000<br>niah_multikey_3_4096: 0.0000<br>niah_multiquery_1024: 0.0585<br>niah_multiquery_2048: 0.0225<br>niah_multiquery_4096: 0.0015<br>niah_multivalue_1024: 0.0750<br>niah_multivalue_2048: 0.0115<br>niah_multivalue_4096: 0.0005<br>niah_single_1_1024: 0.6300<br>niah_single_1_2048: 0.1340<br>niah_single_1_4096: 0.0680<br>niah_single_2_1024: 0.4840<br>niah_single_2_2048: 0.0960<br>niah_single_2_4096: 0.0160<br>niah_single_3_1024: 0.1820<br>niah_single_3_2048: 0.0120<br>niah_single_3_4096: 0.0100<br>ruler_cwe_1024: 0.0212<br>ruler_cwe_2048: 0.0004<br>ruler_cwe_4096: 0.0012<br>ruler_fwe_1024: 0.0007<br>ruler_fwe_2048: 0.0000<br>ruler_fwe_4096: 0.0000<br>ruler_qa_squad_1024: 0.0807<br>ruler_qa_squad_2048: 0.0653<br>ruler_qa_squad_4096: 0.0740<br>ruler_vt_1024: 0.0724<br>ruler_vt_2048: 0.0000<br>ruler_vt_4096: 0.0000</details></td><td data-convpe-autofill="train_bin"><details><summary>查看逐项指标</summary>1024: 21.3703<br>2048: 20.7709<br>4096: 20.6049<br>8192: 20.5226</details></td></tr>
<tr><td>CABLE <a href="#ref-10">[10]</a></td><td>163.691</td><td>16</td><td>183204</td><td>46000</td><td>38.28</td><td>188.05</td><td>87.12</td><td>13.17</td><td>17.67</td><td>57.95</td><td>112.80</td><td>1.22</td><td>1×8</td><td>43900–45900</td><td>119.72</td><td>136.85</td><td data-convpe-autofill="task"><details><summary>查看逐项指标</summary>lambada_openai: perplexity=109.2060, acc=0.2503, acc_stderr=0.0060, perplexity_stderr=5.0522<br>hellaswag: acc=0.2958, acc_norm=0.3269, acc_norm_stderr=0.0047, acc_stderr=0.0046<br>xstorycloze_en: acc=0.5619, acc_stderr=0.0128<br>piqa: acc=0.6404, acc_norm=0.6230, acc_norm_stderr=0.0113, acc_stderr=0.0112<br>arc_easy: acc=0.5383, acc_norm=0.4886, acc_norm_stderr=0.0103, acc_stderr=0.0102<br>winogrande: acc=0.5067, acc_stderr=0.0141<br>boolq: acc=0.5679, acc_stderr=0.0087<br>truthfulqa_mc1: acc=0.2521, acc_stderr=0.0152</details></td><td data-convpe-autofill="ruler"><details><summary>查看逐项指标</summary>niah_multikey_1_1024: 0.1760<br>niah_multikey_1_2048: 0.1660<br>niah_multikey_1_4096: 0.0960<br>niah_multikey_2_1024: 0.0040<br>niah_multikey_2_2048: 0.0000<br>niah_multikey_2_4096: 0.0000<br>niah_multikey_3_1024: 0.0000<br>niah_multikey_3_2048: 0.0000<br>niah_multikey_3_4096: 0.0000<br>niah_multiquery_1024: 0.0140<br>niah_multiquery_2048: 0.0195<br>niah_multiquery_4096: 0.0040<br>niah_multivalue_1024: 0.1080<br>niah_multivalue_2048: 0.1210<br>niah_multivalue_4096: 0.0455<br>niah_single_1_1024: 0.4300<br>niah_single_1_2048: 0.1520<br>niah_single_1_4096: 0.0260<br>niah_single_2_1024: 0.8420<br>niah_single_2_2048: 0.0860<br>niah_single_2_4096: 0.0720<br>niah_single_3_1024: 0.5780<br>niah_single_3_2048: 0.2500<br>niah_single_3_4096: 0.0240<br>ruler_cwe_1024: 0.0034<br>ruler_cwe_2048: 0.0004<br>ruler_cwe_4096: 0.0002<br>ruler_fwe_1024: 0.1153<br>ruler_fwe_2048: 0.0487<br>ruler_fwe_4096: 0.0433<br>ruler_qa_squad_1024: 0.0753<br>ruler_qa_squad_2048: 0.0787<br>ruler_qa_squad_4096: 0.0753<br>ruler_vt_1024: 0.0300<br>ruler_vt_2048: 0.0000<br>ruler_vt_4096: 0.0000</details></td><td data-convpe-autofill="train_bin"><details><summary>查看逐项指标</summary>1024: 20.7108<br>2048: 20.0095<br>4096: 19.6791<br>8192: 19.5183</details></td></tr>
<tr><td>DAPE <a href="#ref-9">[9]</a></td><td>163.512</td><td>16</td><td>183204</td><td>32000</td><td>32.87</td><td>273.71</td><td>59.86</td><td>20.25</td><td>24.41</td><td>41.95</td><td>91.24</td><td>1.33</td><td>2×8</td><td>29300–31300</td><td>108.26</td><td>151.34</td><td data-convpe-autofill="task"><details><summary>查看逐项指标</summary>lambada_openai: perplexity=123.5960, acc=0.2290, acc_stderr=0.0059, perplexity_stderr=5.6160<br>hellaswag: acc=0.3031, acc_norm=0.3414, acc_norm_stderr=0.0047, acc_stderr=0.0046<br>xstorycloze_en: acc=0.5619, acc_stderr=0.0128<br>piqa: acc=0.6376, acc_norm=0.6376, acc_norm_stderr=0.0112, acc_stderr=0.0112<br>arc_easy: acc=0.5337, acc_norm=0.4756, acc_norm_stderr=0.0102, acc_stderr=0.0102<br>winogrande: acc=0.5217, acc_stderr=0.0140<br>boolq: acc=0.5810, acc_stderr=0.0086<br>truthfulqa_mc1: acc=0.2313, acc_stderr=0.0148</details></td><td data-convpe-autofill="ruler"><details><summary>查看逐项指标</summary>niah_multikey_1_1024: 0.2760<br>niah_multikey_1_2048: 0.2820<br>niah_multikey_1_4096: 0.1720<br>niah_multikey_2_1024: 0.0220<br>niah_multikey_2_2048: 0.0040<br>niah_multikey_2_4096: 0.0000<br>niah_multikey_3_1024: 0.0200<br>niah_multikey_3_2048: 0.0000<br>niah_multikey_3_4096: 0.0000<br>niah_multiquery_1024: 0.1340<br>niah_multiquery_2048: 0.0615<br>niah_multiquery_4096: 0.0165<br>niah_multivalue_1024: 0.1540<br>niah_multivalue_2048: 0.1185<br>niah_multivalue_4096: 0.0305<br>niah_single_1_1024: 0.7640<br>niah_single_1_2048: 0.7240<br>niah_single_1_4096: 0.4060<br>niah_single_2_1024: 0.9420<br>niah_single_2_2048: 0.3660<br>niah_single_2_4096: 0.1340<br>niah_single_3_1024: 0.5080<br>niah_single_3_2048: 0.3380<br>niah_single_3_4096: 0.0300<br>ruler_cwe_1024: 0.1642<br>ruler_cwe_2048: 0.0302<br>ruler_cwe_4096: 0.0030<br>ruler_fwe_1024: 0.0260<br>ruler_fwe_2048: 0.0000<br>ruler_fwe_4096: 0.0000<br>ruler_qa_squad_1024: 0.0793<br>ruler_qa_squad_2048: 0.0880<br>ruler_qa_squad_4096: 0.0880<br>ruler_vt_1024: 0.0436<br>ruler_vt_2048: 0.0388<br>ruler_vt_4096: 0.0144</details></td><td data-convpe-autofill="train_bin"><details><summary>查看逐项指标</summary>1024: 19.4721<br>2048: 18.7060<br>4096: 18.3073<br>8192: 18.0852</details></td></tr>
</tbody>
</table>
<!-- LM_RESULTS_END -->

**结论：**

在最大上下文 8192 的 WikiText 评测中，DAPE 的 word-PPL 最低（32.87），ConvPE 为 48.31，RoPE 为 224.20。该协议包含超出训练长度 1024 的上下文，不能将排序直接解释为训练长度内的语言建模能力或其他任务的综合排序。新增方法属于统一骨架上的机制移植，不能将其他论文原始数值直接视为本组结果。

同硬件基准中，ConvPE 每次训练更新较 RoPE 多耗时 4.68%；这是固定预算的执行开销，不能据此推断达到相同精度所需的总训练成本。

## 实验 2：CIFAR-100 图像分类

**目的：**

比较二维位置机制在同一 ViT-Tiny 上的分类性能。

**配置：**

- 训练集：CIFAR-100 官方 train，50,000 张 32×32 图像；使用现有 [cifar-100-python.tar.gz](/cfs/common_users/zhijian.qiao/datasets/ConvPE/eval/other/cifar/cifar-100-python.tar.gz)。增强依次为 ToTensor、padding=4 的随机裁剪、随机水平翻转、RandomErasing、作者给定均值标准差归一化。
- 测试集：CIFAR-100 官方 test 全部 10,000 张，不进行随机训练增强。
- 指标：测试集 top-1 accuracy（%，越高越好）为排序主指标，从低到高排列；同时报告交叉熵（越低越好）。使用第 100 epoch 权重；不依据测试集选择 PE 超参数。
- 初始化与训练范围：seed 0，从头全量训练；九个方法共享骨架初始权重。ViT-Tiny 12 层、宽度 192、3 heads、head dim 64，patch 4×4，8×8 token 网格，无 CLS token，token logits 平均分类。
- 资源与预算：每方法 1 节点 × 8 H20；DDP，GBS 256，100 epochs，19600 updates，梯度累积 1，activation checkpointing 关闭。每 epoch 最后一个不足 256 的 batch 保留，样本数与作者单卡流程一致。
- 优化器：排除 emb/cls 的二维参数用 Muon，LR 0.003、momentum 0.95、weight decay 0.1；其他参数用 AdamW，LR 0.0003、betas=(0.9,0.999)、epsilon=1e-8、weight decay 0.01。
- 调度与精度：按 epoch 的 CosineAnnealingLR，无 warmup；FP16 autocast 与 GradScaler，沿用作者未设置梯度裁剪的流程。梯度溢出时所有 rank 同步降低 scale 并重试同一 batch，两个优化器均不更新，成功更新才计步；前向 loss 非有限则报错。保存第十步、每 epoch 及最终权重。
- 评测配置：原图与水平翻转图分别推理，logits 各占 0.5；完整 10,000 张测试集评测；九种方法共用协议。
- 同硬件训练效率：GBS 256，32×32 图像，FP16、AdamW/Muon 和同批溢出重试，无梯度裁剪。各方法均从 seed 0 初始化，不加载训练权重；预热后优化器状态已建立。使用固定真实 CIFAR-100 样本及增强，每个 rank 独立固定增强 seed，各方法输入 tensor 哈希一致；计时包含溢出重试，重试次数另存原始 JSON。
- 同硬件推理效率：单卡 FP16，batch 1 测延迟、batch 256 测吞吐，各预热 20 次、计时 100 次；沿用原图与水平翻转 logits 平均，吞吐按原始图像计数。使用 seed 0 初始模型测系统性能，不代替最终权重的准确率评测；原有六种方法与冻结实现输出一致，新增三种二维适配定义见控制变量。
- 完整云训练效率：各方法均为 8 卡，耗时取平台 StartTime 至 EndTime，包含初始化、100 轮训练、逐轮验证、checkpoint 保存和退出；不含排队。GPU·小时为运行小时乘 8，不是账单金额；端到端训练原图吞吐按 100×50,000 张计算。来源见 [summary.json](/cfs/common_users/zhijian.qiao/outputs/convpe-reproduction/20261031/cifar100-final-eval-extended/summary.json)。

**控制变量：**

- **二维位置机制**；公共骨架与训练配置保持一致。
  - **二维 RoPE**：每个 64 维 head 分为行/列各 32 维，base=10000。
  - **ConvPE2D**：w3d4，一个 bias head 共享到三个 attention heads。
  - **可学习二维绝对 PE**：8×8×192 位置参数，截断正态初始化 std=0.02。
  - **固定二维 sin-cos**：行/列各 96 维，base=10000，无可训练位置参数。
  - **NoPE**：不加入显式位置编码或位置 bias；共享 patch 投影及 token 平均池化使模型对 patch 排列置换不变。
  - **二维 ALiBi**：采用 [CROMA](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf) 的二维欧氏 patch 距离负偏置，三个 head 使用原 ALiBi 非二次幂斜率规则；双向 attention，无可训练位置参数。
  - **CABLE2D（新适配）**：每 head 从 patch 内容预测非负节点权重，相邻节点的均值作为无向边代价；两种横纵顺序的 L 形路径代价取平均，再乘 query 的 softplus scale 形成负偏置。均匀节点权重对应 Manhattan 距离，不连接展平序列的跨行边界。
  - **DAPE2D（新适配）**：Kerple 使用二维欧氏距离；每个 query 对应的 8×8 key 网格拼接各 head 的 QK 分数和 Kerple bias，经共享权重的横向、纵向两层 1×3 卷积后取平均，隐藏宽度 32、LeakyReLU，结果加回 QK 与 Kerple bias；分数累积和 softmax 使用 FP32，卷积遵循外层 FP16 autocast；双向 attention，不使用因果 mask。
  - **Composite2D（新适配）**：保留 QK + 固定偏置 + query 相对位置项；采用半径 8 的完整二维 key−query 有符号偏移表，固定项按 head 独立、query 表跨 head 共享，query 相对项按 √64 缩放。固定偏置为四维参数，用 AdamW；二维 query 表用 Muon。三种新适配均在本组视觉骨架上定义，不代表对应论文已发表的视觉实现。

<table>
<thead>
<tr><th rowspan="2">方法</th><th rowspan="2">参数 M</th><th rowspan="2">GBS</th><th rowspan="2">训练 Step</th><th rowspan="2">Epoch</th><th colspan="2">最终权重效果</th><th colspan="3">同硬件训练：4 卡 H20</th><th colspan="4">同硬件推理：单卡 H20</th><th colspan="3">完整云训练：8 卡 H20</th></tr>
<tr><th>Accuracy ↑ (%)</th><th>CE ↓</th><th>ms/更新 ↓</th><th>原图/s ↑</th><th>显存 GiB ↓</th><th>B1 ms ↓</th><th>B256 ms ↓</th><th>B256 原图/s ↑</th><th>B256 显存 GiB ↓</th><th>端到端分钟 ↓</th><th>GPU·小时 ↓</th><th>端到端训练原图/s ↑</th></tr>
</thead>
<tbody>
<tr><td>NoPE <a href="#ref-6">[6]</a></td><td>5.363</td><td>256</td><td>19600</td><td>100</td><td>56.03</td><td>2.51</td><td>66.45</td><td>3853</td><td>0.79</td><td>5.42</td><td>12.30</td><td>20818</td><td>0.21</td><td>28.08</td><td>3.74</td><td>2967</td></tr>
<tr><td>二维 ALiBi <a href="#ref-5">[5]</a> <a href="#ref-7">[7]</a></td><td>5.363</td><td>256</td><td>19600</td><td>100</td><td>61.38</td><td>2.14</td><td>66.63</td><td>3842</td><td>0.79</td><td>5.32</td><td>11.67</td><td>21937</td><td>0.21</td><td>27.57</td><td>3.68</td><td>3023</td></tr>
<tr><td>Composite2D <a href="#ref-4">[4]</a></td><td>5.596</td><td>256</td><td>19600</td><td>100</td><td>63.50</td><td>2.02</td><td>78.29</td><td>3270</td><td>0.91</td><td>6.98</td><td>15.84</td><td>16164</td><td>0.26</td><td>32.27</td><td>4.30</td><td>2583</td></tr>
<tr><td>二维 RoPE <a href="#ref-3">[3]</a> <a href="#ref-8">[8]</a></td><td>5.363</td><td>256</td><td>19600</td><td>100</td><td>64.17</td><td>1.96</td><td>78.44</td><td>3264</td><td>0.81</td><td>12.38</td><td>18.90</td><td>13544</td><td>0.21</td><td>37.77</td><td>5.04</td><td>2207</td></tr>
<tr><td>可学习二维绝对 PE <a href="#ref-2">[2]</a></td><td>5.376</td><td>256</td><td>19600</td><td>100</td><td>64.36</td><td>1.89</td><td>63.50</td><td>4032</td><td>0.79</td><td>5.36</td><td>12.22</td><td>20955</td><td>0.21</td><td>27.80</td><td>3.71</td><td>2998</td></tr>
<tr><td>固定二维 sin-cos <a href="#ref-1">[1]</a></td><td>5.363</td><td>256</td><td>19600</td><td>100</td><td>65.07</td><td>1.91</td><td>66.65</td><td>3841</td><td>0.79</td><td>5.51</td><td>12.29</td><td>20826</td><td>0.21</td><td>27.92</td><td>3.72</td><td>2985</td></tr>
<tr><td>CABLE2D <a href="#ref-10">[10]</a></td><td>5.377</td><td>256</td><td>19600</td><td>100</td><td>65.25</td><td>1.82</td><td>98.90</td><td>2588</td><td>0.99</td><td>10.44</td><td>19.91</td><td>12859</td><td>0.24</td><td>41.67</td><td>5.56</td><td>2000</td></tr>
<tr><td>DAPE2D <a href="#ref-9">[9]</a></td><td>5.374</td><td>256</td><td>19600</td><td>100</td><td>66.01</td><td>1.79</td><td>92.74</td><td>2760</td><td>1.72</td><td>10.93</td><td>53.05</td><td>4826</td><td>0.57</td><td>39.93</td><td>5.32</td><td>2087</td></tr>
<tr><td>ConvPE2D <a href="#ref-11">[11]</a></td><td>5.377</td><td>256</td><td>19600</td><td>100</td><td>67.98</td><td>1.69</td><td>82.03</td><td>3121</td><td>0.81</td><td>9.13</td><td>13.62</td><td>18803</td><td>0.22</td><td>36.95</td><td>4.93</td><td>2255</td></tr>
</tbody>
</table>

**结论：**

本次 seed 0、100 epoch 对比中，ConvPE2D 的 top-1 为 67.98%，比二维 RoPE 高 3.81 个百分点；在九个方法中准确率最高、交叉熵最低。新增二维适配中，DAPE2D 为 66.01%、CABLE2D 为 65.25%，分别低于 ConvPE2D 1.97 和 2.73 个百分点。原论文报告 ConvPE2D w3d4 为 69.5%、RoPE 为 62.7%，优势方向一致；本次使用二维 axial RoPE，作者代码使用展平序列的一维 RoPE，且初始化及分布式抽样随机序列不同，因此不能视为严格数值复现，也不能据单个 seed 推断稳定排序。论文未报告可学习二维绝对 PE 和固定二维 sin-cos 的同组结果。

同硬件基准中，ConvPE2D 的训练更新耗时为二维 RoPE 的 1.046 倍；其初始权重下的性能基准与最终准确率、完整云训练耗时分别对应不同测量条件。

## 参考文献

按首次公开年份升序排列；同年按首次公开时间排列。未注明年份的作者手稿列于最后。表内引用表示机制来源；CABLE2D、DAPE2D、Composite2D 是本实验新增二维适配，不能将对应引用解释为已发表的同名二维算法。

<a id="ref-1"></a>

[1] **2017**. Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 正弦位置编码的来源；表中二维行列拼接为本实验实现。

<a id="ref-2"></a>

[2] **2020**. Alexey Dosovitskiy et al. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). 2020 预印本，ICLR 2021；ViT 与可学习位置嵌入的参考，本实验使用 8×8 参数网格。

<a id="ref-3"></a>

[3] **2021**. Jianlin Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). 一维 RoPE。

<a id="ref-4"></a>

[4] **2021**. Tyler A. Chang, Yifan Xu, Weijian Xu, Zhuowen Tu. [Convolutions and Self-Attention: Re-interpreting Relative Positions in Pre-trained Language Models](https://aclanthology.org/2021.acl-long.333/). ACL-IJCNLP 2021；Composite 的机制来源。

<a id="ref-5"></a>

[5] **2021**. Ofir Press, Noah A. Smith, Mike Lewis. [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409). 2021 预印本，ICLR 2022；ALiBi。

<a id="ref-6"></a>

[6] **2023**. Amirhossein Kazemnejad et al. [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466). NeurIPS 2023；NoPE 的语言模型研究参考，视觉 NoPE 为本实验无位置机制对照。

<a id="ref-7"></a>

[7] **2023**. Anthony Fuller, Koreen Millard, James R. Green. [CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders](https://arxiv.org/abs/2311.00566). NeurIPS 2023；二维 ALiBi 的欧氏距离定义。

<a id="ref-8"></a>

[8] **2024**. Byeongho Heo, Song Park, Dongyoon Han, Sangdoo Yun. [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298). ECCV 2024；二维 axial RoPE 的参考。

<a id="ref-9"></a>

[9] **2024**. Chuanyang Zheng et al. [DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](https://arxiv.org/abs/2410.04798). 本实验 DAPE V2–Kerple 的机制来源。

<a id="ref-10"></a>

[10] **2025**. Ali Veisi, Hamidreza Amirzadeh, Amir M. Mansourian. [Context-aware Biases for Length Extrapolation](https://aclanthology.org/2025.emnlp-main.1545/). EMNLP 2025；CABLE。

<a id="ref-11"></a>

[11] **未注明年份**. Wang Zehao. [ConvPE: Convolutional Position Embedding Improves Transformers](/cfs/common_users/zhijian.qiao/code/convpe/paper/convpe.pdf). 作者仓库随附论文；convpe.tex 的日期字段为空，不从文件修改时间推断发表年份。
