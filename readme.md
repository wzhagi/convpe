# ConvPE
卷积位置编码（ConvPE）显著改善了 Transformer 的性能与样本效率。相比基线模型，ConvPE 仅增加 2.64% 的参数和 2.70% 的前向推理时间，却将训练损失降低 2.82%。达到基线最终训练损失所需的 token 仅为基线的 68.7%，且在本文评测的多数任务上优于基线。

ConvPE (Convolutional Position Embedding) substantially improves both the performance and sample efficiency of Transformer models. At the cost of only 2.64\% more parameters and a 2.70\% increase in forward-pass time, it achieves a 2.82\% reduction in training loss and substantial improvements on most evaluation benchmarks. It also requires only 68.7\% of the training tokens needed by the baseline to reach the baseline's final training loss.

| ![convpe](/paper/FIGURES/convpe.jpeg) | ![score_bias](/paper/FIGURES/score_bias.jpeg) |
|---|---|

## Pretrain curves on the first 3B tokens of FineWeb-Edu-10BT

![](/paper/FIGURES/pretrain.png)

## Pretraining Evaluations

| Dataset | Metric | RoPE baseline | ConvPE w128d16 | Random |
|:---:|:---:|:---:|:---:|:---:|
| wikitext | perplexity | 46.5636 | **42.0498** |-|
| lambada | perplexity | 118.1028 | **88.4866** |-|
| lambada | acc | 0.2507 | **0.2604** |0|
| hellaswag | acc | 0.2894 | **0.2949** |0.25|
| hellaswag | acc_norm | 0.3148 | **0.3272** |0.25|
| xstorycloze_en | acc | 0.5460 | **0.5526** |0.5|
| piqa | acc | 0.6251 | **0.6338** |0.5|
| piqa | acc_norm | 0.6208 | **0.6235**|0.5|
| arc_easy | acc | 0.5114 | **0.5421** |0.25|
| arc_easy | acc_norm | 0.4487 | **0.4701** |0.25|
| winogrande | acc | 0.5091 | **0.5193** |0.5|
| boolq | acc | 0.5676 | **0.5893** |0.5|
| truthfulqa_mc1 | acc | 0.2399 | 0.2375 |**0.25**|

## Context Extension Evaluations
#### FineWeb-Edu-10BT Validation Loss of 1M Tokens
|Context| RoPE baseline | ConvPE w128d16 |
|:---:|:---:|:---:|
|1K|3.1247|**3.0312**|
|2K|3.1684|**3.0021**|
|4K|4.2690|**3.0278**|
|8K|5.6443|**3.3575**|

#### RULER Tasks
| Task | Context | RoPE baseline | ConvPE w128d16 |
|:---:|:---:|:---:|:---:|
| niah_single_1   | 1K | 0.614 | **0.890** |
|                 | 2K | 0.004 | **0.722** |
|                 | 4K | 0.000 | **0.314** |
| niah_single_2   | 1K | 0.612 | **0.664** |
|                 | 2K | 0.000 | **0.038** |
| niah_single_3   | 1K | 0.090 | **0.180** |
|                 | 2K | 0.000 | **0.014** |
| niah_multikey_1 | 1K | 0.198 | **0.200** |
|                 | 2K | 0.118 | **0.194** |
|                 | 4K | 0.000 | 0.002 |
| niah_multikey_2 | 1K | 0.002 | 0.000 |
| niah_multikey_3 | 1K | 0.000 | 0.008 |
| niah_multiquery | 1K | 0.104 | **0.186**|
|                 | 2K | 0.029 | **0.179** |
|                 | 4K | 0.000 | 0.000 |
| niah_multivalue | 1K | 0.149 | **0.187** |
|                 | 2K | 0.127 | **0.186** |
|                 | 4K | 0.000 | 0.000 |
| ruler_vt        | 1K | **0.032** | 0.008 |
|                 | 2K | 0.000 | **0.052** |
|                 | 4K | 0.000 | **0.107** |
| ruler_cwe       | 1K | 0.002 | 0.002 |
| ruler_fwe       | 1K | 0.001 | **0.012** |
| ruler_qa_squad  | 1K | 0.148 | **0.191** |
|                 | 2K | 0.078 | **0.091** |
|                 | 4K | 0.017 | **0.052** |


## ConvPE2D validation accuracy on CIFAR-10, CIFAR-100.
|model|CIFAR-10 valid acc|CIFAR-100 valid acc|
|:-:|:-:|:-:|
|RoPE baseline 5.346M|0.910|0.627|
|ConvPE2D w3d4 5.359M|0.931|**0.695**|
|ConvPE2D w3d8 5.372M|**0.933**|0.683|
|ConvPE2D w5d4 5.402M|0.929|0.682|
|ConvNet 4.335M | 0.945 | 0.737 |
## Requirements

### Tilelang kernels 
#### Make sure your torch cuda version and tilelang cuda version is the same with python venv cuda version or system cuda version
    apt install python3-dev
    pip install -r requirements_tilelang.txt

### Pytorch Flex kernels (Alternative)
    apt install python3-dev
    pip install -r requirements_flex.txt

#### Edit transformer_convpe.py import

    from attn_flex import convpe_attention as attention

## Demo Training  

    python3 train_wikitext.py

#### Run train_wikitext_plot.ipynb to see results.

    python3 train_cifar.py

## Pretraining

### 1. Prepare Data

#### Download  Data
First Download fineweb-edu-10BT first 4 parquets into data/fineweb-edu-10BT
https://www.modelscope.cn/datasets/AI-ModelScope/fineweb-edu/tree/master/sample/10BT

    ls data/fineweb-edu-10BT/
    000_00000.parquet  001_00000.parquet  002_00000.parquet  003_00000.parquet

#### Build data bin

    python3 step1_prepare_parquet.py

### 2. Pre-training

#### Train full dataset (Optional)

    set step2_pretrain.py TrainingArguments max_steps=2000 to -1

#### Start training
##### Use attn_tilelang_pretrain.py kernel for higher throughput (Optional)
    python3 step2_pretrain.py

#### Run step2_plot.ipynb to see results.

### 3. Pre-training evaluation

    python3 step3_pretrain_eval.py

#### Run step3_plot.ipynb to see results.

### 4. Context Length Extension

    python3 step4_len_ext_eval.py

#### Run step4_plot.ipynb to see results.

    python3 step5_len_ext_loss_eval.py


