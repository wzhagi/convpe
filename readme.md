# ConvPE
ConvPE 卷积位置编码显著提升了Transformer的参数效率和训练性能。0.51%的额外参数和1.95%的额外计算，换取0.14 (4.32%)的预训练loss下降，从3.24降到3.1。

ConvPE (Convolutional Position Embedding) substantially improves both the parameter efficiency and training performance of Transformers. At the cost of only 0.51% additional parameters and 1.95% additional computation, it achieves a 0.14 (4.32%) reduction in pretraining loss, decreasing from 3.24 to 3.1.

![](/paper/FIGURES/attn_score_bias.png)

## Pretrain curves on the first 3B tokens of FineWeb-Edu-10BT

![](/paper/FIGURES/pretrain.png)

## Pretraining Evaluations

| Dataset | Metric | RoPE 124.7M | ConvPE 125.3M | Random |
|:---:|:---:|:---:|:---:|:---:|
| wikitext | perplexity | 55.6150 | **46.5696(-9.0454)** |-|
| lambada | perplexity | 354.7769 | **139.2030(-215.5739)** |-|
| lambada | acc | 0.1760 | **0.2457(+0.0697)** |0|
| hellaswag | acc | 0.2798 | **0.2944(+0.0145)** |0.25|
| hellaswag | acc_norm | 0.2913 | **0.3141(+0.0228)** |0.25|
| xstorycloze_en | acc | 0.5367 | **0.5606(+0.0238)** |0.5|
| piqa | acc | 0.6251 | **0.6409(+0.0158)** |0.5|
| piqa | acc_norm | 0.6099 | **0.6349(+0.0250)** |0.5|
| arc_easy | acc | 0.4958 | **0.4996(+0.0038)** |0.25|
| arc_easy | acc_norm | 0.4474 | **0.4646(+0.0173)** |0.25|
| winogrande | acc | **0.5083** | 0.4996(-0.0087) |0.5|
| boolq | acc | 0.5994 |**0.6006(+0.0012)** |0.5|
| truthfulqa_mc1 | acc | 0.2485 | 0.2338(-0.0147) |**0.25**|

## Training curves on WikiText-103 for 10 epochs

![](/paper/FIGURES/main_figure.png)

## ConvPE2D effectiveness verification in CIFAR-10, CIFAR-100.
|model|CIFAR-10 val acc|CIFAR-100 val acc|
|:-:|:-:|:-:|
|RoPE baseline 5.3M|0.910|0.627|
|ConvPE 3x3 5.4M|**0.915**|0.647|
|ConvPE 5x5 5.5M|0.908|**0.648**|

## Requirements
    apt install python3-dev
    pip install -r requirements.txt

## Demo Training

#### Delete the "results" folder or its sub-folders. (Optional)

Sub-folders and files in the "results" folder may cause training to be automatically skipped.

#### Start training    

    python3 train_wikitext.py

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

set step2_pretrain.py TrainingArguments max_steps=1000 to -1

#### Start training
    python3 step2_pretrain.py

run step2_plot.ipynb to see results.

### 3. Pre-training evaluation

    python3 step3_pretrain_eval.py

run step3_plot.ipynb to see results.

## Tilelang kernels

### 1. Cuda Version
Make sure your torch cuda version and tilelang cuda version is the same with python venv cuda version or system cuda version

#### Create a new venv and install tilelang requirements

    pip install -r requirements_tilelang.txt