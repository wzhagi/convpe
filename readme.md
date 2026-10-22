# ConvPE
ConvPE 卷积位置编码显著提升了Transformer的参数效率和训练性能。3.2%的额外参数和大致等比例的计算，换取14.6%的训练loss下降，从2.60降到2.22。

ConvPE (Convolutional Position Embedding) substantially improves both the parameter efficiency and training performance of Transformers. At the cost of only 3.2% additional parameters and roughly proportional computation, it achieves a 14.6% reduction in training loss, decreasing from 2.60 to 2.22.

![](/paper/FIGURES/attn_score_bias.png)

## Pretrain curves on the first 3B tokens of FineWeb-Edu-10BT

![](/paper/FIGURES/pretrain.png)

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

## Configration (Optional)
### WikiText
Default to train tiny model on WikiText-103 for 1 epoch

    # edit trainer_train.py main function
    DEFAULTS = {
        "output_dir": "results",
        "epochs": 1,
        "batch_size": 4,
        "lr": 3e-4,
        "block_size": 1024,
        "num_workers": 4,
    }
    configs = []
    configs.append(...)
### CIFAR
Default to train tiny model on CIFAR-10,CIFAR-100 for 100 epochs

    # edit train_cifar.py main part
    if __name__ == "__main__":
        configs = []
        configs.append(...)

## Training

### Delete the "results" folder or its sub-folders. (Optional)

My results for comparison are in the "wzh_results" folder.

Sub-folders and files in the "results" folder may cause training to be automatically skipped.

### Start training    

    python3 trainer_train.py

    python3 train_cifar.py