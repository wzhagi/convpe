# ConvPE
ConvPE substantially improves Transformers parameter efficiency and training efficiency by Convolutional Position Embedding. ConvPE achieves substantial reductions in validation loss compared with the Rotary Position Embedding (RoPE) baseline, while adding only a small number of backbone parameters (+0.15% params, -3.83% loss; +1.87% params, -8.03% loss). As model scale increases, the best configuration achieves a 10.3% loss reduction with only a 4.34% increase in backbone parameters, demonstrating high performance and parameter efficiency. ConvPE uses convolution to generate attention score biases, making it compatible with various Transformer architectures and high-dimensional modalities.

![](/paper/FIGURES/attn_score_bias.png)
## Training curves on WikiText-103 for 10 epochs

![](/main_figure.png)

## ConvPE2D effectiveness verification in CIFAR-10, CIFAR-100.
|model|CIFAR-10 val acc|CIFAR-100 val acc|
|:-:|:-:|:-:|
|RoPE baseline 5.3M|0.906|0.630|
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