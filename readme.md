# ConvPE
ConvPE substantially improves Transformers parameter efficiency and training efficiency by Convolutional Position Embedding.
![](/paper/FIGURES/attn_score_bias.png)
## Training curve on WikiText-103 for 10 epochs

![](/main_figure.png)

## Abstract
Convolutional Position Embedding (ConvPE) achieves substantial reductions in validation loss compared with the Rotary Position Embedding (RoPE) baseline, while adding only a small number of backbone parameters (+0.15% params, -3.83% loss; +1.87% params, -8.03% loss). As model scale increases, the best configuration achieves a 10.3% loss reduction with only a 4.34% increase in backbone parameters, demonstrating high performance and parameter efficiency. ConvPE uses convolution to generate attention score biases, making it compatible with various Transformer architectures and high-dimensional modalities.

## Requirements
    apt install python3-dev
    pip install -r requirements.txt

## Configration (Optional)
    edit trainer_train.py main function
    edit cifar_convpe.py file

## Training
    ./trainer_train.py
    ./cifar_train.py