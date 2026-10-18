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
    edit trainer_train.py main function
    edit cifar_convpe.py file

## Training
    ./trainer_train.py
    ./cifar_train.py

## FP8 Training
    install transformer engine or use it's dorker image (recomanded)
<https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/pytorch/->

    edit transformer_convpe.py

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8BlockScaling

    Annotate CausualTransformer forward with te.autocast

    @te.autocast(recipe=Float8BlockScaling())
    def forward(self, x: Tensor) -> Tensor:
    
    change MQA wq, wo from nn.Linear to te.Linear
    change FFN to te.LayerNormMLP