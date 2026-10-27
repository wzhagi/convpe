import torch, time, os
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from convpe2d import *

print(torch.__version__)
device = "cuda"
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class ViT(nn.Module):
    def __init__(
        self, dim_model, num_heads, num_layers, out_features, patch_shape, max_seq_len
    ):
        super().__init__()
        self.emb = nn.Conv2d(3, dim_model, patch_shape, patch_shape)
        self.rope = RotaryPositionalEmbedding(64, max_seq_len)
        self.model = VisionTransformer(
            num_layers,
            dim_model,
            num_heads,
            rope=self.rope,
            max_seq_len=max_seq_len,
        )
        self.cls = nn.Linear(dim_model, out_features)

    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.emb.forward(x).flatten(2).mT.contiguous()
        x = self.model(x)
        x = self.cls(x)
        x = x.mean(-2)
        if not self.training:
            orig, flipped = x.split(x.shape[0] // 2, dim=0)
            x = 0.5 * orig + 0.5 * flipped
        return x


class ViT_ConvPE(nn.Module):
    def __init__(
        self,
        dim_model,
        num_heads,
        num_layers,
        out_features,
        patch_shape,
        max_seq_len,
        convpe_win,
        convpe_dim,
        convpe_heads,
    ):
        super().__init__()
        self.emb = nn.Conv2d(3, dim_model, patch_shape, patch_shape)
        self.model = VisionTransformer(
            num_layers,
            dim_model,
            num_heads,
            convpe_win=convpe_win,
            convpe_dim=convpe_dim,
            convpe_heads=convpe_heads,
            max_seq_len=max_seq_len,
        )
        self.cls = nn.Linear(dim_model, out_features)

    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.emb.forward(x).flatten(2).mT.contiguous()
        x = self.model(x, 8, 8)
        x = self.cls(x)
        x = x.mean(-2)
        if not self.training:
            orig, flipped = x.split(x.shape[0] // 2, dim=0)
            x = 0.5 * orig + 0.5 * flipped
        return x


class ViT_Tiny(ViT):
    def __init__(self, out_features, patch_shape, max_seq_len, **kwargs):
        super().__init__(192, 3, 12, out_features, patch_shape, max_seq_len)


class ViT_Tiny_ConvPEw3(ViT_ConvPE):
    def __init__(self, out_features, patch_shape, max_seq_len, **kwargs):
        super().__init__(192, 3, 12, out_features, patch_shape, max_seq_len, 3, 4, 1)


class ViT_Tiny_ConvPEw5(ViT_ConvPE):
    def __init__(self, out_features, patch_shape, max_seq_len, **kwargs):
        super().__init__(192, 3, 12, out_features, patch_shape, max_seq_len, 5, 16, 1)


class Conv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding, stride, bias):
        super().__init__()
        self.in_channels = channels_in
        self.out_channels = channels_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.linear = nn.Linear(
            channels_in * kernel_size * kernel_size, channels_out, bias=bias
        )

    def forward(self, x):
        w = self.linear.weight.view(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        return F.conv2d(
            x,
            w,
            self.linear.bias,
            stride=self.stride,
            padding=self.padding,
        )


class ConvGroup(nn.Sequential):
    def __init__(self, channels_in, channels_out):
        super().__init__(
            Conv2d(channels_in, channels_out, 3, 1, 1, False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channels_out),
            nn.GELU(),
            Conv2d(channels_out, channels_out, 3, 1, 1, False),
            nn.BatchNorm2d(channels_out),
            nn.GELU(),
        )


class ConvNet(nn.Module):
    def __init__(self, out_features, **kwargs):
        super().__init__()
        self.channels = [3, 24, 64, 256, 512]
        self.pad = nn.ReflectionPad2d(2)
        self.emb = nn.Conv2d(3, 24, 2, bias=False)
        self.net = nn.Sequential(
            nn.GELU(),
            ConvGroup(24, self.channels[2]),
            ConvGroup(self.channels[2], self.channels[3]),
            ConvGroup(self.channels[3], self.channels[4]),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
        )
        self.cls = nn.Linear(self.channels[4], out_features)

    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.pad(x)
        x = self.emb(x)
        x = self.net(x)
        x = self.cls(x)
        if not self.training:
            orig, flipped = x.split(x.shape[0] // 2, dim=0)
            x = 0.5 * orig + 0.5 * flipped
        return x


def cifar10():
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomErasing(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), [0.2470, 0.2435, 0.2616]
                ),
            ]
        ),
    )

    val_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), [0.2470, 0.2435, 0.2616]
                ),
            ],
        ),
    )

    return {"train": train_data, "val": val_data, "num_class": 10}


def cifar100():
    train_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomErasing(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        ),
    )

    val_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ],
        ),
    )
    return {"train": train_data, "val": val_data, "num_class": 100}


def is_muon_parameter(name: str, param) -> bool:
    if param.ndim != 2 or "cls" in name or "emb" in name:
        return False
    return True


def muon_parameters(model):
    return [p for n, p in model.named_parameters() if is_muon_parameter(n, p)]


def adam_parameters(model):
    return [p for n, p in model.named_parameters() if not is_muon_parameter(n, p)]


def train(epochs, batch_size, learn_rate, model, seed, dataset, config, **kwargs):
    torch.manual_seed(seed)
    log_path = f"./results/{dataset.__name__}_{model.__name__}_muon/{seed}.csv"
    dataset = dataset()
    model = model(
        out_features=dataset["num_class"], patch_shape=(4, 4), max_seq_len=64, **config
    ).to(device)
    print(model)
    print(
        f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, mode="a", encoding="utf-8")
    print(config)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        dataset["val"],
        batch_size,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )
    loss_func = nn.functional.cross_entropy
    adamw = torch.optim.AdamW(adam_parameters(model), learn_rate)
    scheduler_adamw = torch.optim.lr_scheduler.CosineAnnealingLR(adamw, epochs)
    muon = torch.optim.Muon(muon_parameters(model), learn_rate * 10)
    scheduler_muon = torch.optim.lr_scheduler.CosineAnnealingLR(muon, epochs)
    optimizers = [adamw, muon]
    train_total_time = 0
    val_total_time = 0
    print(
        "epoch, train_acc,   val_acc,train_loss,  val_loss,train_time,  val_time,total_time"
    )
    print(
        "epoch, train_acc,   val_acc,train_loss,  val_loss,train_time,  val_time,total_time",
        file=log_file,
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        train_acc, train_loss, train_time = train_epoch(
            train_dataloader, model, loss_func, optimizers
        )
        scheduler_adamw.step()
        scheduler_muon.step()
        val_acc, val_loss, val_time = val_epoch(val_dataloader, model, loss_func)
        train_total_time += train_time
        val_total_time += val_time

        print(
            f"{epoch:>5},{train_acc:>10.3f},{val_acc:>10.3f},{train_loss:>10f},{val_loss:>10f},{train_time:>10.1f},{val_time:>10.1f},{train_total_time + val_total_time:>10.1f}"
        )
        print(
            f"{epoch:>5},{train_acc:>10.3f},{val_acc:>10.3f},{train_loss:>10f},{val_loss:>10f},{train_time:>10.1f},{val_time:>10.1f},{train_total_time + val_total_time:>10.1f}",
            file=log_file,
            flush=True,
        )
    log_file.close()


@torch.compile
def forward(model, source):
    return model(source)


@torch.compile
def backward(loss):
    loss.backward()


def train_epoch(dataloader, model, loss_fn, optimizers):
    total_time = time.time()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    avg_loss, acc = 0, 0
    scaler = torch.amp.GradScaler()
    for i, (source, target) in enumerate(dataloader):
        source: Tensor = source.to(device, non_blocking=True)
        target: Tensor = target.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            pred = forward(model, source)
            loss = loss_fn(pred, target)
        backward(scaler.scale(loss))
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()
        model.zero_grad()

        avg_loss += loss.item()
        acc += (pred.argmax(1) == target).type(torch.float).sum().item()
    avg_loss /= num_batches
    acc /= size
    total_time = time.time() - total_time
    return (acc, avg_loss, total_time)


def val_epoch(dataloader, model, loss_fn):
    total_time = time.time()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, acc = 0, 0
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for source, target in dataloader:
            source: Tensor = source.to(device, non_blocking=True)
            target: Tensor = target.to(device, non_blocking=True)
            pred = forward(model, source)
            avg_loss += loss_fn(pred, target).item()
            acc += (pred.argmax(1) == target).type(torch.float).sum().item()
    avg_loss /= num_batches
    acc /= size
    total_time = time.time() - total_time
    return (acc, avg_loss, total_time)


if __name__ == "__main__":
    configs = []
    for seed in range(1):
        for dataset in [
            # cifar10,
            cifar100,
        ]:
            for model in [
                ViT_Tiny,
                # ViT_Tiny_ConvPEw3,
                ViT_Tiny_ConvPEw5,
                # ConvNet
            ]:
                configs.append(
                    {
                        "model": model,
                        "epochs": 100,
                        "batch_size": 256,
                        "learn_rate": 3e-4,
                        "dataset": dataset,
                        "seed": seed,
                    }
                )
    for config in configs:
        train(**config, config=config)
