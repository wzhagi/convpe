import os, copy, re, math

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trainer_model_tilelang import LanguageModel, LanguageModelConfig
from muon import AdamMuon, is_muon_param

torch.backends.cudnn.benchmark = True
VOCAB_SIZE = 50304  # pad 50257 to 50304
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# ============ Main ============
def main():
    max_seq_len = 1024
    base_nope = LanguageModelConfig(
        name="base_nope",
        seed=42,
        num_layers=12,
        dim_model=768,
        num_heads=12,
        dim_head=128,
        dim_rope=0,
        convpe_win=0,
        convpe_dim=0,
        convpe_heads=0,
        convpe_rank=0,
        convpe_groups=1,
        convpe_share=False,
        vocab_size=VOCAB_SIZE,
        max_seq_len=max_seq_len,
    )

    base_rope = copy.deepcopy(base_nope)
    base_rope.name = "base_rope"
    base_rope.dim_rope = 64

    def convpe_config(prototype, name: str):
        config = copy.deepcopy(prototype)
        config.name = f"{config.name}_{name}"
        m = re.fullmatch(r"w(\d+)d(\d+)r(\d+)h(\d+)g(\d+)(s)?", name)
        w, d, r, h, g, s = m.groups()
        config.convpe_win = int(w)
        config.convpe_dim = int(d)
        config.convpe_rank = int(r)
        config.convpe_heads = int(h)
        config.convpe_groups = int(g)
        config.convpe_share = s is not None
        return config

    configs = []


    # configs.append(convpe_config(base_nope, "w128d4r0h1g1")) #2.60it/s
    # params: 768*((128*25+768*8)*12+50257) + (768*4+4*128*128)*12 = 125.5M

    configs.append(convpe_config(base_nope, "w128d16r0h1g1")) #2.55it/s
    # params: 768*((128*25+768*8)*12+50257) + (768*16+16*128*128)*12 = 128.0M

    configs.append(base_rope) #2.7it/s
    # params: 768*((128*25+768*8)*12+50257)=124.7M

    # ============ Dataset ============
    bin_path = "data/fineweb-4parquet.bin"
    total_blocks = len(np.memmap(bin_path, dtype=np.uint16, mode="r")) // max_seq_len
    train_dataset = MemmapDataset(bin_path, max_seq_len, 0, total_blocks)
    print(f"total_blocks={total_blocks}")
    for config in configs:
        run_train(config, train_dataset)


class MemmapDataset(Dataset):
    def __init__(self, bin_path, block_size, start_block=0, end_block=None):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        total = len(self.data) // block_size
        if end_block is None:
            end_block = total
        self.start_block = start_block
        self.end_block = end_block
        self.n_blocks = end_block - start_block
        assert self.n_blocks > 0, "empty dataset slice"

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        block_idx = self.start_block + idx
        start = block_idx * self.block_size
        chunk = self.data[start : start + self.block_size].astype(np.int64)
        return {"input_ids": torch.from_numpy(chunk)}


# ============ Train ============
def run_train(config, train_dataset):
    print(f"\n=== training {config.name} ===")
    model = LanguageModel(config)
    print(model)
    # print("tie weights:",model.tok_emb.weight is model.lm_head.weight)
    # for n,p in model.named_parameters():
    #     print("is muon param:",is_muon_param(n,p),n)

    output_dir = f"results_pretrain/{config.name}_{config.seed}"

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=-1,
        lr_scheduler_type="warmup_stable_decay",
        warmup_steps=500,
        lr_scheduler_kwargs={
            "num_decay_steps": 40000,
            "decay_type": "linear",
        },
        max_grad_norm=2.0,
        logging_steps=100,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        report_to="none",
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        accelerator_config={
            "non_blocking": True,
        },
        remove_unused_columns=False,
        bf16=True,
        torch_compile=True,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        optimizers=(
            AdamMuon(model.named_parameters(), muon_lr=6e-3, adam_lr=6e-4),
            None,
        ),
    )
    trainer.model_accepts_loss_kwargs = False

    trainer.train(resume_from_checkpoint=get_latest_checkpoint(output_dir))
    trainer.save_model(output_dir)
    print(f"saved model to {output_dir}")


def get_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if re.fullmatch(r"checkpoint-\d+", d)]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, ckpts[-1])


if __name__ == "__main__":
    main()
