import os, re

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
from trainer_model import LanguageModel
from muon import AdamMuon

torch.backends.cudnn.benchmark = True
VOCAB_SIZE = 50304  # pad 50257 to 50304
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# ============ Dataset（与原代码一致，block_size 由新的 max_seq_len 决定）============
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


# ============ 上下文扩展微调 ============
def run_lenext(pretrain_dir,rope_ext=""):
    print(f"\n=== lenext fine-tune from {pretrain_dir} ===")

    # 直接从原文件夹读取训练好的模型；config 中的 max_seq_len 已被人工加长
    model = LanguageModel.from_pretrained(pretrain_dir)
    if rope_ext=="yarn":
        model.to_yarn_()
    if rope_ext=="convpe":
        model.to_yarn_()
        model.to_convpe_()
    config = model.config
    max_seq_len = config.max_seq_len
    print(f"loaded config: name={config.name}, seed={config.seed}, max_seq_len={max_seq_len}")
    print(model)

    output_dir = f"results_lenext/{config.name}{rope_ext}_{config.seed}"

    bin_path = "data/fineweb-13parquet-part000.bin"
    total_blocks = len(np.memmap(bin_path, dtype=np.uint16, mode="r")) // max_seq_len
    train_dataset = MemmapDataset(bin_path, max_seq_len, 0, total_blocks)
    print(f"total_blocks={total_blocks}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        max_grad_norm=4.0,
        logging_steps=100,
        save_steps=500,
        save_total_limit=1,
        report_to="none",
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        accelerator_config={"non_blocking": True},
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
        optimizers=(AdamMuon(model.named_parameters(),adam_lr=3e-5,muon_lr=3e-4), None),
    )
    trainer.train(get_latest_checkpoint(output_dir))
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

# ============ Main ============
def main():
    pretrain_dirs = [
        "results_pretrain/base_rope_42",
        "results_pretrain/base_nope_w128d16r0h1g1_42",
    ]
    for d in pretrain_dirs:
        if not os.path.isdir(d):
            print(f"[skip] not found: {d}")
            continue
        if d == "results_pretrain/base_rope_42":
            run_lenext(d,"yarn")
            # run_lenext(d,"convpe")
        else:
            run_lenext(d)
            


if __name__ == "__main__":
    main()