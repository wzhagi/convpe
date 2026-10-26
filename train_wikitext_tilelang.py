import os, copy, re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse, torch, math
from transformers import (
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets
from trainer_model_tilelang import LanguageModel, LanguageModelConfig

torch.backends.cudnn.benchmark = True
# ============ 数据集配置 ============
TRAIN_DATASETS = [
    # {
    #     "repo": "Salesforce/wikitext",
    #     "name": "wikitext-2-raw-v1",
    #     "split": "train",
    #     "field": "text",
    #     "display": "WikiText-2",
    #     "max_docs": None,
    # },
    {
        "repo": "Salesforce/wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "train",
        "field": "text",
        "display": "WikiText-103",
        "max_docs": None,
    },
]

TEST_DATASETS = [
    {
        "repo": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "field": "text",
        "display": "WikiText-2",
        "max_docs": None,
    },
]


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def main(cfg: dict | None = None):
    DEFAULTS = {
        "output_dir": "results",
        "epochs": 1,
        "batch_size": 4,
        "lr": 3e-4,
        "block_size": 1024,
        "num_workers": 2,
    }
    args = argparse.Namespace(**{**DEFAULTS, **(cfg or {})})
    configs = []
    eval_dirs = []
    # eval_dirs.append("results/tiny_rope_42")

    tiny_nope = LanguageModelConfig(
        name="tiny_nope",
        seed=42,
        num_layers=12,
        dim_model=192,
        num_heads=3,
        dim_head=64,
        dim_rope=0,
        convpe_win=0,
        convpe_dim=0,
        convpe_heads=1,
        convpe_rank=0,
        convpe_groups=1,
        convpe_share=False,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.block_size,
    )
    tiny_rope = copy.deepcopy(tiny_nope)
    tiny_rope.name = "tiny_rope"
    tiny_rope.dim_rope = 64

    base_nope = copy.deepcopy(tiny_nope)
    base_nope.name = "base_nope"
    base_nope.dim_model = 768
    base_nope.num_heads = 12
    base_nope.dim_head = 128

    base_rope = copy.deepcopy(base_nope)
    base_rope.name = "base_rope"
    base_rope.dim_rope = 64

    def convpe_config(prototype, name:str):
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

    # configs.append(tiny_rope)
    # configs.append(convpe_config(tiny_nope, "w128d6r0h3g1"))
    # configs.append(convpe_config(tiny_rope, "w128d6r0h3g1"))
    # configs.append(convpe_config(tiny_nope, "w128d6r0h1g1"))
    # configs.append(convpe_config(tiny_nope, "w128d6r0h3g3"))
    # configs.append(convpe_config(tiny_nope, "w128d12r0h3g1"))
    # configs.append(convpe_config(tiny_nope, "w64d6r0h3g1"))
    # configs.append(convpe_config(tiny_nope, "w128d6r128h3g1"))
    # configs.append(convpe_config(tiny_rope, "w128d6r128h3g1"))

    # configs.append(base_rope)  # 11it/s

    # configs.append(convpe_config(base_nope, "w128d4r0h1g1")) #10.6it/s
    # configs.append(convpe_config(base_nope, "w128d8r0h1g1")) #10.5it/s
    # configs.append(convpe_config(base_nope, "w128d16r0h1g1")) #10.4it/s
    configs.append(convpe_config(base_nope, "w128d4r0h4g1s")) #10.4it/s

    # configs.append(convpe_config(base_nope, "w128d4r128h4g1"))
    # configs.append(convpe_config(base_nope, "w128d8r128h4g1"))
    # configs.append(convpe_config(base_nope, "w128d16r0h4g4s"))


    print("========== configs to be trained ==========")
    for config in configs:
        print(config.name)
    print("========== start training ==========")
    for config in configs:
        output_dir = run_train(args, config)
        eval_dirs.append(output_dir)
    print("========== start evaluation ==========")
    for eval_dir in eval_dirs:
        try:
            run_eval(args, eval_dir)
        except Exception:
            pass


# ============ 与训练/评测一致的预处理 ============
def build_text_dataset(cfg, block_size):
    ds = load_dataset(cfg["repo"], cfg.get("name"), split=cfg["split"])
    if cfg.get("max_docs") is not None:
        n = min(cfg["max_docs"], len(ds))
        ds = ds.select(range(n))
    field = cfg["field"]

    def tokenize_fn(examples):
        # 不要 attention_mask，避免与补 eos 后的 input_ids 长度不一致
        out = tokenizer(examples[field], return_attention_mask=False)
        eos = tokenizer.eos_token_id
        for ids in out["input_ids"]:
            ids.append(eos)
        return out

    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {cfg['display']}",
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        return {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }

    ds = ds.map(group_texts, batched=True, desc=f"Packing {cfg['display']}")
    return ds


def build_train_dataset(block_size):
    parts = []
    for cfg in TRAIN_DATASETS:
        ds = build_text_dataset(cfg, block_size)
        print(f"[train:{cfg['display']}] blocks={len(ds)}")
        if len(ds) > 0:
            parts.append(ds)
    if not parts:
        raise RuntimeError("All training datasets are empty.")
    return concatenate_datasets(parts)


def run_train(args, config):
    model = LanguageModel(config)
    print(model)
    output_dir = f"{args.output_dir}/{config.name}_{config.seed}"
    print(f"Output dir: {output_dir}")
    train_dataset = build_train_dataset(args.block_size)
    print(f"Total train blocks: {len(train_dataset)}")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        dataloader_drop_last=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=100,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=args.num_workers,
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
    )
    checkpoint = get_latest_checkpoint(output_dir)
    if checkpoint:
        print(f"Resuming from {checkpoint}")
    else:
        print("No checkpoint found, training from scratch")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(output_dir)
    print(f"Saved model to {output_dir}")
    return output_dir


def run_eval(args, checkpoint_path):
    print(f"Loading model from: {checkpoint_path}")
    model = LanguageModel.from_pretrained(checkpoint_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_args = TrainingArguments(
        output_dir=checkpoint_path,
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        do_train=False,
        do_eval=True,
        bf16=True,
        torch_compile=True,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    lines = []
    for cfg in TEST_DATASETS:
        ds = build_text_dataset(cfg, args.block_size)
        n_blocks = len(ds)
        print(f"[{cfg['display']}] blocks={n_blocks}")
        if n_blocks == 0:
            print(f"[{cfg['display']}] skipped (empty).")
            continue
        metrics = trainer.evaluate(eval_dataset=ds)
        loss = metrics["eval_loss"]
        ppl = math.exp(loss)
        line = f"[{cfg['display']}] loss={loss:.4f} " f"ppl={ppl:.2f} blocks={n_blocks}"
        print(line)
        lines.append(line)
    print("\n==== Summary ====")
    out_path = os.path.join(checkpoint_path, "eval_results.txt")
    with open(out_path, "w") as f:
        for line in lines:
            print(line)
            f.write(line + "\n")
    print(f"Saved eval results to {out_path}")


def get_checkpoint_dirs(root):
    if not os.path.isdir(root):
        return []
    checkpoints = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and re.fullmatch(r"checkpoint-\d+", name):
            checkpoints.append(path)
    checkpoints.sort(key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))
    return checkpoints


def get_latest_checkpoint(output_dir):
    checkpoints = get_checkpoint_dirs(output_dir)
    checkpoints.sort(key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))
    return checkpoints[-1] if checkpoints else None


if __name__ == "__main__":
    main()
