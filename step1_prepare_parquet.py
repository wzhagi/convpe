# prepare_parquet.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import glob
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer

# First Download fineweb-edu-10BT first 4 parquets into data/fineweb-edu-10BT
# ls data/fineweb-edu-10BT/
# 000_00000.parquet  001_00000.parquet  002_00000.parquet  003_00000.parquet

PARQUET_DIR = "data/fineweb-edu-10BT"
NUM_PARQUETS = 4
OUTPUT_BIN = "data/fineweb-4parquet.bin"

TMP_BIN = OUTPUT_BIN + ".tmp"
WRITE_CHUNK = 20_000_000
READ_BATCH = 2000  # 每批读多少文档进内存
NUM_PROC = 8  # tokenize 进程数，按 CPU 核数调
CHUNKSIZE = 200  # 每个 worker 每次领多少文档

# 全局 tokenizer：fork 前初始化，子进程直接继承
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def tokenize_text(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def main():
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))[:NUM_PARQUETS]
    if not files:
        raise RuntimeError(f"No parquet in {PARQUET_DIR}")
    print(f"parquet files (fixed order): {[os.path.basename(f) for f in files]}")
    print(f"num_proc={NUM_PROC}, read_batch={READ_BATCH}")

    total = 0
    buf = []
    buf_len = 0

    with open(TMP_BIN, "wb") as f, Pool(NUM_PROC) as pool:
        for path in files:
            name = os.path.basename(path)
            ds = load_dataset("parquet", data_files=path, split="train")
            n_docs = len(ds)
            print(f"\n=== {name}: {n_docs:,} docs ===")

            shard_tokens = 0
            pbar = tqdm(
                total=n_docs, desc=name, unit="doc", dynamic_ncols=True, mininterval=0.5
            )

            for start in range(0, n_docs, READ_BATCH):
                texts = ds[start : start + READ_BATCH]["text"]
                # imap 保序：结果顺序与 texts 完全一致
                for ids in pool.imap(tokenize_text, texts, chunksize=CHUNKSIZE):
                    buf.append(np.array(ids, dtype=np.uint16))
                    buf_len += len(ids)
                    shard_tokens += len(ids)
                    total += len(ids)

                    if buf_len >= WRITE_CHUNK:
                        f.write(np.concatenate(buf).tobytes())
                        buf, buf_len = [], 0

                pbar.update(len(texts))
                pbar.set_postfix(
                    {
                        "tokens": f"{shard_tokens/1e9:.3f}B",
                        "total": f"{total/1e9:.3f}B",
                    }
                )

            pbar.close()
            print(
                f"  {name} done: {shard_tokens:,} tokens, "
                f"cumulative={total/1e9:.4f}B"
            )

        if buf:
            f.write(np.concatenate(buf).tobytes())

    os.rename(TMP_BIN, OUTPUT_BIN)
    size_gb = os.path.getsize(OUTPUT_BIN) / 1e9
    print(f"\nDone. tokens={total:,} ({total/1e9:.4f}B), size={size_gb:.2f} GB")


if __name__ == "__main__":
    main()
