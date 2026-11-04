# prepare_parquet.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import glob
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer

PARQUET_DIR = "data/fineweb-edu-10BT"

# 显式指定要处理的 parquet 文件（顺序即处理顺序）。
# - 只写文件名：会在 PARQUET_DIR 下找，例如 "000_00000.parquet"
# - 写完整/相对路径：原样使用，例如 "data/other/xxx.parquet"
PARQUET_FILES = [
    "013_00000.parquet",
]

OUTPUT_PREFIX = "data/fineweb-13parquet"
OUTPUT_EXT = ".bin"

# 按 token 数切分，依次使用列表中的大小。
# 列表用完后，剩下的所有 token 会合并写入最后一个分片文件（不再继续切）。
# - None：单文件
# - [150*2**20, 10*2**20]：先写 ~150M tokens，再写 ~10M tokens，
#   之后剩余全部写进下一个文件。
# 注意：这里的单位是 token 数，GPT2 token 用 uint16 存（2 字节/token）。
#       若要按 MB 指定，请写 [150*2**20 // 2, 10*2**20 // 2]。
SHARD_SIZES = [150*2**20, 10*2**20]

WRITE_CHUNK = 20_000_000
READ_BATCH = 2000
NUM_PROC = 8
CHUNKSIZE = 200

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def tokenize_text(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def resolve_parquet_files(parquet_dir, names):
    """把用户指定的文件名/路径解析成实际存在的路径列表（保持顺序、去重检查）。"""
    if not names:
        raise RuntimeError("PARQUET_FILES is empty")
    paths = []
    seen = set()
    for name in names:
        p = name if (os.path.isabs(name) or os.path.dirname(name)) else os.path.join(parquet_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"parquet not found: {p}")
        ap = os.path.abspath(p)
        if ap in seen:
            raise ValueError(f"duplicate parquet in PARQUET_FILES: {p}")
        seen.add(ap)
        paths.append(p)
    return paths


class ShardedBinWriter:
    """按 token 数把 uint16 token 流写入一个或多个 .bin 文件。

    shard_sizes:
      - None：单文件 {prefix}{ext}
      - list[int]：按顺序使用每个大小；列表用尽后，剩余 token 全部写入
        最后一个分片文件（不再继续切分）。
    """

    def __init__(self, prefix, ext=".bin", shard_sizes=None):
        self.prefix = prefix
        self.ext = ext
        if shard_sizes is None:
            self.sizes = None
        else:
            if not isinstance(shard_sizes, (list, tuple)) or len(shard_sizes) == 0:
                raise ValueError("shard_sizes must be a non-empty list or None")
            self.sizes = [int(s) for s in shard_sizes]
            if any(s <= 0 for s in self.sizes):
                raise ValueError("all shard sizes must be > 0")

        self.shard_idx = 0
        self.cur_tokens = 0
        self.total_tokens = 0
        self.files = []
        self.shard_token_counts = []
        self.f = None
        self.final_path = None
        self.tmp_path = None
        self._open_new()

    def _cur_limit(self):
        if self.sizes is None:
            return None
        if self.shard_idx < len(self.sizes):
            return self.sizes[self.shard_idx]
        # 列表用完后：不再限制大小，剩余 token 全部写入当前文件
        return None

    def _base_path(self):
        if self.sizes is None:
            return f"{self.prefix}{self.ext}"
        return f"{self.prefix}-part{self.shard_idx:03d}{self.ext}"

    def _open_new(self):
        self.final_path = self._base_path()
        self.tmp_path = self.final_path + ".tmp"
        self.f = open(self.tmp_path, "wb")
        self.cur_tokens = 0

    def _close_current(self):
        self.f.close()
        os.replace(self.tmp_path, self.final_path)
        self.files.append(self.final_path)
        self.shard_token_counts.append(self.cur_tokens)
        self.shard_idx += 1

    def write(self, ids):
        arr = np.asarray(ids, dtype=np.uint16)
        pos, n = 0, arr.size
        while pos < n:
            limit = self._cur_limit()
            if limit is None:
                take = n - pos
            else:
                take = min(limit - self.cur_tokens, n - pos)
            if take <= 0:
                self._close_current()
                self._open_new()
                continue
            self.f.write(arr[pos:pos + take].tobytes())
            self.cur_tokens += take
            self.total_tokens += take
            pos += take
            if limit is not None and self.cur_tokens >= limit:
                self._close_current()
                self._open_new()

    def close(self):
        if self.f is None:
            return
        if self.cur_tokens > 0:
            self._close_current()
        else:
            self.f.close()
            try:
                os.remove(self.tmp_path)
            except FileNotFoundError:
                pass
        self.f = None


def main():
    files = resolve_parquet_files(PARQUET_DIR, PARQUET_FILES)
    print(f"parquet files (in order): {[os.path.basename(f) for f in files]}")
    print(f"num_proc={NUM_PROC}, read_batch={READ_BATCH}, "
          f"shard_sizes={SHARD_SIZES}, write_chunk={WRITE_CHUNK}")

    writer = ShardedBinWriter(OUTPUT_PREFIX, OUTPUT_EXT, shard_sizes=SHARD_SIZES)

    total = 0
    buf = []
    buf_len = 0

    def flush_buf():
        nonlocal buf, buf_len
        if not buf:
            return
        writer.write(np.concatenate(buf))
        buf, buf_len = [], 0

    with Pool(NUM_PROC) as pool:
        for path in files:
            name = os.path.basename(path)
            ds = load_dataset("parquet", data_files=path, split="train")
            n_docs = len(ds)
            print(f"\n=== {name}: {n_docs:,} docs ===")

            shard_tokens_cnt = 0
            pbar = tqdm(
                total=n_docs, desc=name, unit="doc",
                dynamic_ncols=True, mininterval=0.5,
            )

            for start in range(0, n_docs, READ_BATCH):
                texts = ds[start:start + READ_BATCH]["text"]
                for ids in pool.imap(tokenize_text, texts, chunksize=CHUNKSIZE):
                    buf.append(np.array(ids, dtype=np.uint16))
                    buf_len += len(ids)
                    shard_tokens_cnt += len(ids)
                    total += len(ids)
                    if buf_len >= WRITE_CHUNK:
                        flush_buf()

                pbar.update(len(texts))
                pbar.set_postfix({
                    "part": f"{writer.shard_idx:03d}",
                    "this_doc_tokens": f"{shard_tokens_cnt/1e9:.3f}B",
                    "total": f"{total/1e9:.3f}B",
                })

            pbar.close()
            print(f"  {name} done: {shard_tokens_cnt:,} tokens, "
                  f"cumulative={total/1e9:.4f}B")

        flush_buf()

    writer.close()

    print(f"\nDone. total tokens={total:,} ({total/1e9:.4f}B), "
          f"{len(writer.files)} shard(s):")
    for p, cnt in zip(writer.files, writer.shard_token_counts):
        size_mb = os.path.getsize(p) / 1e6
        print(f"  {p}  {cnt:,} tokens  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()