import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json, zipfile, urllib.request, nltk
import torch
from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
from trainer_model import LanguageModel

torch._dynamo.config.recompile_limit = 64

CHECKPOINTS = [
    "results_pretrain/base_rope_42",
    "results_pretrain/base_nope_w128d16r0h1g1_42",
]

TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    # "ruler_qa_hotpot",
]

SEQ_LENGTHS = [
    1024,
    2048,
    # 4096,
]


def main(name, tasks):
    print(f"===== evaluating {name} =====")
    results = {}
    for checkpoint in CHECKPOINTS:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(tokenizer)
        model = LanguageModel.from_pretrained(checkpoint).to("cuda")
        print(model)
        model.eval()
        model.compile()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = evaluator.simple_evaluate(
                model=HFLM(
                    pretrained=model, tokenizer=tokenizer, batch_size=1, max_length=5120
                ),
                tasks=tasks,
                metadata={
                    "max_seq_lengths": SEQ_LENGTHS,
                    "tokenizer": checkpoint,
                },
            )
        results[checkpoint] = result["results"]
        out_path = os.path.join(checkpoint, f"{name}_eval.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                result["results"],
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        print(f"saved -> {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


def download_punkt_tab(force: bool = False) -> None:
    """自包含：配置路径 → 检查 → 下载 → 解压 → 拍平 → 验证，全部在一个函数内完成。"""
    import os
    import ssl
    import shutil
    import zipfile
    import urllib.request
    from pathlib import Path

    import nltk

    # ---------- 1. 配置 NLTK 数据目录（绝对路径，写入搜索路径） ----------
    nltk_data_dir = Path("./data/nltk").resolve()
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data_dir))

    tokenizers_dir = nltk_data_dir / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    punkt_tab_dir = tokenizers_dir / "punkt_tab"

    # ---------- 2. 严格校验函数 ----------
    def _ok() -> bool:
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            return False
        try:
            nltk.sent_tokenize("Hello world. This is a test.")
            return True
        except Exception:
            return False

    if not force and _ok():
        print("punkt_tab 已就绪，跳过下载")
        return

    # ---------- 3. 候选下载源 ----------
    urls = [
        "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip",
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip",
    ]

    # ---------- 4. 下载（跳过 SSL 验证，兼容证书链不全的镜像） ----------
    zip_path = tokenizers_dir / "punkt_tab.zip"
    ctx = ssl._create_unverified_context()
    last_err = None

    for url in urls:
        print(f"downloading: {url}")
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=60) as resp, \
                    open(zip_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"downloaded -> {zip_path} ({zip_path.stat().st_size} bytes)")
            break
        except Exception as e:
            last_err = e
            print(f"[warn] 下载失败: {url} -> {e}")
            if zip_path.exists():
                zip_path.unlink()
    else:
        raise RuntimeError(f"所有下载源均失败，最后错误: {last_err}")

    # ---------- 5. 解压 ----------
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            top = sorted({n.split("/")[0] for n in z.namelist()})
            print("zip 顶层条目:", top)
            z.extractall(tokenizers_dir)
    finally:
        if zip_path.exists():
            zip_path.unlink()

    # ---------- 6. 拍平嵌套目录 ----------
    inner = punkt_tab_dir / "punkt_tab"
    if inner.is_dir():
        for x in inner.iterdir():
            shutil.move(str(x), str(punkt_tab_dir / x.name))
        inner.rmdir()
        print(f"[fix] 拍平嵌套: {inner} -> {punkt_tab_dir}")

    # ---------- 7. 最终验证 ----------
    if not _ok():
        print("解压后目录内容:")
        for p in sorted(punkt_tab_dir.rglob("*"))[:30]:
            print("  ", p)
        raise RuntimeError(
            "punkt_tab 解压后仍无法加载。"
        )
    print("punkt_tab 安装并验证成功")


if __name__ == "__main__":
    download_punkt_tab()
    for task in TASKS:
        main(task, [task])

