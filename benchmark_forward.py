import os, copy, re, argparse

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from trainer_model_tilelang import LanguageModel, LanguageModelConfig

torch.backends.cudnn.benchmark = True
VOCAB_SIZE = 50304  # pad 50257 to 50304


def build_configs():
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
    configs.append(base_rope)
    configs.append(convpe_config(base_nope, "w128d16r0h1g1"))
    return configs


# ============ 工具 ============
def _get_logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============ 单个 config 的 bench ============
@torch.no_grad()
def bench_config(config, batch_size, seq_len, warmup, iters):
    device = torch.device("cuda")
    model = LanguageModel(config).to(device)
    model.eval()
    n_params = count_params(model)
    model = torch.compile(model)
    input_ids = torch.randint(
        0, VOCAB_SIZE, (batch_size, seq_len), device=device, dtype=torch.long
    )
    # warmup（含首次 compile）
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids)
        logits = _get_logits(out)
    end.record()
    torch.cuda.synchronize()

    per_iter_ms = start.elapsed_time(end) / iters
    tokens_per_iter = batch_size * seq_len
    tokens_per_sec = tokens_per_iter / (per_iter_ms / 1000.0)
    it_per_sec = 1000.0 / per_iter_ms

    result = {
        "name": config.name,
        "params_M": n_params / 1e6,
        "logits_shape": tuple(logits.shape),
        "ms": per_iter_ms,
        "it_s": it_per_sec,
        "tok_s": tokens_per_sec,
    }

    del model, out, logits, input_ids
    torch.cuda.empty_cache()
    return result


# ============ main ============
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=1000)
    a = p.parse_args()

    configs = build_configs()

    print(
        f"batch_size={a.batch_size} seq_len={a.seq_len} "
        f"warmup={a.warmup} iters={a.iters}"
    )
    print(f"configs: {[c.name for c in configs]}\n")

    results = []
    for config in configs:
        print(f"=== {config.name} ===")
        r = bench_config(config, a.batch_size, a.seq_len, a.warmup, a.iters)
        print(f"  params:      {r['params_M']:.2f} M")
        print(f"  logits:      {r['logits_shape']}")
        print(f"  per-iter:    {r['ms']:.2f} ms")
        print(f"  it/s:        {r['it_s']:.2f}")
        print(f"  throughput:  {r['tok_s']:.1f} tokens/s\n")
        results.append(r)

    if len(results) > 1:
        print("=" * 78)
        print(f"{'name':<28}{'params(M)':>12}{'ms/it':>10}{'it/s':>10}{'tok/s':>14}")
        print("-" * 78)
        for r in results:
            print(f"{r['name']:<28}{r['params_M']:>12.2f}"
                  f"{r['ms']:>10.2f}{r['it_s']:>10.2f}"
                  f"{r['tok_s']:>14.1f}")
        print("=" * 78)

        # 以第一个为 baseline 的相对参数量 / 相对时间
        if len(results) >= 2:
            base = results[0]
            print(f"\ncompare with baseline（{base['name']}）:")
            for r in results[1:]:
                d_params = (r["params_M"] - base["params_M"]) / base["params_M"] * 100.0
                d_time = (r["ms"] - base["ms"]) / base["ms"] * 100.0
                print(f"  {r['name']:<28} "
                      f"Δparams {d_params:+.2f}%   "
                      f"Δtime {d_time:+.2f}%")


if __name__ == "__main__":
    main()

# batch_size=4 seq_len=1024 warmup=1000 iters=1000
# configs: ['base_rope', 'base_nope_w128d16r0h1g1']

# === base_rope ===
#   params:      124.84 M
#   logits:      (4, 1024, 50304)
#   per-iter:    22.53 ms
#   it/s:        44.39
#   throughput:  181803.1 tokens/s

# === base_nope_w128d16r0h1g1 ===
#   params:      128.13 M
#   logits:      (4, 1024, 50304)
#   per-iter:    23.14 ms
#   it/s:        43.22
#   throughput:  177020.9 tokens/s

# ==============================================================================
# name                           params(M)     ms/it      it/s         tok/s
# ------------------------------------------------------------------------------
# base_rope                         124.84     22.53     44.39      181803.1
# base_nope_w128d16r0h1g1           128.13     23.14     43.22      177020.9
# ==============================================================================

# compare with baseline（base_rope）:
#   base_nope_w128d16r0h1g1      Δparams +2.64%   Δtime +2.70%