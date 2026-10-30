import json
import math
import numpy as np
import torch
from trainer_model import LanguageModel

# ============ 配置 ============
CHECKPOINTS = [
    "results_pretrain/base_rope_42",
    "results_pretrain/base_nope_w128d16r0h1g1_42",
]
BIN_PATH = "data/fineweb-4parquet.bin"
WINDOW_SIZES = [1024, 2048, 4096, 8192]
NUM_TEST_TOKENS = 1024 * 1024            # 直接取 1M 个 token，每个长度下至少有 128 个块
BATCH_SIZE = 1                           # 批大小，显存不够就降到 1~2


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_ppl(model, token_stream, window_size, batch_size=1, device="cuda"):
    model.eval()
    num_blocks = len(token_stream) // window_size
    if num_blocks == 0:
        return 0.0, 0.0

    truncated = token_stream[: num_blocks * window_size]
    blocks = truncated.view(num_blocks, window_size)

    total_loss = 0.0
    total_samples = 0

    for i in range(0, num_blocks, batch_size):
        batch = blocks[i : i + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batch, labels=batch)
            loss = outputs.loss
        total_loss += loss.float().item() * batch.size(0)
        total_samples += batch.size(0)

    avg_loss = total_loss / total_samples
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {NUM_TEST_TOKENS} tokens from {BIN_PATH}...")
    data = np.memmap(BIN_PATH, dtype=np.uint16, mode="r")
    test_data = data[:NUM_TEST_TOKENS].astype(np.int64)
    token_stream = torch.from_numpy(test_data)
    print(f"Loaded {len(token_stream)} tokens for evaluation.\n")

    results = {}

    for checkpoint in CHECKPOINTS:
        print(f"===== Evaluating {checkpoint} =====")
        model = LanguageModel.from_pretrained(checkpoint).to(device)
        model.eval()
        results[checkpoint] = {}

        for w_size in WINDOW_SIZES:
            if len(token_stream) < w_size:
                print(f"Skipping window size {w_size}: token stream too short.")
                continue

            print(f"--> Window Size: {w_size}  ({len(token_stream) // w_size} blocks)")
            try:
                loss, ppl = eval_ppl(
                    model, token_stream, window_size=w_size, batch_size=BATCH_SIZE, device=device
                )
                results[checkpoint][str(w_size)] = {
                    "loss": round(loss, 4),
                    "ppl": round(ppl, 4),
                }
                print(f"    Loss = {loss:.4f} | PPL = {ppl:.4f}")
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at window_size={w_size}, skipping.")
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()
        print()

    out_path = "results_pretrain/finweb_ppl_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"All results saved -> {out_path}")
    print("========== Final Summary ==========")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()