import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
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
    "wikitext",
    "lambada_openai",
    "hellaswag",
    "xstorycloze_en",
    "piqa",
    "arc_easy",
    "winogrande",
    "boolq",
    "truthfulqa_mc1",
]


def main():
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
                model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4),
                tasks=TASKS,
            )
        results[checkpoint] = result["results"]
        out_path = os.path.join(checkpoint, "pretrain_eval.txt")
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


if __name__ == "__main__":
    main()
