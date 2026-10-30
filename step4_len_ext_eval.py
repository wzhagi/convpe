import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch

from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

from trainer_model import LanguageModel
import nltk

nltk.data.path.insert(0, "./data/nltk")
torch._dynamo.config.recompile_limit = 64

CHECKPOINTS = [
    "results_pretrain/base_rope_42",
    "results_pretrain/base_nope_w128d16r0h1g1_42",
]

TASKS = {
    # "niah_single_1",
    # "niah_single_2",
    # "niah_single_3",
    # "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    # "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    # "ruler_qa_squad",
    "ruler_qa_hotpot",
}

SEQ_LENGTHS = [1024, 2048]


def main(name, tasks):
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


if __name__ == "__main__":
    for task in TASKS:
        main(task, [task])
