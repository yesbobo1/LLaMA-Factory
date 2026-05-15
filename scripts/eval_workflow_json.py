import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import Dataset
from transformers import GenerationConfig

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.parser import get_dataset_list
from llamafactory.extras import logging
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


logger = logging.get_logger(__name__)


@dataclass
class Example:
    system: str
    tools: str
    prompt_messages: list[dict[str, str]]
    label_text: str


def _extract_first_json(text: str) -> tuple[Optional[Any], Optional[str]]:
    if text is None:
        return None, None

    stripped = text.strip()
    if not stripped:
        return None, None

    decoder = json.JSONDecoder()
    candidates = []
    for ch in ("{", "["):
        pos = stripped.find(ch)
        if pos >= 0:
            candidates.append(pos)
    for pos in sorted(set(candidates)):
        try:
            obj, end = decoder.raw_decode(stripped[pos:])
            raw = stripped[pos : pos + end]
            return obj, raw
        except Exception:
            pass

    brace_positions = [i for i, c in enumerate(stripped) if c in "{["]
    for pos in brace_positions[:64]:
        try:
            obj, end = decoder.raw_decode(stripped[pos:])
            raw = stripped[pos : pos + end]
            return obj, raw
        except Exception:
            continue

    return None, None


def _is_workflow_like(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "nodes" not in obj:
        return False
    if "connections" not in obj:
        return False
    if not isinstance(obj.get("nodes"), list):
        return False
    if not isinstance(obj.get("connections"), dict):
        return False
    return True


def _build_eval_examples(dataset_path: str, max_samples: Optional[int]) -> list[dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    if max_samples is not None:
        data = data[: max_samples]
    return data


def _split_train_val(examples: list[dict[str, Any]], val_size: float, seed: int) -> list[dict[str, Any]]:
    ds = Dataset.from_list(examples)
    if val_size <= 0:
        return list(ds)
    test_size = int(val_size) if val_size > 1 else val_size
    split = ds.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return list(split["test"])


def _to_llamafactory_examples(
    raw_examples: list[dict[str, Any]],
    dataset: str,
    dataset_dir: str,
) -> list[Example]:
    dataset_attr = get_dataset_list([dataset], dataset_dir)[0]

    out: list[Example] = []
    for ex in raw_examples:
        messages = ex.get(dataset_attr.messages, None)
        if not isinstance(messages, list) or len(messages) == 0:
            continue

        system_text = ""
        if (
            dataset_attr.system_tag
            and len(messages) > 0
            and isinstance(messages[0], dict)
            and messages[0].get(dataset_attr.role_tag) == dataset_attr.system_tag
        ):
            system_text = str(messages[0].get(dataset_attr.content_tag, "") or "")
            messages = messages[1:]
        elif dataset_attr.system:
            system_text = str(ex.get(dataset_attr.system, "") or "")

        tools_text = str(ex.get(dataset_attr.tools, "") or "") if dataset_attr.tools else ""

        if len(messages) < 2:
            continue

        last = messages[-1]
        if not isinstance(last, dict) or last.get(dataset_attr.role_tag) != dataset_attr.assistant_tag:
            continue

        label_text = str(last.get(dataset_attr.content_tag, "") or "")
        prompt_messages_raw = messages[:-1]
        if len(prompt_messages_raw) == 0:
            continue

        tag_to_role = {
            dataset_attr.user_tag: "user",
            dataset_attr.assistant_tag: "assistant",
            dataset_attr.observation_tag: "observation",
            dataset_attr.function_tag: "function",
        }

        prompt_messages: list[dict[str, str]] = []
        for m in prompt_messages_raw:
            if not isinstance(m, dict):
                prompt_messages = []
                break
            role_tag = m.get(dataset_attr.role_tag, None)
            if role_tag not in tag_to_role:
                prompt_messages = []
                break
            content = str(m.get(dataset_attr.content_tag, "") or "")
            prompt_messages.append({"role": tag_to_role[role_tag], "content": content})

        if not prompt_messages:
            continue
        if prompt_messages[-1]["role"] != "user":
            continue

        out.append(Example(system=system_text, tools=tools_text, prompt_messages=prompt_messages, label_text=label_text))

    return out


def _flatten(list_of_lists: list[list[int]]) -> list[int]:
    out: list[int] = []
    for xs in list_of_lists:
        out.extend(xs)
    return out


def _generate_one(
    model,
    tokenizer,
    template_obj,
    ex: Example,
    generation_config: GenerationConfig,
    max_input_len: int,
) -> str:
    encoded_parts = template_obj._encode(tokenizer, ex.prompt_messages, ex.system, ex.tools)
    input_ids = _flatten(encoded_parts)
    if max_input_len is not None and len(input_ids) > max_input_len:
        input_ids = input_ids[-max_input_len:]
    input_tensor = torch.tensor([input_ids], device=model.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_tensor,
            generation_config=generation_config,
            eos_token_id=template_obj.get_stop_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = output[0][len(input_ids) :].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ansa_sft2")
    parser.add_argument("--dataset_dir", default="data")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--adapter_name_or_path", default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--cutoff_len", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--prediction_mode", choices=["generate", "reference"], default="generate")
    parser.add_argument("--out_jsonl", default="workflow_eval_predictions.jsonl")
    parser.add_argument("--out_summary", default="workflow_eval_summary.json")
    args = parser.parse_args()

    dataset_attr = get_dataset_list([args.dataset], args.dataset_dir)[0]
    dataset_path = os.path.join(args.dataset_dir, dataset_attr.dataset_name)
    raw = _build_eval_examples(dataset_path=dataset_path, max_samples=args.max_samples)
    val_raw = _split_train_val(raw, val_size=args.val_size, seed=args.seed)
    eval_examples = _to_llamafactory_examples(val_raw, dataset=args.dataset, dataset_dir=args.dataset_dir)

    if args.max_eval_samples is not None:
        rnd = random.Random(args.seed)
        rnd.shuffle(eval_examples)
        eval_examples = eval_examples[: args.max_eval_samples]

    model = None
    tokenizer = None
    template_obj = None
    generation_config = None

    if args.prediction_mode == "generate":
        if args.model_name_or_path is None:
            raise ValueError("--model_name_or_path is required when prediction_mode=generate")

        infer_args = dict(
            model_name_or_path=args.model_name_or_path,
            adapter_name_or_path=args.adapter_name_or_path,
            trust_remote_code=args.trust_remote_code,
            infer_backend="huggingface",
            template=args.template,
            cutoff_len=args.cutoff_len,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        model_args, data_args, finetuning_args, generating_args = get_infer_args(infer_args)

        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
        model = load_model(tokenizer, model_args, finetuning_args)
        generation_config = GenerationConfig(
            do_sample=bool(generating_args.do_sample),
            temperature=float(generating_args.temperature),
            top_p=float(generating_args.top_p),
            top_k=int(generating_args.top_k),
            repetition_penalty=float(generating_args.repetition_penalty),
            max_new_tokens=int(generating_args.max_new_tokens),
        )

    started = time.time()
    total = 0
    json_parse_ok = 0
    workflow_like_ok = 0
    both_parse_equal = 0
    label_parse_ok = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(eval_examples):
            if args.prediction_mode == "reference":
                pred_text = ex.label_text
            else:
                pred_text = _generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    template_obj=template_obj,
                    ex=ex,
                    generation_config=generation_config,
                    max_input_len=args.cutoff_len,
                )

            pred_obj, pred_raw = _extract_first_json(pred_text)
            label_obj, _ = _extract_first_json(ex.label_text)

            pred_ok = pred_obj is not None
            label_ok = label_obj is not None
            wf_ok = _is_workflow_like(pred_obj) if pred_ok else False
            eq_ok = (pred_ok and label_ok and pred_obj == label_obj)

            total += 1
            if pred_ok:
                json_parse_ok += 1
            if wf_ok:
                workflow_like_ok += 1
            if label_ok:
                label_parse_ok += 1
            if eq_ok:
                both_parse_equal += 1

            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "system": ex.system,
                        "user": ex.prompt_messages[-1]["content"],
                        "label": ex.label_text,
                        "predict": pred_text,
                        "predict_json_raw": pred_raw,
                        "scores": {
                            "predict_json_parse_ok": pred_ok,
                            "predict_workflow_like": wf_ok,
                            "label_json_parse_ok": label_ok,
                            "json_equal": eq_ok,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    elapsed = time.time() - started
    summary = {
        "dataset": args.dataset,
        "dataset_path": dataset_path,
        "val_size": args.val_size,
        "seed": args.seed,
        "total_eval": total,
        "predict_json_parse_rate": (json_parse_ok / total) if total else 0.0,
        "predict_workflow_like_rate": (workflow_like_ok / total) if total else 0.0,
        "label_json_parse_rate": (label_parse_ok / total) if total else 0.0,
        "json_exact_match_rate": (both_parse_equal / total) if total else 0.0,
        "seconds": elapsed,
        "prediction_mode": args.prediction_mode,
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    logger.info_rank0(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
