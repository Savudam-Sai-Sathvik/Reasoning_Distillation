from __future__ import annotations
import os
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["VLLM_PREFER_TRITON_OPS"] = "0"
# os.environ["TRITON_INTERPRET"] = "1"


import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer


from utils import load_vllm_llm, prompt_vllm, build_vllm_prompt
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# LANGUAGE_MAP = {
#     "en": "English", "hi": "Hindi", "bn": "Bengali", 
#     "kn": "Kannada", "ta": "Tamil"
# }
LANGUAGE_MAP = {
    "en": "English", "english": "English", "English": "English",
    "hi": "Hindi",   "hindi": "Hindi",     "Hindi": "Hindi",
    "bn": "Bengali", "bengali": "Bengali", "Bengali": "Bengali",
    "kn": "Kannada", "kannada": "Kannada", "Kannada": "Kannada",
    "ta": "Tamil",   "tamil": "Tamil"      
}

LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
LANGUAGE_LABELS = {
    "english": "English",
    "hindi": "Hindi",
    "bengali": "Bengali",
    "kannada": "Kannada",
    "tamil": "Tamil",
}
ANSWER_TAG_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
LAST_LINE_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _options_to_text(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"({letters[idx]}) {choice}" for idx, choice in enumerate(options)
    )

# def generate_and_parse(student_base_model, tokenizer, messages,temp= 0,top_p =1, max_new_tokens=2048, use_tqdm =True, adapter_path=None):
#     # outputs = prompt_vllm(
#     #     model,
#     #     tokenizer,
#     #     messages,   # already a batch
#     #     max_new_tokens=2048
#     #     )
#     prompts = [build_vllm_prompt(tokenizer, messages) for messages in messages]
#     sampling_params = SamplingParams(
#         temperature= temp,
#         top_p=top_p,
#         max_tokens=max_new_tokens,
#     )
#     # outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)

#     if adapter_path:
#         outputs = student_base_model.generate(
#             prompts,
#             sampling_params=sampling_params,
#             lora_request=LoRARequest("adapter", 1, adapter_path),
#             use_tqdm=use_tqdm
#         )
#     else:
#         outputs = student_base_model.generate(
#             prompts,
#             sampling_params=sampling_params,
#             use_tqdm=use_tqdm
#         )
#     parsed_outputs = []

#     for output in outputs:
#         pattern = r"####\s*ANSWER:\s*"
#         split_output = re.split(pattern, output, maxsplit=1)

#         if len(split_output) == 2:
#             reasoning = split_output[0].strip()
#             answer_text = split_output[1]

#             match = re.search(r"\(?([A-J])\)?", answer_text)
#             final_answer = match.group(1) if match else ""
#         else:
#             reasoning = output.strip()
#             final_answer = ""
#         parsed_outputs.append({
#             "reasoning": reasoning,
#             "final_answer": final_answer,
#             "raw_generation": output
#         })

#     return parsed_outputs
def generate_and_parse(model, tokenizer, messages,adapter_path=None):
    outputs = prompt_vllm(
        model,
        tokenizer,
        messages,   # already a batch
        max_new_tokens=2048
        )
    # sampling_params = SamplingParams(max_tokens=2048)

    # outputs = student.generate(
    #         messages,
    #         sampling_params=sampling_params,
    #         lora_request=LoRARequest(
    #             lora_name="adapter",
    #             lora_int_id=1,
    #             lora_path=adapter_path,  # ← adapter path goes here
    #         )
    #     )

    parsed_outputs = []
    
    # pattern = r"\nFinal answer:*"
    for output in outputs:
        pattern = r"####\s*ANSWER:\s*"
        split_output = re.split(pattern, output, maxsplit=1)

        if len(split_output) == 2:
            reasoning = split_output[0].strip()
            answer_text = split_output[1]

            match = re.search(r"\(?([A-J])\)?", answer_text)
            final_answer = match.group(1) if match else ""
        else:
            reasoning = output.strip()
            final_answer = ""
        parsed_outputs.append({
            "reasoning": reasoning,
            "final_answer": final_answer,
            "raw_generation": output
        })

    return parsed_outputs


def get_system_prompt(language: str) -> str:

    REASONING = (
        "Reasoning:\n"
        "Identify what the question is asking.\n"
        "For each option, refer to its actual wording and evaluate: "
        "'What would this mean if true? Does it hold under scrutiny?' "
        "Eliminate with evidence, not assumption.\n\n"
    )

    VERDICT = (
        "Verdict:\n"
        "What is the precise definition of the key concept this question tests? "
        "Did you apply it correctly? Are there options you dismissed too quickly? "
        "You must select one of the provided options."
        "Even if no option is perfect, pick the closest one."
        "Confirm your final choice, then end with:\n\n"
        "#### ANSWER: (letter)\n"
    )

    if language.lower() == "english":
        return (
            "You are an expert tutor solving a multiple choice question.\n\n"
            + REASONING
            + VERDICT
        )
    else:
        TRANSLATION = (
            "Translation:\n"
            f"Translate the question and every option from {language} into English. "
            "Flag any term that is ambiguous or hard to translate precisely.\n\n"
        )
        return (
            f"You are an expert multilingual tutor solving a multiple choice question.\n"
            f"The question is in {language}. All reasoning must be in English.\n\n"
            + TRANSLATION
            + REASONING
            + VERDICT
        )
def prepare_dataset(data_path: str):
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows 
 
def format_teacher_prompt(instruction: str, language: str):
    system_prompt = get_system_prompt(language)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},]
def _build_instruction(row: dict[str, Any]) -> str:
    options = row["options"]
    if not isinstance(options, list):
        options = list(options)
    return f"{row['question']}\n\n{_options_to_text(options)}"   

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference + eval on test JSONL")
    parser.add_argument("--base_model", required=True,
                        help="Base student model path")
    parser.add_argument("--adapter_path", default="",
                        help="Optional PEFT adapter path")
    # parser.add_argument("--merged_model", required=True,
    #                     help="student model path")
    parser.add_argument("--test_data", required=True, help="Test JSONL path")
    parser.add_argument("--output_predictions", required=True,
                        help="Predictions JSONL path")
    parser.add_argument("--report_file", required=True,
                        help="Metrics report text file")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)
    output_path = Path(args.output_predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = prepare_dataset(args.test_data)
    batch_size = 128
    written = 0
    print("loading distilled model from:____________________________________________________________________________", args.base_model)
    student, tokenizer = load_vllm_llm(
        # args.merged_model,
        # args.adapter_path,
        args.base_model,
        enforce_eager=True,
        tensor_parallel_size=1, 
         max_model_len=32768   
    )
    with output_path.open("w", encoding="utf-8") as fp:
        for start in tqdm(range(0, len(dataset), batch_size), desc="Generating", dynamic_ncols=True):
            batch = dataset[start: start + batch_size]

            prompts = []
            for row in batch:
                question_with_choices = _build_instruction(row)
                full_language_name = LANGUAGE_MAP.get(row["language"], row["language"])
                prompt = format_teacher_prompt(question_with_choices, full_language_name)
                prompts.append(prompt)

            parsed_outputs = generate_and_parse(
                student,
                tokenizer,
                prompts,
            )
            # print(parsed_outputs[0],"gold_ans:",batch[0]["answer"])
            for row, output in zip(batch, parsed_outputs):
                full_lang = LANGUAGE_MAP.get(row["language"], row["language"])
                record = {
                    "language": full_lang,
                    "subject": row["subject"],
                    "question": row["question"],
                    "gold_answer": row["answer"],          # e.g. "F"
                    "predicted_answer": output["final_answer"],
                    "generation": output["raw_generation"],
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    LOGGER.info("Saved %d predictions to %s", written, output_path)


    predictions = []
    with output_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            predictions.append(json.loads(line.strip()))

    from collections import defaultdict
    lang_correct = defaultdict(int)
    lang_total   = defaultdict(int)

    for pred in predictions:
        lang = pred["language"]
        lang_total[lang] += 1
        if pred["gold_answer"].strip().upper() == pred["predicted_answer"].strip().upper():
            lang_correct[lang] += 1

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as rf:
        overall_correct = sum(lang_correct.values())
        overall_total   = sum(lang_total.values())
        
        for lang in sorted(lang_total.keys()):
            acc = lang_correct[lang] / lang_total[lang] * 100
            line = f"{lang} ACCURACY: {acc:.2f}"
            rf.write(line + "\n")
            LOGGER.info(line)
        
        overall_acc = overall_correct / overall_total * 100 if overall_total else 0
        summary = f"OVERALL ACCURACY: {overall_acc:.2f}"
        rf.write(summary + "\n")
        LOGGER.info(summary)
    LOGGER.info("Report saved to %s", report_path)

if __name__ == "__main__":
    main()
