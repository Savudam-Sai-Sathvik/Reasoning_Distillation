
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any
import re
from tqdm.auto import tqdm

from datasets import Dataset, concatenate_datasets
from vllm import LLM, SamplingParams
from data.mmlupro import MMLUPro
from utils import load_vllm_llm, prompt_vllm

LANGUAGE_MAP = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", 
    "kn": "Kannada", "ta": "Tamil"
}


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    re.IGNORECASE | re.DOTALL,
)


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



def sample_datasets(samples_per_language: list[int], split: str = "test", seed: int = 42) -> Dataset:
    """Fetch and sample the requested number of rows for each language using MMLUPro class."""
    if len(samples_per_language) != len(LANGUAGES):
        raise ValueError(f"--num_samples must contain {len(LANGUAGES)} integers.")

    all_subsets = []
    test_subsets = []
    for i, count in enumerate(samples_per_language):
        lang_name = LANGUAGES[i]
        if count <= 0:
            continue

        LOGGER.info(f"Loading {count} samples for language: {lang_name}")
        
        loader = MMLUPro(language=lang_name, split=split)

        lang_dataset = loader.load_mmlu_pro()

        if len(lang_dataset) == 0:
            LOGGER.warning(f"No data found for {lang_name} in dataset.jsonl")
            continue

      
        shuffled = lang_dataset.shuffle(seed=seed)
        sampled_lang = shuffled.select(range(min(count, len(shuffled))))
        #use the remaining data as test
        remaining_start = min(count, len(shuffled))
        remaining_end = min(remaining_start + 250, len(shuffled))
        if remaining_end > remaining_start:
            test_subsets.append(shuffled.select(range(remaining_start, remaining_end)))
        else:
            LOGGER.warning(f"No remaining data for test set in {lang_name}")
        all_subsets.append(sampled_lang)


    if not all_subsets:
        raise ValueError("No samples were collected. Check your dataset.jsonl and language tags.")

    return (concatenate_datasets(all_subsets),concatenate_datasets(test_subsets))
    # return concatenate_datasets(all_subsets)
# def robust_sample_datasets(samples_per_language: list[int], split: str = "test", seed: int = 42) -> Dataset:
#     if len(samples_per_language) != len(LANGUAGES):
#         raise ValueError(f"--num_samples must contain {len(LANGUAGES)} integers.")

#     all_subsets = []
#     test_subsets = []
#     for i, count in enumerate(samples_per_language):
#         lang_name = LANGUAGES[i]
#         if count <= 0:
#             continue
#         LOGGER.info(f"Loading {count} samples for language: {lang_name}")
#         loader = MMLUPro(language=lang_name, split=split)
#         lang_dataset = loader.load_mmlu_pro()
#         lan_domain_specific_datasets = lang_dataset['subject']



def generate_and_parse(model, tokenizer, messages):
    # parse_output = {}
    # sampling_params = SamplingParams(
    #     max_tokens=2048,
    #     temperature=0.7,
    #     top_p=0.9,
    #     repetition_penalty=1.1,
    #     stop=["\n\n#### "]
    # )

    # outputs = model.generate(messages, sampling_params)
    outputs = prompt_vllm(
            model,
            tokenizer,
            messages,
            # temperature=0.7,
            max_new_tokens=2048,
            # top_p=0.9,
        )

    parsed_outputs = []

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
    

def format_teacher_prompt(instruction: str, language: str):
    system_prompt = get_system_prompt(language)
    # system_prompt = (
    #     f"You are an expert multilingual tutor solving a multiple choice question.\n"
    #     f"The question may be in {language}. All reasoning must be in English.\n\n"

    #     "Translation:\n"
    #     "Translate the question and every option into English. "
    #     "Flag any term that is ambiguous or hard to translate precisely.\n\n"

    #     "Reasoning:\n"
    #     "Identify what the question is asking.\n"
    #     "For each option, refer to its actual wording and evaluate: "
    #     "'What would this mean if true? Does it hold under scrutiny?' "
    #     "Eliminate with evidence, not assumption.\n\n"

    #     "Verdict:\n"
    #     "Did you reason from the actual text, or your assumptions about it? "
    #     "Are there options you dismissed too quickly? "
    #     "Confirm your final choice, then end with:\n\n"

    #     "#### ANSWER: (letter)"
    # )
    # system_prompt = (
    #     f"You are an expert multilingual tutor solving a multiple choice question.\n"
    #     f"All reasoning must be in English, even if the question is in {language}.\n\n"

    #     "Reasoning:\n"
    #     "First, identify what the question is asking.\n"
    #     "For each option, quote the key claim, then evaluate it: "
    #     "'What would this mean if true? Does it hold under scrutiny?' "
    #     "Eliminate with evidence, not assumption.\n\n"

    #     "Verdict:\n"
    #     "Did you evaluate the actual wording of each option, or your paraphrase of it? "
    #     "Review your reasoning. Did you apply every concept correctly? "
    #     "Are there options you dismissed too quickly? Confirm your final choice.\n\n"

    #     "End your response with:\n"
    #     "#### ANSWER: (letter)"
    # )
    # system_prompt = (
    # f"You are an expert multilingual tutor solving a multiple choice question.\n"
    # f"All reasoning must be in English, even if the question is in {language}.\n\n"

    # "Reasoning:\n"
    # "Evaluate every option. For each, ask: 'What would this mean if true? "
    # "Does it hold under scrutiny?' Eliminate with evidence, not assumption.\n\n"

    # "Verdict:\n"
    # "Review your reasoning. Did you apply every concept correctly? "
    # "Any options dismissed too quickly? Deliver your final judgment.\n\n"

    # "End your response with:\n"
    # "#### ANSWER: (letter)"
# )
    #first train.jsonl
    # system_prompt = (
    #         f"You are an expert multilingual tutor.\n"
    #         f"Answer the multiple choice question in English.\n\n"
    #         "Structure your response as follows:\n\n"
    #         "Reasoning:\n"
    #         "Walk through the problem step by step. For each option, briefly explain "
    #         "why it is correct or incorrect.\n\n"
    #         "#### ANSWER: (J)"
    #     )
    
    # system_prompt = (
    #     f"You are an expert tutor. Answer the multiple choice question in {language}.\n"
    #     "Think step by step and then give a single final answer in the format:\n"
    #     "#### ANSWER: (J)"
    # )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query teacher and build train corpus JSONL"
    )
    parser.add_argument(
        "--teacher_model",
        required=True,
        help="Hugging Face path to the teacher model",
        default="/scratch/scai/phd/aiz248311/col772/a4/models/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--num_samples",
        type=str,
        required=True,
        help=(
            "Comma-separated sample counts for english,hindi,bengali,"
            "kannada,tamil"
        ),
        default=(2000,2000,2000,2000,2000)
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL path for train corpus",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="Target fraction of GPU memory for vLLM; lower if startup fails",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def _parse_num_samples(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain exactly 5 comma-separated integers "
            "for english,hindi,bengali,kannada,tamil"
        )

    try:
        counts = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "--num_samples must contain only integers"
        ) from exc

    if any(count < 0 for count in counts):
        raise ValueError("--num_samples values must be >= 0")

    return counts


def _build_instruction(row: dict[str, Any]) -> str:
    options = row["options"]
    if not isinstance(options, list):
        options = list(options)
    return f"{row['question']}\n\n{_options_to_text(options)}"


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    samples_per_language = _parse_num_samples(args.num_samples)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampled, test_dataset = sample_datasets(
        samples_per_language=samples_per_language,
        split=args.split,
        seed=args.seed,
    )
    LOGGER.info("Collected %d samples", len(sampled))

    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    if test_dataset:
        test_path = output_path.with_stem("test")
        with test_path.open("w", encoding="utf-8") as fp:
            for row in test_dataset:
                fp.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        LOGGER.info("Saved %d test rows to %s", len(test_dataset), test_path)
    batch_size = 128
    written = 0
    with output_path.open("w", encoding="utf-8") as fp:
      for start in tqdm(range(0, len(sampled), batch_size), desc="Generating reasoning traces", dynamic_ncols=True):
            batch = sampled.select(range(start, min(start + batch_size, len(sampled))))
            # batch = sampled.select(range(start, start + batch_size))
            prompts = []
            q_s = []
            
            for row in batch:
                # print(row)
                question_with_choices = _build_instruction(row)
                full_language_name = LANGUAGE_MAP.get(row["language"], row["language"])
                prompt = format_teacher_prompt(question_with_choices, full_language_name)

                q_s.append(question_with_choices)
                prompts.append(prompt)

            parsed_outputs = generate_and_parse(
                teacher,
                tokenizer,
                prompts   # batch
            )
            # print(prompts[0])
            # print(parsed_outputs[0],"crt_ans:", str(batch[0].get("answer", "")).upper()[:1])
            for i, parsed in enumerate(parsed_outputs):

                if not parsed["final_answer"]:
                    continue

                record = {
                    "question": q_s[i],
                    "reasoning": parsed["reasoning"],
                    "final_answer": parsed["final_answer"],
                    "gold_answer": str(batch[i].get("answer", "")).upper()[:1],
                    "language": batch[i]["language"],
                    "subject": batch[i].get("subject"),
                    "prompt": prompts[i],
                    "teacher_generation": parsed["raw_generation"]
                }

                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    LOGGER.info("Saved %d rows to %s", written, output_path)


if __name__ == "__main__":
    main()