import os
import argparse
import torch
import datetime as dt
from tqdm.auto import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from peft.peft_model import PeftModel
from transformers import AutoTokenizer, T5TokenizerFast, T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.utils import logging

logging.get_logger("transformers").setLevel(logging.ERROR)

import evaluate
from metrics import calc_acc
from utils import load_file, save_file_jsonl, process_retriever_passage, \
    PROMPT_DICT, TASK_INST, EVAL_DATASET, EVAL_DATASET_REFINER_OUTPUT, TRAIN_DATASET_REFINER_OUTPUT

PROMPT_REFINER_LORA_INPUT = PROMPT_DICT["prompt_refiner_lora"]
PROMPT_REFINER_PREFIX_INPUT = PROMPT_DICT["prompt_refiner_prefix"]
PROMPT_DOWNSTREAM_LORA_INPUT = PROMPT_DICT["prompt_downstream_lora"]
PROMPT_DOWNSTREAM_PREFIX_INPUT = PROMPT_DICT["prompt_downstream_prefix"]
PROMPT_RECOMP_ABSTRACTIVE_INPUT = PROMPT_DICT["prompt_recomp_abstractive"]


def parse_args():
    parser = argparse.ArgumentParser(description="Get evluation for train and evaluation task")

    task_choice = list(EVAL_DATASET_REFINER_OUTPUT.keys()) + list(TRAIN_DATASET_REFINER_OUTPUT.keys())
    parser.add_argument(
        "--task",
        type=str,
        choices=task_choice,
        help=f"Task of text summarization from evaluate dataset. Choices: {task_choice}",
        required=False,
        default="train"
    )
    parser.add_argument(
        "--inference_name",
        type=str,
        choices=["refiner", "downstream", "recomp_abstractive"],
        help="will be used for naming model output stored in data"
    )
    parser.add_argument(
        "--base_model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None, help="path to directory containing PEFT adapter weights"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, default=None, help="path to directory of tokenizer config file"
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU for evaluation."
    )
    parser.add_argument(
        "--top_n", default=10, type=int, help="Number of retrieved documents as input"
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Whether to use flash attention2"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Where to read the evaluation data."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the evaluation output."
    )

    args = parser.parse_args()
    return args


def batch_evaluate(model, tokenizer, lst_batch):
    inputs = tokenizer(lst_batch, padding="longest", return_tensors="pt")
    if not isinstance(model, (PeftModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM)):
        model = accelerator.unwrap_model(model)

    with torch.no_grad():
        if isinstance(model, (AutoModelForCausalLM, PeftModel)):
            preds = model.generate(
                **inputs.to(accelerator.device),
                top_p=1,
                temperature=None,
                do_sample=False,
                max_new_tokens=2048,
                repetition_penalty=1.0,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True)
        else:
            preds = model.generate(
                **inputs.to(accelerator.device),
                output_scores=True,
                max_new_tokens=2048,
                return_dict_in_generate=True,
                use_cache=True)

    if isinstance(model, AutoModelForCausalLM):
        start_idx = inputs.input_ids.shape[1]
    else:
        start_idx = 0

    pred_token_ids = preds.sequences[:, start_idx:]
    lst_output = tokenizer.batch_decode(pred_token_ids, skip_special_tokens=True)
    lst_output = [out.rstrip(tokenizer.eos_token).strip() for out in lst_output]
    assert len(lst_output) == len(lst_batch), f"{lst_batch}\n{lst_output}"
    return lst_output


if __name__ == '__main__':
    args = parse_args()

    kwargs = InitProcessGroupKwargs(timeout=dt.timedelta(seconds=21600))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    accelerator.print("Task:", args.task)
    accelerator.print("Adapter:", args.adapter_path)

    # slurm.init_distributed_mode(args)

    eval_metrics = [
        evaluate.load('src/rouge.py', cache_dir=f"./.task/evaluation_accelerate/{args.task}")
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.base_model_name_or_path,
        padding="longest",
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if isinstance(tokenizer, (T5TokenizerFast, T5Tokenizer)):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model_name_or_path,
            # torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager" # "sdpa",
            # torch_dtype=torch.float16
        )
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)

    model = accelerator.prepare(model)
    gen_kwargs = dict(tokenizer=tokenizer)

    if args.input:
        qa_data = load_file(args.input)
    elif args.task.startswith('train'):
        qa_data = load_file(os.path.join('./train_data/', TRAIN_DATASET_REFINER_OUTPUT[args.task]))
    else:
        qa_data = load_file(os.path.join(f'./eval_data/', EVAL_DATASET[args.task]))

    if os.path.exists(f"./.task/evaluate/{args.task}.tmp"):
        lst_prev_predicts = load_file(f'./.task/evaluate/{args.task}.tmp')
        accelerator.print(f'Resumed task "{args.task}" from cache')
    else:
        os.makedirs("./.task/evaluate/", exist_ok=True)
        lst_prev_predicts = []

    remain_data = qa_data[len(lst_prev_predicts):]
    # remain_data.sort(key=lambda x: len(x["context"]), reverse=True)
    accelerator.wait_for_everyone()

    # Evaluate!
    model.eval()
    lst_data = []
    lst_batch = []
    if args.inference_name == "refiner":
        prompt_template = PROMPT_REFINER_LORA_INPUT
    elif args.inference_name == "recomp_abstractive":
        prompt_template = PROMPT_RECOMP_ABSTRACTIVE_INPUT
    else:
        prompt_template = PROMPT_DOWNSTREAM_LORA_INPUT

    if args.inference_name == "downstream":
        instruction = TASK_INST.get(args.task, None)
    else:
        instruction = None
    if instruction is not None:
        accelerator.print(f"use instruction: {instruction}")

    accelerator.print(f"Use template: {prompt_template}")

    with accelerator.split_between_processes(remain_data) as data_part:
        for i, data in enumerate(tqdm(
            data_part,
            desc=f"Evaluate-{accelerator.process_index}",
            position=accelerator.process_index,
            # the progress bar will otherwise be cleared up
            # and the cursor position unchanged when finished
            leave=False,
            delay=1.,
        )):
            question, context = process_retriever_passage(
                data, instruction=instruction, n_docs=args.top_n)
            prompt = prompt_template.format(question=question, context=context)

            data["context"] = context
            lst_batch.append(prompt)
            lst_data.append(data)

            if (len(lst_batch) < args.per_gpu_eval_batch_size) \
                and (len(data_part) != i + 1):
                continue

            lst_batch_predicts = batch_evaluate(model, tokenizer, lst_batch)
            assert len(lst_data) == len(lst_batch_predicts)

            for metric in eval_metrics:
                metric.add_batch(
                    predictions=lst_batch_predicts,
                    references=[data["context"] for data in lst_data],
                )

            # put inference output into data part
            for data, pred in zip(lst_data, lst_batch_predicts):
                data[args.inference_name] = pred

            lst_batch.clear()
            lst_data.clear()

    lst_processed_data = gather_object(data_part)[:len(qa_data)]
    accelerator.wait_for_everyone()
    accelerator.print(f"Gathered {len(lst_processed_data)} data")

    # sort data by original order
    d_data_idx = {data["question"]: i for i, data in enumerate(qa_data)}
    lst_processed_data.sort(key=lambda data: d_data_idx[data["question"]])

    if accelerator.is_main_process:
        acc = 0
        for data in lst_processed_data:
            is_match = calc_acc(data, data[args.inference_name])
            acc += is_match is True

        eval_result = {"acc": acc / len(lst_processed_data)}
        for metric in eval_metrics:
            eval_result.update(metric.compute())

        accelerator.print(eval_result)

        file_suffix = '&'.join([f"{k}={v:.3f}" for k, v in eval_result.items()])
        if args.output_dir is None:
            if args.task.startswith('train'):
                args.output_dir = "./train_data/"
            else:
                args.output_dir = f"./eval_data/top_{args.top_n}"

        save_path = os.path.join(args.output_dir, f'{args.task}_{args.inference_name}_{file_suffix}.jsonl')
        os.makedirs(args.output_dir, exist_ok=True)
        save_file_jsonl(lst_processed_data, save_path)
