import os
import argparse
import datetime as dt
from tqdm.auto import tqdm

import torch
from llmlingua import PromptCompressor
from transformers.utils import logging
from accelerate.utils import gather_object
from accelerate import Accelerator, InitProcessGroupKwargs

logging.get_logger("transformers").setLevel(logging.ERROR)

from metrics import calc_acc
from utils import load_file, save_file_jsonl, process_retriever_passage, \
    TASK_INST, EVAL_DATASET, TRAIN_DATASET_REFINER_OUTPUT


def parse_args():
    parser = argparse.ArgumentParser(description="Get evluation for train and evaluation task")

    task_choice = list(EVAL_DATASET.keys())
    parser.add_argument(
        "--task",
        type=str,
        choices=task_choice,
        help=f"Task of text summarization from evaluate dataset. Choices: {task_choice}.",
        required=False,
        default="train"
    )
    parser.add_argument(
        "--inference_name", type=str, help="will be used for naming model output stored in data."
    )
    parser.add_argument(
        "--input", type=str, help="Path of input data."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the evaluation output."
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="name to directory containing evaluate data."
    )
    parser.add_argument(
        "--use_llmlingua2", action="store_true", default=None, help="Whether to use llmlingua-2 compressor based on the paper."
    )
    parser.add_argument(
        "--rate", default=0.5, type=float, help="Compress rate in paper."
    )
    parser.add_argument(
        "--dynamic_context_compression_ratio", default=0.25, type=float, help="Dynamic context compression ratio in paper."
    )

    args = parser.parse_args()
    return args


def compress_prompt(llmlingua, model, data, top_n=10, rate=0.5, dynamic_context_compression_ratio=0.25):
    llmlingua.model = accelerator.unwrap_model(model)
    llmlingua.device = model.device

    question, context = process_retriever_passage(data, n_docs=top_n)
    lst_restore_context = []
    for i, ctx in enumerate(context.split("\n---\n")):
        ctx = ctx.replace("## ", f"[{i}]")
        lst_restore_context.append(ctx)

    instruction = TASK_INST.get(args.task, None)

    processed = llmlingua.compress_prompt(
        lst_restore_context[:top_n],
        question=question,
        instruction='' if instruction is None else instruction,
        use_sentence_level_filter=False,
        rate=rate,
        # Set the special parameter for LongLLMLingua
        condition_in_question="after_condition",
        reorder_context="sort",
        dynamic_context_compression_ratio=dynamic_context_compression_ratio,
        condition_compare=True,
        context_budget="+100",
        rank_method="longllmlingua",
    )

    return processed["compressed_prompt"]


if __name__ == '__main__':
    args = parse_args()

    kwargs = InitProcessGroupKwargs(timeout=dt.timedelta(seconds=21600))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    accelerator.print("Task:", args.task)

    # slurm.init_distributed_mode(args)

    # client = Groq(api_key=GROQ_TOKEN)

    llmlingua = PromptCompressor(
        use_llmlingua2=args.use_llmlingua2,
        device_map="cpu",
        model_config={"torch_dtype": torch.float16})
    model = accelerator.prepare(llmlingua.model)

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
    lst_data = []
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
            context = compress_prompt(
                llmlingua=llmlingua,
                model=model,
                data=data,
                top_n=args.top_n,
                rate=args.rate,
                dynamic_context_compression_ratio=args.dynamic_context_compression_ratio)

            data[args.inference_name] = context
            lst_data.append(data)

    lst_processed_data = gather_object(data_part)[:len(qa_data)]
    accelerator.wait_for_everyone()
    accelerator.print(f"Gathered {len(lst_processed_data)} data")

    # sort data by original order
    d_data_idx = {data["question"]: i for i, data in enumerate(qa_data)}
    lst_processed_data.sort(key=lambda data: d_data_idx[data["question"]])

    if accelerator.is_main_process:
        acc = 0.
        for data in lst_processed_data:
            is_match = calc_acc(data, data[args.inference_name])
            acc += is_match is True

        eval_result = {"acc": acc / len(lst_processed_data)}

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