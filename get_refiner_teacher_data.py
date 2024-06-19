import os
import re
import time
import argparse
import nvidia_smi
import torch
# import torch.distributed as dist
# from torch.nn import DataParallel
# from torch.nn.parallel import DistributedDataParallel as DDP
# from accelerate import init_empty_weights, infer_auto_device_map
from groq import Groq
from openai import OpenAI
import evaluate
from tqdm.auto import tqdm

# from src import slurm
from utils import load_file, save_file_jsonl, process_retriever_passage, \
    postprocess_summarization, model_generate, EVAL_DATASET, DATASET_TYPE, TRAIN_DATASET
from metrics import calc_acc


GROQ_TOKEN = ""
OPENAI_TOKEN = ""
# os.environ["GROQ_API_KEY"] = GROQ_TOKEN
groq_client = Groq(api_key=GROQ_TOKEN)
openai_client = OpenAI(api_key=OPENAI_TOKEN) #, base_url="https://api.chatanywhere.tech/v1")


def assemble_conversation(question, context, task_type="question", strict=False):
    # print("Strict prompt:", strict)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_STRICT.format(task_type=task_type)
                                      if strict
                                      else SYSTEM_PROMPT_RELAX.format(task_type=task_type)},
        {"role": "user", "content": RETRIEVAL_PROMPT.format(
            question=question, context=context, task_type=task_type.capitalize())},
    ]
    return messages


def api_generate(client, messages, model, max_token):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.,
        max_tokens=max_token
    )
    ans = chat_completion.choices[0].message.content
    return ans


SYSTEM_PROMPT_RELAX = """You are an expert research assistant. Your job is to find the quotes from the markdown documents that are relevant to a {task_type}.
Please mark quotes with sections and titles of documents to group quotes by different information only from relevant documents.
Rule of labelling sections: if the fact of the first dummy quote "aaa" from a document with title "## AAA" is consistent with that of the second dummy quote "bbb" from a document with title "## BBB", and they contradict to that of the third dummy quote "ccc" from a document with title "## CCC", then label them as:
```
1.1. ## AAA
aaa

1.2. ## BBB
bbb

2.1. ## CCC
ccc

```
Quotes should be verbatim and context completed.
Please respond without any explanation."""

SYSTEM_PROMPT_STRICT = """You are an expert research assistant. Your job is to find the quotes from the markdown documents that either support or contradict to a {task_type}.
Please mark quotes with sections and titles of documents to group quotes by different information only from relevant documents that are helpful with answering to the {task_type}.
Rule of labelling sections: if the fact of the first dummy quote "aaa" from a document with title "## AAA" is consistent with that of the second dummy quote "bbb" from a document with title "## BBB", and they contradict to that of the third dummy quote "ccc" from a document with title "## CCC", then label them as:
```
1.1. ## AAA
aaa

1.2. ## BBB
bbb

2.1. ## CCC
ccc

```
Quotes should be verbatim and context-completed.
Please respond without any explanation."""

RETRIEVAL_PROMPT = """{task_type}: {question}\nDocument (multiple documents are separated by "---"):\n{context}"""

regex_section = re.compile(r"((\d+\.)+|\*)\s+(##\s)?([^\n]+?)(:\s|\n+)(.*)")
regex_quote = re.compile(r'^["\'].*["\']$')
regex_dummy = re.compile(r"([A-Z])\1{2}")


def batch_extract_from_qa(
        qa_data,
        model,
        tokenizer,
        batch_size=4,
        instruction=None,
        task_type="question",
        strict_prompt=False,
        inference_name="output",
        top_n=10,
        max_input_len=None,
        eval_metrics=[]
    ):
    os.makedirs(f".task/{args.task}", exist_ok=True)
    if torch.cuda.is_available():
        nvidia_smi.nvmlInit()
        lst_cuda_devices = []
        for i in range(torch.cuda.device_count()):
            lst_cuda_devices.append(nvidia_smi.nvmlDeviceGetHandleByIndex(i))

    print(f"Total QA pairs: {len(qa_data)}")
    # pub_health = load_file('../eval_data/health_claims_processed.jsonl')
    # triviaqa = load_file('../eval_data/triviaqa_test.jsonl')
    # popqa = load_file('../eval_data/popqa_longtail.jsonl')
    # retrieve_data =  load_file("../train_data/extractor_retrieve_wiki.jsonl")

    # tokenizer = AutoTokenizer.from_pretrained("selfrag/selfrag_llama2_7b", torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="mps")
    # model = model.eval()

    lst_batch = []
    lst_titled_ans = []
    for data in (pbar:= tqdm(qa_data, mininterval=1., desc="preprocess")):
        # instruction = TASK_INST.get(task, None)
        question, context = process_retriever_passage(
            data, n_docs=top_n, instruction=instruction, max_len=max_input_len, sort=False)

        messages = assemble_conversation(question, context, task_type=task_type, strict=strict_prompt)

        # try:
        #     ans = groq_generate(client, messages=messages, model="llama2-70b-4096", max_token=2048)
        # except:
        #     time.sleep(60)
        #     try:
        #         ans = groq_generate(client, messages=messages, model="llama2-70b-4096", max_token=2048)
        #     except:
        #         time.sleep(60*10)
        #         client = Groq(api_key=GROQ_TOKEN)
        #         ans = groq_generate(client, messages=messages, model="llama2-70b-4096", max_token=2048)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lst_batch.append(prompt)
        if (len(lst_batch) == batch_size) or (len(qa_data) - len(lst_titled_ans) == len(lst_batch)):
            lst_output, _, _ = model_generate(
                lst_batch,
                model=model,
                tokenizer=None,
                max_new_tokens=2048,
                temperature=0.,
                top_p=1,
                do_sample=False,
            )

            lst_output = [o[0].rstrip(tokenizer.eos_token).rstrip("<|eot_id|>") for o in lst_output]
            lst_titled_ans.extend(lst_output)

            if torch.cuda.is_available():
                postfix = {
                    f"GPU{i}": nvidia_smi.nvmlDeviceGetUtilizationRates(device).memory
                    for i, device in enumerate(lst_cuda_devices)
                }
            else:
                postfix = {}

            for metric in eval_metrics:
                metric.add_batch(
                    predictions=lst_output,
                    references=lst_batch,
                )
                postfix.update(metric.compute())

            lst_batch.clear()
            pbar.set_postfix(postfix, refresh=True)

            save_file_jsonl(lst_titled_ans, f'.task/{args.task}/extract_expunge.tmp')

    for ans, data in tqdm(zip(lst_titled_ans, qa_data), desc="postprocess"):
        data[inference_name] = postprocess_summarization(ans)

    return qa_data


def extract_from_qa(
        qa_data,
        model,
        tokenizer=None,
        task_type="question",
        strict_prompt=False,
        max_input_len=None
    ):
    print(f"Total qa pairs: {len(qa_data)}")
    lst_titled_ans = []
    for data in (pbar:= tqdm(qa_data[len(lst_titled_ans):])):
        # instruction = TASK_INST.get(task, None)
        question, context = process_retriever_passage(data, n_docs=10, max_len=max_input_len)
        messages = assemble_conversation(question, context, task_type=task_type, strict=strict_prompt)
        # try:
        #     ans = groq_generate(client, messages=messages, model="llama2-70b-4096", max_token=2048)
        # except:
        #     time.sleep(60)
        #     try:
        #         ans = groq_generate(client, messages=messages, model="llama2-70b-4096", max_token=2048)
        #     except:
        #         time.sleep(60*10)
        #         client = Groq(api_key=GROQ_TOKEN)
        #         ans = groq_generate(client, messages=messages, model="llama2-70b-4096", max_token=2048)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lst_output, _, _ = model_generate(
            prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.,
            do_sample=False,
        )
        ans = lst_output[0]
        titled_ans = {
            "question": question,
            "context": context,
            "output": postprocess_summarization(ans),
        }
        # fallback to retrieved context if output is empty
        # if len(titled_ans["output"].strip()) == 0:
        #     titled_ans["output"] = context
        if "answers" in data:
            titled_ans["answers"] = data["answers"]
        if "choices" in data:
            titled_ans["choices"] = data["choices"]
        if "label" in data:
            titled_ans["label"] = data["label"]
        if "answerKey" in data:
            titled_ans["answerKey"] = data["answerKey"]
        lst_titled_ans.append(titled_ans)

        if len(lst_titled_ans) % 10 == 0:
            acc = sum(calc_acc(data, ans) is True for ans in lst_titled_ans) / len(lst_titled_ans)
            pbar.set_postfix({"Acc": acc})
            save_file_jsonl(lst_titled_ans, f'extract_expunge.tmp')

        return lst_titled_ans


def parse_args():
    parser = argparse.ArgumentParser(description="Get monitor teacher data for train and evaluation task")

    task_choice = list(EVAL_DATASET.keys()) + list(TRAIN_DATASET.keys())
    parser.add_argument(
        "--task",
        type=str,
        choices=task_choice,
        help=f"Task of text extraction from evaluate dataset. Choices: {task_choice}",
        required=True
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU for evaluation."
    )
    parser.add_argument(
        "--max_input_len", type=int, default=None, help="Will be used to truncate input context"
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="Number of retrieved document as model input"
    )
    parser.add_argument(
        "--inference_name", type=str, help="will be used for naming model output stored in data"
    )
    parser.add_argument(
        "--start", type=int, default=None, help="Index from which the data starts"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Index till which the data ends"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Path of input data"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path of output data"
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # slurm.init_distributed_mode(args)

    # client = Groq(api_key=GROQ_TOKEN)

    eval_metrics = [
        # evaluate.load("exact_match", cache_dir=f"./.task/get_monitor_teacher_data/{args.task}"),
        evaluate.load("src/rouge.py", cache_dir=f"./.task/get_monitor_teacher_data/{args.task}"),
    ]

    print("loading tokenizer")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding="longest", padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model")
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)

    from vllm import LLM
    model = LLM(args.model_name_or_path, tensor_parallel_size=torch.cuda.device_count())

    data_slice = slice(args.start, args.end)
    name_suffix = f'{"" if args.start is None else f"_from{args.start}"}{"" if args.end is None else f"_to{args.start}"}'
    if args.task.endswith("train"):
        if args.input is None:
            qa_data = load_file(os.path.join('./train_data/', TRAIN_DATASET[args.task]))
        else:
            qa_data = load_file(args.input)
        qa_data = qa_data[data_slice]
        qa_data = batch_extract_from_qa(qa_data[data_slice],
                                        model=model,
                                        tokenizer=tokenizer,
                                        batch_size=args.per_gpu_eval_batch_size,
                                        inference_name=args.inference_name,
                                        top_n=args.top_n,
                                        eval_metrics=eval_metrics,
                                        strict_prompt=args.task == "pubhealth_train",
                                        max_input_len=args.max_input_len)
        if not isinstance(args.output, str):
            args.output = f'./train_data/{args.task}_refiner_teacher_expunge{name_suffix}.jsonl'

    else:
        if args.input is None:
            qa_data = load_file(os.path.join(f'./eval_data/', EVAL_DATASET[args.task]))
        else:
            qa_data = load_file(args.input)
        qa_data = qa_data[data_slice]
        qa_data = batch_extract_from_qa(qa_data,
                                        model=model,
                                        tokenizer=tokenizer,
                                        batch_size=args.per_gpu_eval_batch_size,
                                        inference_name=args.inference_name,
                                        task_type=DATASET_TYPE.get(args.task, "question"),
                                        top_n=args.top_n,
                                        eval_metrics=eval_metrics,
                                        strict_prompt=args.task == "fever",
                                        max_input_len=args.max_input_len)
        
        if not isinstance(args.output, str):
            args.output = f'./eval_data/top_{args.top_n}/{args.task}_refiner_teacher_expunge{name_suffix}.jsonl'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_file_jsonl(qa_data, args.output)

    # dist.destroy_process_group()