import os
import torch
import backoff
import argparse
# from groq import Groq
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError
from tqdm.auto import tqdm

from utils import load_file, save_file_jsonl, process_retriever_passage, \
    postprocess_summarization, model_generate, PROMPT_DICT, TASK_INST, EVAL_DATASET, TRAIN_DATASET
from metrics import calc_acc


# GROQ_TOKEN = ""
OPENAI_TOKEN = ""
# os.environ["GROQ_API_KEY"] = GROQ_TOKEN
# groq_client = Groq(api_key=GROQ_TOKEN)
openai_client = OpenAI(api_key=OPENAI_TOKEN)
# openai_client = OpenAI(api_key=OPENAI_TOKEN, base_url="https://api.chatanywhere.tech/v1")


def assemble_conversation(system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages


@backoff.on_exception(backoff.expo, (APIStatusError, APIConnectionError, APITimeoutError))
def api_generate(client, messages, model, max_token):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.,
        max_tokens=max_token
    )
    ans = chat_completion.choices[0].message.content
    return ans


def openai_extract_from_qa(
        qa_data,
        model,
        inference_name,
        context_key,
        instruction=None
    ):
    os.makedirs(f".task/{args.task}", exist_ok=True)

    print(f"Total QA pairs: {len(qa_data)}")

    lst_question_context = []
    lst_titled_ans = []
    for data in (pbar:= tqdm(qa_data, mininterval=1.)):
        question = ('' if instruction is None else f'{instruction}\n') + data["question"]

        if context_key is None:
            context = ''
            prompt = PROMPT_DICT["prompt_no_input"].format(instruction=question)
        else:
            context = data[context_key]

        if context_key == "context":
            lst_restore_context = []
            for i, ctx in enumerate(context.split("\n---\n")):
                ctx = ctx.replace("## ", f"[{i}]")
                lst_restore_context.append(ctx)

            context = '\n'.join(lst_restore_context)
            prompt = PROMPT_DICT["prompt_no_input_retrieval"].format(paragraph=context, instruction=question)
        elif context_key == "refiner":
            template = SYSTEM_PROMPT_REFINER
            context = template.format(context=context)
        else:
            template = SYSTEM_PROMPT
            context = template.format(context=context)

        lst_question_context.append({"question": data["question"]})

        if context_key != "context" and context_key is not None:
            question = PROMPT_DICT["prompt_no_input"].format(instruction=question)
            messages = assemble_conversation(context, question)
        else:
            messages = assemble_conversation('', prompt)

        output = api_generate(
            client=openai_client,
            messages=messages,
            model=model,
            max_token=1024)
        lst_titled_ans.append(output)

    for qa, ans, data in zip(lst_question_context, lst_titled_ans, qa_data):
        qa[inference_name] = ans
        if "answers" in data:
            qa["answers"] = data["answers"]
        if "answer" in data:
            qa["answer"] = data["answer"]
        if "choices" in data:
            qa["choices"] = data["choices"]
        if "label" in data:
            qa["label"] = data["label"]
        if "answerKey" in data:
            qa["answerKey"] = data["answerKey"]
        if "type" in data:
            qa["type"] = data["type"]
        if "level" in data:
            qa["level"] = data["level"]
        if "context" in data:
            qa["context"] = data["context"]
        if "output" in data:
            qa["output"] = data["output"]
        if "refiner" in data:
            qa["refiner"] = data["refiner"]

    return lst_question_context


def batch_extract_from_qa(
        qa_data,
        model,
        tokenizer,
        inference_name,
        context_key,
        batch_size=4,
        instruction=None
    ):
    os.makedirs(f".task/{args.task}", exist_ok=True)
    print(f"Total QA pairs: {len(qa_data)}")
    # pub_health = load_file('./eval_data/health_claims_processed.jsonl')
    # triviaqa = load_file('./eval_data/triviaqa_test.jsonl')
    # popqa = load_file('./eval_data/popqa_longtail.jsonl')
    # retrieve_data =  load_file("./train_data/extractor_retrieve_wiki.jsonl")

    # tokenizer = AutoTokenizer.from_pretrained("selfrag/selfrag_llama2_7b", torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="mps")
    # model = model.eval()

    lst_question_context = []
    lst_batch = []
    lst_titled_ans = []
    for data in (pbar:= tqdm(qa_data, mininterval=1.)):
        question = ('' if instruction is None else f'{instruction}\n') + data["question"]
        lst_question_context.append({"question": data["question"]})

        context = data[context_key]
        if context_key == "context" and inference_name != "refiner":
            # baseline setting
            lst_restore_context = []
            for i, ctx in enumerate(context.split("\n---\n")):
                ctx = ctx.replace("## ", f"[{i}]")
                lst_restore_context.append(ctx)

            context = '\n'.join(lst_restore_context)
            template = SYSTEM_PROMPT
            context = template.format(context=context)
            # question = PROMPT_DICT["prompt_no_input"].format(instruction=question)
            # prompt = PROMPT_DICT["prompt_no_input_retrieval"].format(paragraph=context, instruction=question)
        elif context_key == "refiner":
            template = SYSTEM_PROMPT_REFINER
            context = template.format(context=context)
        else:
            template = SYSTEM_PROMPT
            context = template.format(context=context)

        # if context_key != "context":
        question = PROMPT_DICT["prompt_no_input"].format(instruction=question)
        messages = assemble_conversation(context, question)
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

            lst_batch.clear()

    for qa, ans, data in zip(lst_question_context, lst_titled_ans, qa_data):
        qa[inference_name] = ans
        if "answers" in data:
            qa["answers"] = data["answers"]
        if "answer" in data:
            qa["answer"] = data["answer"]
        if "choices" in data:
            qa["choices"] = data["choices"]
        if "label" in data:
            qa["label"] = data["label"]
        if "answerKey" in data:
            qa["answerKey"] = data["answerKey"]
        if "type" in data:
            qa["type"] = data["type"]
        if "level" in data:
            qa["level"] = data["level"]
        if "context" in data:
            qa["context"] = data["context"]
        if "output" in data:
            qa["output"] = data["output"]
        if "refiner" in data:
            qa["refiner"] = data["refiner"]

    return lst_question_context


def extract_from_qa(qa_data, model, tokenizer=None, context_key="refiner", max_input_len=None):
    print(f"Total qa pairs: {len(qa_data)}")
    lst_titled_ans = []
    for data in (pbar:= tqdm(qa_data[len(lst_titled_ans):])):
        # instruction = TASK_INST.get(task, None)
        question, context = process_retriever_passage(data, n_docs=10, threshold=None, max_len=max_input_len)
        if context_key == "context":
            template = SYSTEM_PROMPT
        else:
            template = SYSTEM_PROMPT_REFINER
        messages = assemble_conversation(question, data[context_key], template)
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
    parser = argparse.ArgumentParser(description="Get executor data for train and evaluation task")

    task_choice = list(EVAL_DATASET.keys()) + list(TRAIN_DATASET.keys())
    parser.add_argument(
        "--task",
        type=str,
        choices=task_choice,
        help=f"Task of text extraction from evaluate dataset. Choices: {task_choice}",
        required=True
    )
    parser.add_argument(
        "--inference_name", type=str, help="will be used for naming model output stored in data"
    )
    parser.add_argument(
        "--input", type=str, help="Path of input data"
    )
    parser.add_argument(
        "--context_key", type=str, default=None, help="the key of context to derive answer"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument(
        "--use_openai", action="store_true", default=None,
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU for evaluation."
    )
    parser.add_argument(
        "--start", type=int, default=None, help="Index from which the data starts"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Index till which the data ends"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path of output data"
    )
    args = parser.parse_args()
    return args


SYSTEM_PROMPT_REFINER = """You are an AI assistant backboned by selective content from different documents, answer user's question helpfully and precisely, with the guidance of the following steps:
* If there are no content provided: determine whether it is still possible to answer precisely to the question.
* If is possible, offer a helpful answer. Otherwise, offer the reason of impossibility.

* If there exists contents: determine whether the necessary information to answer the question is either directly mentioned or can be inferred from the documents.
* When there exists different information that can answer to the question, determine whether it is the question too opaque that causes the problem.
* If not, answer with a summarized information. Otherwise, also provide advice or ask question to disambiguate.
* When summarizing, ensure to include contents that are relevant to the question.
Here is the content:
{context}"""

SYSTEM_PROMPT = """You are an AI assistant backboned by selective content from different documents, answer user's question helpfully and precisely, with the guidance of the following steps:
* If there are no content provided: determine whether it is still possible to answer precisely to the question.
* If is possible, offer a helpful answer. Otherwise, offer the reason of impossibility.

* If there exists contents: determine whether the necessary information to answer the question is either directly mentioned or can be inferred from the documents.
* When there exists different information that can answer to the question, determine whether it is the question too opaque that causes the problem.
* If not, answer with a summarized information. Otherwise, also provide advice or ask question to disambiguate.
* When summarizing, ensure to include contents that are relevant to the question.
Here is the content:
{context}"""


if __name__ == '__main__':
    args = parse_args()

    # slurm.init_distributed_mode(args)

    # client = Groq(api_key=GROQ_TOKEN)

    if args.use_openai:
        model = args.model_name_or_path
    else:
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

    qa_data = load_file(args.input)
    instruction = TASK_INST.get(args.task, None)
    if instruction is not None:
        print(f"use instruction: {instruction}")

    if args.use_openai:
        lst_processed_data = openai_extract_from_qa(qa_data[data_slice],
                                                    model=model,
                                                    inference_name=args.inference_name,
                                                    context_key=args.context_key,
                                                    instruction=instruction)
    else:
        lst_processed_data = batch_extract_from_qa(qa_data[data_slice],
                                                    model=model,
                                                    tokenizer=None if args.use_openai else tokenizer,
                                                    inference_name=args.inference_name,
                                                    context_key=args.context_key,
                                                    batch_size=args.per_gpu_eval_batch_size,
                                                    instruction=instruction)
    acc = 0
    for data in lst_processed_data:
        is_match = calc_acc(data, data[args.inference_name])
        acc += is_match is True

    eval_result = {"acc": acc / len(lst_processed_data)}

    print(f"Task: {args.task}\tInference Name: {args.inference_name}\tContext Key: {args.context_key}")
    print(eval_result)

    if args.task.startswith("train"):
        file_suffix = ''
    else:
        file_suffix = '_' + '&'.join([f"{k}={v:.3f}" for k, v in eval_result.items()])

    file_name = f"{args.task}_{args.inference_name}_{args.context_key}{file_suffix}.jsonl"
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input)

    save_file_jsonl(lst_processed_data, os.path.join(args.output_dir, file_name))
    print(f"Saved {file_name} to {args.output_dir}")
    # dist.destroy_process_group()