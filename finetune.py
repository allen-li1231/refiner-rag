# coding=utf-8

import os
import copy
import math
import logging
import argparse
from functools import partial

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, is_deepspeed_available

import torch
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, concatenate_datasets
import evaluate

from tqdm.auto import tqdm
from typing import Optional, Union, List, Dict, Sequence

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM
)
from peft import (LoraConfig, PrefixTuningConfig, TaskType,
                  PeftModel, PeftModelForCausalLM, get_peft_model
)
from utils import PROMPT_DICT, TASK_INST, process_retriever_passage

logger = get_logger(__name__)

PROMPT_REFINER_LORA_INPUT = PROMPT_DICT["prompt_refiner_lora"]
PROMPT_REFINER_PREFIX_INPUT = PROMPT_DICT["prompt_refiner_prefix"]
PROMPT_EXECUTOR_LORA_INPUT = PROMPT_DICT["prompt_downstream_lora"]
PROMPT_EXECUTOR_PREFIX_INPUT = PROMPT_DICT["prompt_downstream_prefix"]


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["monitor", "executor", "monitor+executor"],
        help=f"Task of text extraction from evaluate dataset. Choices: 'monitor' or 'executor'.",
        required=False,
        default=None
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=10,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_prefix_tuning",
        action="store_true",
        help="If passed, will use prefix-tuning to train the model.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--resume_from_adapter",
        type=str,
        default=None,
        help="If the training should continue from a PEFT adapter folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--use_special_tokens",
        action="store_true",
        help=(
            "Use special tokens."
        ),
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to split from dataset for evaluation"
    )
    parser.add_argument(
        "--train_executor_file", type=str, default=None, help="A csv or a json file containing the training data for executor."
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args


def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int) -> Dict:
    """Tokenize a list of strings."""
    input_ids = labels = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_seq_length,
            truncation=True,
    ).input_ids
    input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()
    # print(input_ids_lens)

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def encode_with_prompt_completion_format(
        example,
        tokenizer,
        max_seq_length,
        n_docs,
        target_key,
        context_markups=None,
        prompt_template=PROMPT_REFINER_LORA_INPUT
    ):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    question, context = process_retriever_passage(example, n_docs=n_docs)
    source_text = prompt_template.format(question=question, context=context)

    target_text = example[target_key].rstrip(tokenizer.eos_token) + tokenizer.eos_token
    examples_tokenized = _tokenize_fn(source_text + target_text, tokenizer, max_seq_length)
    sources_tokenized = _tokenize_fn(source_text, tokenizer, max_seq_length)

    # ensure that cross entropy ignores source_text
    input_ids = examples_tokenized["input_ids"].flatten()
    source_len = sources_tokenized["input_ids_lens"]
    labels = copy.deepcopy(input_ids)
    labels[ :source_len-1] = -100

    if context_markups is not None:
        context_start = False
        for j, orig_token in enumerate(labels[source_len:]):
            if context_start is False and orig_token == context_markups[0]:
                context_start = True
                assert labels[source_len+j] == context_markups[0]
                start_idx = j+source_len
                end_idx = None
                for k, orig_token_2 in enumerate(labels[start_idx:]):
                    if orig_token_2 == context_markups[1]:
                        end_idx = start_idx + k
                if end_idx is None:
                    end_idx =  start_idx + k
                else:
                    assert labels[end_idx] == context_markups[1]
                labels[start_idx+1:end_idx] = -100
                context_start = False
    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten()
    }


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)


def save_accelerate_model(model, accelerator, output_dir):
    accelerator.save_state(output_dir)
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if isinstance(unwrapped_model, PeftModel):
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to mannually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )


def main():
    args = parse_args()

    # A hacky way to make llama work with flash attention and torch < 2.1
    # if args.use_flash_attn:
    #     from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    #     replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # if isinstance(args.start, int):
    #     if isinstance(args.end, int):
    #         slicing = range(args.start, args.end)
    #     else:
    #         slicing = range(args.start, len(raw_datasets))
    #     raw_datasets.select(slicing)
    # elif isinstance(args.end, int):
    #     slicing = range(args.end)
    #     from datasets import DatasetDict
    #     DatasetDict().
    #     raw_datasets.select(slicing)

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, padding_side="left")
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side="left")
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
            torch_dtype=torch.bfloat16
            # local_files_only=True
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)


    # no default pad token for llama
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        if args.use_special_tokens is True:
            if args.task == "monitor":
                special_token_dict = {# "pad_token": "<pad>",
                                      "additional_special_tokens": ["[MONITOR]"]}
            elif args.task == "executor":
                special_token_dict = {# "pad_token": "<pad>",
                                      "additional_special_tokens": ["[EXECUTOR]"]}
            elif args.task == "monitor+executor":
                special_token_dict = {# "pad_token": "<pad>",
                                      "additional_special_tokens": ["[MONITOR]",
                                                                    "[EXECUTOR]"]}
        else:
            special_token_dict = {}
        num_added_tokens = tokenizer.add_special_tokens(special_token_dict)

        if args.task == "executor":
            context_markups = [tokenizer.convert_tokens_to_ids(token) for token in ["<paragraph>", "</paragraph>"]]
        else:
            context_markups = None

        if args.use_special_tokens is False:
            assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        elif args.task == "monitor" and isinstance(args.model_name_or_path, str) and args.tokenizer_name is None:
            assert num_added_tokens == 1, "To train monitor, special tokens must be added to the original tokenizers."
        elif args.task == "executor" and isinstance(args.model_name_or_path, str) and args.tokenizer_name is None:
            assert num_added_tokens == 1, "To train executor, special tokens must be added to the original tokenizers."
        elif args.task == "monitor+executor" and isinstance(args.model_name_or_path, str) and args.tokenizer_name is None:
            assert num_added_tokens == 2, "To train monitor+executor, special tokens must be added to the original tokenizers."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    tokenizer.pad_token = tokenizer.eos_token \
                          if not isinstance(tokenizer.pad_token, str) \
                          else tokenizer.pad_token

    # resize the embeddings to vocab size
    embeddings = model.get_input_embeddings()
    if is_deepspeed_available():
        import deepspeed
        with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
            embedding_size = embeddings.weight.shape[0]
    else:
        embedding_size = embeddings.weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        accelerator.print(f"Embedding size {embedding_size} -> {len(tokenizer)}")

    if isinstance(args.resume_from_adapter, str):
        accelerator.print(f"Resuming from PEFT checkpoint: {args.resume_from_adapter}")
        model = PeftModelForCausalLM.from_pretrained(model, args.resume_from_adapter, is_trainable=True)
    elif args.use_lora:
        accelerator.print("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            #modules_to_save=modules_to_save,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
    elif args.use_prefix_tuning:
        accelerator.print("Initializing prefix tuning model...")
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    if accelerator.mixed_precision == "fp16":
        cast_training_params(model, dtype=torch.float16)

    model.print_trainable_parameters()


    # dataset initialize
    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        n_docs=args.n_docs,
        target_key="exemplar" if args.task == "executor" else "output",
        context_markups=context_markups if args.use_special_tokens is True else None,
        prompt_template=PROMPT_REFINER_PREFIX_INPUT if args.use_prefix_tuning else PROMPT_REFINER_LORA_INPUT
    )
    # elif "messages" in raw_datasets["train"].column_names:
    #     encode_function = partial(
    #         encode_with_messages_format,
    #         tokenizer=tokenizer,
    #         max_seq_length=args.max_seq_length,
    #     )
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names
                            if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: example['labels'].ne(-100).any())
        train_dataset = lm_datasets["train"]
        
        if args.task == "monitor+executor" and isinstance(args.train_executor_file, str):
            additional_raw_datasets = load_dataset(
                "json",
                data_files={"train": args.train_executor_file},
                **dataset_args,
            )

            encode_function = partial(
                encode_with_prompt_completion_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                target_key="executor_teacher",
                context_markups=context_markups if args.use_special_tokens is True else None,
                prompt_template=PROMPT_EXECUTOR_PREFIX_INPUT if args.use_prefix_tuning else PROMPT_EXECUTOR_LORA_INPUT
            )
            lm_datasets = additional_raw_datasets.map(
                encode_function,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[name for name in additional_raw_datasets["train"].column_names
                                if name not in ["input_ids", "labels", "attention_mask"]],
                desc="Tokenizing and reformatting executor data",
            )
            lm_datasets.set_format(type="pt")
            lm_datasets = lm_datasets.filter(lambda example: example['labels'].ne(-100).any())
            additional_train_dataset = lm_datasets["train"]

            train_dataset = concatenate_datasets([train_dataset, additional_train_dataset], axis=0)

    if args.do_eval:
        train_dataset, eval_dataset = (train_dataset
                                       .train_test_split(test_size=0.1, seed=args.seed)
                                       .values())

    accelerator.print("Sample of encoded train dataset:", train_dataset[0])
    # with open("processed.json", "w") as outfile:
    #     new_data = []
    #     for item in train_dataset:
    #         print(item)
    #         labels = [int(i) for i in item["labels"]]
    #         input_ids = [int(i) for i in item["input_ids"]]
    #         new_data.append({"labels": labels, "input_ids": input_ids})
    #     json.dump(new_data, outfile)
    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                          padding="longest" if args.per_device_train_batch_size > 1 else False),
        batch_size=args.per_device_train_batch_size
    )

    if args.do_eval:
        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                              padding="longest" if args.per_device_train_batch_size > 1 else False),
            batch_size=args.per_device_train_batch_size
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.do_eval:
        eval_dataloader = accelerator.prepare(eval_dataloader)

        eval_metrics = [
            evaluate.load(
                "exact_match",
                num_process=accelerator.num_processes,
                process_id=accelerator.process_index
            ),
            evaluate.load(
                "src/rouge.py",
                num_process=accelerator.num_processes,
                process_id=accelerator.process_index
            ),
        ]

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(f"Loading checkpoint...")
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        
        accelerator.print(f"Resumed from accelerate checkpoint: {path}")
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.lstrip("epoch_")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.lstrip("step_"))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = starting_epoch * num_update_steps_per_epoch
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0.
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and completed_steps <= resume_step:
                    progress_bar.update(1)
                    completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.logging_steps and (completed_steps + 1) % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    lr = lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({"loss": avg_loss, "lr": lr}, refresh=False)
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr,
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0.

                if isinstance(checkpointing_steps, int):
                    if (completed_steps + 1) % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_accelerate_model(model, accelerator=accelerator, output_dir=output_dir)

                if completed_steps >= args.max_train_steps:
                    break

                # empty cache to alleviate OOM and warnings
                if not is_deepspeed_available():
                    getattr(torch, accelerator.device.type).empty_cache()
                else:
                    from deepspeed import get_accelerator
                    get_accelerator().empty_cache()

                progress_bar.update(1)
                completed_steps += 1

        if args.do_eval:
            model.eval()
            for batch in tqdm(eval_dataloader, desc="Evaluate"):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                with torch.no_grad():
                    outputs = model(**batch)

                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                for metric in eval_metrics:
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

            eval_result = dict()
            for metric in eval_metrics:
                eval_result.update(metric.compute())

            if args.with_tracking:
                accelerator.log(
                    eval_result,
                    step=completed_steps,
                )
            progress_bar.set_postfix({"loss": avg_loss, "lr": lr, **eval_result})

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)

            save_accelerate_model(model, accelerator=accelerator, output_dir=output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(save_directory=args.output_dir)
        
        save_accelerate_model(model, accelerator=accelerator, output_dir=args.output_dir)


if __name__ == "__main__":
    # --train_file ../eval_data/factscore_extract_expunge.jsonl --model_name_or_path meta-llama/Llama-2-7b-chat-hf --use_lora --warmup_ratio 0.01 --lr_scheduler_type "cosine" --output_dir ./LLM --seed 430 --logging_steps 10 --use_special_tokens --learning_rate 2e-5 --gradient_accumulation_steps 8 --per_device_train_batch_size 1 --end 1000
    main()
