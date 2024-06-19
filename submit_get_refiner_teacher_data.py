import os
import argparse


# NUM_GPUS = 4

TASKS = [
    "arc_c_train",
    "triviaqa_train",
    "hotpotqa_train",
    "pubhealth_train",
]

downstream_model_names = (
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "/hpc2hdd/home/yxu409/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841",
    "Qwen/Qwen2-72B-Instruct"
)
downstream_inference_name=(
    "Llama_2_70b",
    "Llama_3_70b",
    "Llama_3_8b",
    "Mixtral_8x7B",
    "Qwen2_72B"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Get evaluation for train and evaluation task")

    parser.add_argument(
        "--train_dir", type=str, default=None, help="name to directory containing evaluate data"
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="name to directory containing evaluate data"
    )

    args = parser.parse_args()
    return args


def run_task(
        task,
        top_n=10,
        train_dir=None):

    if train_dir is None:
        train_dir = f"./train_data"

    matched_file_name = None
    for file_name in os.listdir(train_dir):
        if file_name.startswith(task):
            matched_file_name = file_name
        elif file_name == f"{task}_teacher_models.jsonl":
            matched_file_name = file_name
            print(f"continue recording with {file_name}")
            break

    if matched_file_name is None:
        raise IndexError(f"Warning: {task} task not evaluated, maybe file not found in", train_dir)

    file_path = os.path.abspath(os.path.join(train_dir, matched_file_name))
    output_path = os.path.abspath(os.path.join(train_dir, f"{task}_teacher_models.jsonl"))
    model_list = zip(downstream_model_names, downstream_inference_name)

    for model, inference in model_list:
        print(f"Evaluating {file_path} using {inference}")
        os.system(f"""
python ./get_refiner_teacher_data.py \
--model_name_or_path {model} \
--per_gpu_eval_batch_size 12777 \
--task {task} \
--top_n {top_n} \
--inference_name {inference} \
--input "{file_path}" \
--output "{output_path}"
""")


if __name__ == '__main__':
    args = parse_args()

    for task in TASKS:
        run_task(task=task,
                top_n=args.top_n,
                train_dir=args.train_dir)

    # python ./submit_get_refiner_teacher_data.py --top_n 10
