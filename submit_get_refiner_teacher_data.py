import os
import argparse

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# NUM_GPUS = 4

TASKS = [
    "musique_train",
    "arc_c_train",
    "triviaqa_train",
    "hotpotqa_train",
    "pubhealth_train",
]

downstream_model_names = (
    "Qwen/Qwen2-72B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "dnhkng/RYS-XLarge",
)
downstream_inference_name = (
    "Qwen2_72B",
    "Llama_3.1_70b",
    "Llama_2_70b",
    "Llama_3_70b",
    "RYS_XLarge",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Get evaluation for train and evaluation task")

    parser.add_argument(
        "--train_dir", type=str, default=None, help="name to directory containing training data"
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="number of top retrieval"
    )

    args = parser.parse_args()
    return args


def run_task(
        task,
        top_n=10,
        train_dir=None):

    if train_dir is None:
        train_dir = f"./train_data/"

    matched_file_name = None
    for file_name in os.listdir(train_dir):
        if file_name.startswith(task):
            matched_file_name = file_name
        if file_name == f"{task}_teacher_models.jsonl":
            print(f"Continue recording with {file_name}")
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
    --per_gpu_eval_batch_size 12765 \
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
