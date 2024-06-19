import os
import argparse


NUM_GPUS = 4
BATCH_SIZE_PER_GPU = 1

TASKS = [
    "popqa",
    "arc_c",
    "triviaqa",
    "hotpotqa_dev_distractor",
    # "fever",
    "2wiki_dev",
    # "asqa",
    # "factscore",
    # "hotpotqa_dev_fullwiki",
]

executor_model_names=(
    # "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-2-13b-hf",
    # "meta-llama/Llama-2-70b-hf",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "meta-llama/Llama-2-70b-chat-hf",
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "/hpc2hdd/home/yxu409/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298/",
    "/hpc2hdd/home/yxu409/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841/",
    "/hpc2hdd/home/yxu409/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/41c66b0be1c3081f13defc6bdf946c2ef240d6a6/"
)
executor_inference_name=(
    # "Llama_2_7b",
    # "Llama_2_13b",
    # "Llama_2_70b",
    # "Llama_2_7b_chat",
    # "Llama_2_13b_chat",
    # "Llama_2_70b_chat",
    # "Llama_3_70b",
    # "Llama_3_8b",
    "Mixtral_8x7B",
    "Qwen_2_7B"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Get evluation for train and evaluation task")

    parser.add_argument(
        "--eval_dir", type=str, default=None, help="name to directory containing evaluate data"
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="name to directory containing evaluate data"
    )
    parser.add_argument(
        "--eval_monitor", action="store_true", help="whether to evaluate monitor"
    )
    parser.add_argument(
        "--eval_executor", action="store_true", help="whether to evaluate executor"
    )
    args = parser.parse_args()
    return args


def run_task(
        task,
        inference_name,
        top_n=10,
        eval_dir=None,
        eval_monitor=True,
        eval_executor=True):

    if eval_monitor:
        print(f"Evaluating {task} using {inference_name}")
        os.system(f"""
accelerate launch \
    --main_process_port 29501 \
    --num_machines 1 \
    --num_processes {NUM_GPUS} \
./evaluation_accelerate.py \
    --per_gpu_eval_batch_size {BATCH_SIZE_PER_GPU} \
    --base_model_name_or_path {BASE_MODEL_NAME_OR_PATH} \
    --output_dir ../eval_data/{inference_name}/top_{top_n} \
    --task {task} \
    --top_n {top_n} \
    --inference_name {inference_name}
    """)

    if eval_executor:
        if eval_dir is None:
            eval_dir = f"../eval_data/{inference_name}/top_{top_n}"

        for file_name in os.listdir(eval_dir):
            if file_name.startswith(f"{task}_{inference_name}"):
                file_path = os.path.abspath(os.path.join(eval_dir, file_name))
                print(f"Evaluating {file_path} using {inference_name}")
                model_list = zip(executor_model_names, executor_inference_name)

                for model, inference in model_list:
                    if eval_executor:
                        os.system(f"""
    python ./get_executor_data.py \
    --model_name_or_path {model} \
    --per_gpu_eval_batch_size 12777 \
    --task {task} \
    --inference_name executor_{inference} \
    --context_key {inference_name} \
    --input "{file_path}"
    """)
                return

        print(f"Warning: executor {inference_name} not evaluated in {task} task, maybe file not found in", eval_dir)

            
if __name__ == '__main__':
    args = parse_args()

    for task in TASKS:
        if task in ("popqa", "triviaqa", "arc_c"):
            BASE_MODEL_NAME_OR_PATH = "fangyuan/tqa_abstractive_compressor"
        else:
            BASE_MODEL_NAME_OR_PATH = "fangyuan/hotpotqa_abstractive"

        run_task(task=task,
                inference_name="recomp_abstractive",
                top_n=args.top_n,
                eval_monitor=args.eval_monitor,
                eval_executor=args.eval_executor,
                eval_dir=args.eval_dir)


    # python ./submit_evaluation_recomp.py --top_n 10 --eval_executor