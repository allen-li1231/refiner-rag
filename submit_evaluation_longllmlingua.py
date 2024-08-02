import os
import time
import argparse


TASKS = [
    # "popqa",
    # "arc_c",
    "triviaqa",
    "hotpotqa_dev_distractor",
    # "2wiki_dev",
]

executor_model_names=(
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    # "meta-llama/Llama-2-70b-chat-hf",
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "meta-llama/Meta-Llama-3-8B-Instruct",
)
executor_inference_name=(
    "Llama_2_7b_chat",
    "Llama_2_13b_chat",
    # "Llama_2_70b_chat",
    # "Llama_3_70b",
    # "Llama_3_8b",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Get evluation for train and evaluation task")

    parser.add_argument(
        "--eval_dir", type=str, default=None, help="name to directory containing evaluate data"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="name to output directory."
    )
    parser.add_argument(
        "--eval_refiner", action="store_true", help="whether to evaluate monitor"
    )
    parser.add_argument(
        "--eval_downstream", action="store_true", help="whether to evaluate executor"
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="name to directory containing evaluate data"
    )
    parser.add_argument(
        "--inference_name", type=str, help="whether to evaluate monitor"
    )
    parser.add_argument(
        "--rate", default=0.5, type=float, help="Compress rate in paper."
    )
    parser.add_argument(
        "--dynamic_context_compression_ratio", default=0.25, type=float, help="Dynamic context compression ratio in paper."
    )
    args = parser.parse_args()
    return args


def run_task(
        task,
        inference_name,
        top_n=10,
        rate=0.5,
        dynamic_context_compression_ratio=0.25,
        eval_refiner=True,
        eval_downstream=True,
        eval_dir=None,
        output_dir=None
        ):

        if eval_dir is None:
            eval_dir = f"./eval_data/top_{top_n}"
        if output_dir is None:
            output_dir = f"./eval_data/{inference_name}/top_{top_n}/"

        os.makedirs(output_dir, exist_ok=True)
        is_evaluated = False
        if eval_refiner:
            for file_name in os.listdir(eval_dir):
                if file_name.startswith(task):
                    file_path = os.path.abspath(os.path.join(eval_dir, file_name))
                    print(f"Evaluating {file_path} using {inference_name}")
                    start = time.time()
                    os.system(f"""
accelerate launch \
    --main_process_port 29501 \
    --num_machines 1 \
    --num_processes 4 \
./llmlingua_accelerate.py \
    --task {task} \
    --inference_name {inference_name} \
    --top_n {top_n} \
    --rate {rate} \
    --dynamic_context_compression_ratio {dynamic_context_compression_ratio} \
    --input "{file_path}" \
    --output_dir "{output_dir}"
    """)
                    print(f'Complete Time: {time.time() - start}')
                    is_evaluated = True
                    break
            if not is_evaluated:
                print(f"Warning: monitor {inference_name} not evaluated in {task} task, maybe file not found in", eval_dir)

        if eval_downstream:
            for file_name in os.listdir(output_dir):
                if file_name.startswith(f"{task}_{inference_name}"):
                    file_path = os.path.abspath(os.path.join(output_dir, file_name))
                    print(f"Evaluating {file_path} using {inference_name}")

                    for model, inference in zip(executor_model_names, executor_inference_name):
                        os.system(f"""
python ./get_executor_data.py \
    --model_name_or_path {model} \
    --per_gpu_eval_batch_size 12777 \
    --task {task} \
    --inference_name "executor_{inference}" \
    --context_key {inference_name} \
    --input "{file_path}" \
    --output_dir "{output_dir}" \
""")
                    return

            print(f"Warning: executor {inference_name} not evaluated in {task} task, maybe file not found in", eval_dir)


if __name__ == '__main__':
    args = parse_args()

    for task in TASKS:
        run_task(task=task,
                inference_name=args.inference_name,
                top_n=args.top_n,
                rate=args.rate,
                dynamic_context_compression_ratio=args.dynamic_context_compression_ratio,
                eval_refiner=args.eval_refiner,
                eval_downstream=args.eval_downstream,
                eval_dir=args.eval_dir,
                output_dir=args.output_dir)


    # python ./submit_evaluation_longllmlingua.py --top_n 5 --inference_name longllmlingua --eval_refiner --rate 0.5 --dynamic_context_compression_ratio 0.3 >> longllmlingua_top5.log
    # python ./submit_evaluation_longllmlingua.py --top_n 10 --inference_name longllmlingua --eval_refiner --rate 0.5 --dynamic_context_compression_ratio 0.3 >> longllmlingua_top10.log