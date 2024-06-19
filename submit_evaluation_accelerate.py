import os
import argparse


NUM_GPUS = 1
BATCH_SIZE_PER_GPU = 1
BASE_MODEL_NAME_OR_PATH = "meta-llama/Llama-2-7b-chat-hf"

TASKS = [
    "popqa",
    "arc_c",
    "triviaqa",
    "hotpotqa_dev_distractor",
    "2wiki_dev",
]

downstream_model_names = (
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
)
downstream_inference_name=(
    "Llama_2_7b_chat",
    "Llama_2_13b_chat",
    "Llama_2_70b_chat",
    "Llama_3_70b",
    "Llama_3_8b",
)

openai_model_names = (
    "gpt-3.5-turbo-0301",
    # "gpt-4"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Get evaluation for train and evaluation task")

    parser.add_argument(
        "--adapter_name", type=str, help="name to directory containing PEFT adapter weights"
    )
    parser.add_argument(
        "--eval_dir", type=str, default=None, help="name to directory containing evaluate data"
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="name to directory containing evaluate data"
    )
    parser.add_argument(
        "--eval_baseline", action="store_true", help="whether to evaluate baseline"
    )
    parser.add_argument(
        "--eval_ablation", action="store_true", help="whether to evaluate ablation"
    )
    parser.add_argument(
        "--eval_refiner", action="store_true", help="whether to evaluate Refiner"
    )
    parser.add_argument(
        "--eval_downstream", action="store_true", help="whether to evaluate executor"
    )
    parser.add_argument(
        "--use_openai", action="store_true", help="when provided, evaluate using OpenAI model"
    )
    args = parser.parse_args()
    return args


def run_task(
        task,
        adapter,
        inference_name,
        top_n=10,
        use_openai=False,
        eval_dir=None,
        eval_refiner=False,
        eval_downstream=False,
        eval_baseline=False,
        eval_ablation=False):

    if eval_refiner:
        print(f"Evaluating {task} using {inference_name} adapter: {adapter}")
        os.system(f"""
accelerate launch \
    --main_process_port 29501 \
    --num_machines 1 \
    --num_processes {NUM_GPUS} \
./evaluation_accelerate.py \
    --per_gpu_eval_batch_size {BATCH_SIZE_PER_GPU} \
    --base_model_name_or_path {BASE_MODEL_NAME_OR_PATH} \
    --adapter_path ./checkpoint/{adapter} \
    --output_dir ./eval_data/{adapter}/top_{top_n} \
    --task {task} \
    --top_n {top_n} \
    --inference_name {inference_name}
    """)

    if eval_dir is None and "refiner" in adapter:
        eval_dir = f"./eval_data/top_{top_n}"
    if eval_dir is None:
        raise LookupError("Please specify eval data directory by --eval_dir")

    for file_name in os.listdir(eval_dir):
        if file_name.startswith(f"{task}_{inference_name}"):
            file_path = os.path.abspath(os.path.join(eval_dir, file_name))
            dir_path = os.path.dirname(file_path)
            print(f"Evaluating {file_path} using {adapter}")
            if use_openai:
                model_list = zip(openai_model_names, openai_model_names)
            else:
                model_list = zip(downstream_model_names, downstream_inference_name)

            for model, inference in model_list:
                if eval_downstream:
                    os.system(f"""
python ./get_executor_data.py \
--model_name_or_path {model} \
--per_gpu_eval_batch_size 12777 \
--task {task} \
--inference_name downstream_{inference} \
--context_key {inference_name} \
--input "{file_path}"
""")
                    if eval_baseline:
                        save_path = os.path.join(dir_path, "baseline")
                        os.makedirs(save_path, exist_ok=True)
                        os.system(f"""
python ./get_executor_data.py \
    --model_name_or_path {model} \
    --per_gpu_eval_batch_size 12777 \
    --task {task} \
    --inference_name "baseline_{inference}" \
    --context_key context \
    {"--use_openai" if use_openai else ''} \
    --input "{file_path}" \
    --output_dir "{save_path}" \
""")

                    if eval_ablation:
                        save_path = os.path.join(dir_path, "ablation")
                        os.makedirs(save_path, exist_ok=True)
                        for content_type in ["quote"]:
                            for sec_type in ["star", None]:
                                for title_type in ["origin", "quote", "md", None]:
                                    if sec_type == title_type == content_type:
                                        continue

                                    os.system(f"""
python ./get_executor_data.py \
    --model_name_or_path {model} \
    --per_gpu_eval_batch_size 12777 \
    --task {task} \
    --inference_name "ablation_{inference}" \
    --context_key {f'refiner_{sec_type}_section_{title_type}_title_{content_type}_content'} \
    --input "{file_path}" \
    --output_dir "{save_path}"
""")

            return

    print(f"Warning: executor {adapter} not evaluated in {task} task, maybe file not found in", eval_dir)


if __name__ == '__main__':
    args = parse_args()

    if args.eval_refiner:
        for task in TASKS:
            run_task(task=task,
                    adapter=args.adapter_name,
                    inference_name="refiner",
                    top_n=args.top_n,
                    eval_refiner=args.eval_refiner,
                    eval_dir=args.eval_dir)

    if args.eval_downstream or args.eval_baseline or args.eval_ablation:
        for task in TASKS:
            run_task(task=task,
                    adapter=args.adapter_name,
                    inference_name="refiner",
                    top_n=args.top_n,
                    use_openai=args.use_openai,
                    eval_dir=args.eval_dir,
                    eval_downstream=args.eval_downstream,
                    eval_baseline=args.eval_baseline,
                    eval_ablation=args.eval_ablation)

    # python ./submit_evaluation_accelerate.py --adapter_name al1231/Refiner-7B --top_n 10 --eval_baseline --eval_refiner

    # python ./submit_evaluation_accelerate.py --adapter_name refiner --use_openai --top_n 10 --eval_baseline