import time
from tqdm.auto import tqdm
import pandas as pd
from metrics import calc_acc
from get_refiner_teacher_data import openai_client, api_generate, assemble_conversation

tqdm.pandas()

teacher_model = "gpt-4-turbo"

def get_teacher_output(df, strict=False):
    question, context = df["question"], df["context"]
    message = assemble_conversation(question, context, strict=strict)
    try:
        ans = api_generate(openai_client, message, teacher_model, 2048)
        time.sleep(1)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        time.sleep(60)
        ans = api_generate(openai_client, message, teacher_model, 2048)

    return ans

dataset_type = "llama3_truncated"
wiki_expunge = pd.read_json(f"../train_data/{dataset_type}/wiki_monitor_teacher_expunge.jsonl", lines=True)
hotpot_expunge = pd.read_json(f"../train_data/{dataset_type}/hotpotqa_monitor_teacher_expunge.jsonl", lines=True)
arc_c_expunge = pd.read_json(f"../train_data/{dataset_type}/arc_c_monitor_teacher_expunge.jsonl", lines=True)
trivia = pd.read_parquet("../train_data/trivia_qa")
trivia_expunge = pd.read_json(f"../train_data/{dataset_type}/triviaqa_monitor_teacher_expunge.jsonl", lines=True, orient="records")
pubhealth_expunge = pd.read_json(f"../train_data/{dataset_type}/pubhealth_monitor_teacher_expunge.jsonl")

# EDA
# HotpotQA
hotpot_expunge["context_match"] = hotpot_expunge.apply(lambda df: calc_acc(df, df["context"]) , axis=1)
hotpot_expunge["output_match"] = hotpot_expunge.apply(lambda df: calc_acc(df, df["output"]) , axis=1)

hotpot_expunge["context_match"].mean()
# 0.9270954260506152
hotpot_expunge["output_match"].mean()
# 0.8035976870432407

# senario 1: gold in context but not in output
df_mismatch = hotpot_expunge[hotpot_expunge["context_match"] & ~hotpot_expunge["output_match"]]
df_mismatch.to_json("../train_data/hotpot_mismatch.jsonl", "records", lines=True)

i = 11
print(df_mismatch.loc[i, "question"])
print("Answers:", df_mismatch.loc[i, "answer"])
print("Misatched:", df_mismatch.loc[i, "output"])
print("Matched:", df_mismatch.loc[i, "context"])

# solution1: get GPT-4 output as teacher
hotpot_expunge.loc[hotpot_expunge["context_match"] & ~hotpot_expunge["output_match"], teacher_model] = \
    hotpot_expunge[hotpot_expunge["context_match"] & ~hotpot_expunge["output_match"]].progress_apply(get_teacher_output, axis=1)

# check teacher's performance
df_hotpot_teacher = hotpot_expunge[~hotpot_expunge[teacher_model].isna()]
df_hotpot_teacher["teacher_match"] = df_hotpot_teacher.apply(lambda df: calc_acc(df, df[teacher_model]) , axis=1)

# senario 2: gold in output but not in context
df_mismatch = df_hotpot_teacher[~df_hotpot_teacher["context_match"] & df_hotpot_teacher["output_match"]]

# easy solution: remove all abnormal records
hotpot_expunge = hotpot_expunge[
    (hotpot_expunge["context_match"] & hotpot_expunge["output_match"])
    | (~hotpot_expunge["context_match"] & ~hotpot_expunge["output_match"])]


# arc_c
arc_c_expunge["question"] = arc_c_expunge["question"].str.replace("Given answer candidates, choose the best answer choice.\n", '')
arc_c_expunge["context_match"] = arc_c_expunge.apply(lambda df: calc_acc(df, df["context"]) , axis=1)
arc_c_expunge["output_match"] = arc_c_expunge.apply(lambda df: calc_acc(df, df["output"]) , axis=1)

arc_c_expunge["context_match"].mean()
# 0.26138433515482695
arc_c_expunge["output_match"].mean()
# 0.2731524789522919

i = 617
print(arc_c_expunge.loc[i, "question"])
print("Answers:", arc_c_expunge.loc[i, "answerKey"])
print("Misatched:", arc_c_expunge.loc[i, "output"])
print("Matched:", arc_c_expunge.loc[i, "context"])

arc_c_expunge.loc[arc_c_expunge["context_match"] & ~arc_c_expunge["output_match"], teacher_model] = \
    arc_c_expunge[arc_c_expunge["context_match"] & ~arc_c_expunge["output_match"]].progress_apply(lambda df: get_teacher_output(df, strict=True), axis=1)


# TriviaQA
trivia["answers"] = trivia["answer"].apply(lambda d: d["aliases"])
trivia_expunge = pd.merge(trivia_expunge, trivia[["question", "answers"]], on="question")

trivia_expunge["context_match"] = trivia_expunge.apply(lambda df: calc_acc(df, df["context"]) , axis=1)
trivia_expunge["output_match"] = trivia_expunge.apply(lambda df: calc_acc(df, df["output"]) , axis=1)

trivia_expunge["context_match"].mean()
# 0.8806392192347466    0.8535418821096173
trivia_expunge["output_match"].mean()
# 0.8105286970010341    0.7831081954498449

# senario 1: gold in context but not in output
df_mismatch = trivia_expunge[trivia_expunge["context_match"] & ~trivia_expunge["output_match"]]

i = 61772
print(df_mismatch.loc[i, "question"])
print("Answers:", df_mismatch.loc[i, "answers"])
print("Misatched:", df_mismatch.loc[i, "output"])
print("Matched:", df_mismatch.loc[i, "context"])

# solution: try asking for GPT
trivia_expunge.loc[trivia_expunge["context_match"] & ~trivia_expunge["output_match"], teacher_model] = \
    trivia_expunge[trivia_expunge["context_match"] & ~trivia_expunge["output_match"]].progress_apply(get_teacher_output, axis=1)

# senario 2: gold in output but not in context
df_mismatch = trivia_expunge[~trivia_expunge["context_match"] & trivia_expunge["output_match"]]

i = 61332
print(df_mismatch.loc[i, "question"], df_mismatch.loc[i, "answers"])
print("Matched:", df_mismatch.loc[i, "output"])
print("Mismatched:", df_mismatch.loc[i, "context"])

# solution: set them to empty strings
trivia_expunge["output"][~trivia_expunge["context_match"] & trivia_expunge["output_match"]] = ""

# easy solution: remove all abnormal records
trivia_expunge = trivia_expunge[
    (trivia_expunge["context_match"] & trivia_expunge["output_match"])
    | (~trivia_expunge["context_match"] & ~trivia_expunge["output_match"])]


# Clean into trainable dataset
train_dataset_columns = ["question", "context", "output"]
hotpot_expunge = hotpot_expunge[train_dataset_columns]
arc_c_expunge = arc_c_expunge[train_dataset_columns]
trivia_expunge = trivia_expunge[train_dataset_columns]
pubhealth_expunge = pubhealth_expunge[train_dataset_columns]

hotpot_expunge.shape
# (79157, 3)
arc_c_expunge.shape
# (1119, 3)
trivia_expunge.shape
# (56655, 3)
pubhealth_expunge.shape
# (9513, 3)

train_data = pd.concat([
    # wiki_expunge,
    hotpot_expunge,
    # arc_c_expunge,
    # arc_c_expunge,
    trivia_expunge
], axis=0)
train_data.to_json("../train_data/llama3_truncated/arc_c_hotpotqa_triviaqa_truncated.jsonl", orient="records", lines=True)
train_data = pd.concat([
    pubhealth_expunge,
    arc_c_expunge
], axis=0)
train_data.to_json("../train_data/llama3_truncated/arc_c_pubhealth_truncated.jsonl", orient="records", lines=True)
pubhealth_expunge.to_json("../train_data/llama3_truncated/pubhealth.jsonl", orient="records", lines=True)

train_executor_llama2_70b = pd.read_json("../train_data/monitor_from_llama3_truncated_no_special_token/train_truncated_executor_teacher_Llama_2_70b_monitor.jsonl", orient="records", lines=True)
train_executor_llama3_70b = pd.read_json("../train_data/monitor_from_llama3_truncated_no_special_token/train_truncated_executor_teacher_Llama_3_70b_monitor.jsonl", orient="records", lines=True)

train_executor_llama2_70b.head()
train_executor_llama3_70b.head()