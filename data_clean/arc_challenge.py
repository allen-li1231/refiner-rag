import pandas as pd
from utils import process_arc_instruction, TASK_INST, load_file

df_arc_c = pd.read_parquet("./train_data/arc_challenge_train.parquet")
lst_arc_c_questions = df_arc_c.apply(lambda df: process_arc_instruction(df, TASK_INST["arc_c"]), axis=1)
len(lst_arc_c_questions)

df_extract = pd.read_json("./train_data/arc_c_extract_expunge.jsonl",lines=True)
df_extract["instruction"] = df_extract["question"]
df_extract["question"] = df_arc_c["question"]
df_extract["choices"] = df_arc_c["choices"]
df_extract["answerKey"] = df_arc_c["answerKey"]
df_extract.to_json("./train_data/arc_c_extract_expunge.json", "records")
