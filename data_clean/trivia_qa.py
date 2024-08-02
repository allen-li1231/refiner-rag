import pandas as pd
from model.utils import save_file_jsonl


df_trivia = pd.read_parquet("../train_data/trivia_qa")
save_file_jsonl(df_trivia["question"].to_list(), "../train_data/triviaqa_train_question.jsonl")

# TODO: retrieve passage, leave records where retrieved info or answers are included in the retrieved passages.

df_trivia_extract = pd.read_json("../train_data/triviaqa_train_retrieve.jsonl")