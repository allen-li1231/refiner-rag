import pandas as pd
from utils import save_file_jsonl


df_asqa = pd.read_json("./eval_data/asqa_eval_gtr_top100.json")
df_asqa = df_asqa.rename(columns={"docs": "ctxs"})
df_asqa[["question", "ctxs", "answer"]].to_json("./eval_data/asqa_processed.json", "records")