import pandas as pd


df_pubhealth_train = pd.read_csv("./train_data/pubhealth_train.tsv", sep="\t")

df_pubhealth_train["question"] = df_pubhealth_train["claim"]
df_pubhealth_train = df_pubhealth_train[~df_pubhealth_train["question"].isna() & ~df_pubhealth_train["claim_id"].isna()]

df_pubhealth_train["ctxs"] = df_pubhealth_train.apply(
    lambda df: [{
        "title": df["subjects"] if isinstance(df["subjects"], str) else "(No Title)",
        "text": df["main_text"]}]
    if isinstance(df["main_text"], str)
    else [], axis=1)
df_pubhealth_train["answer"] = df_pubhealth_train["label"]
df_pubhealth_train["id"] = df_pubhealth_train["claim_id"]
df_pubhealth_train[["id", "question", "ctxs", "answer"]].to_json("./train_data/pubhealth_train_processed.jsonl", orient="records", lines=True)


df_pubhealth_train_expunged = pd.read_json("./train_data/pubhealth_Meta-Llama-3-70B.jsonl", lines=True)
df_pubhealth_train_expunged['answer'].unique()

df_pubhealth_train_expunged = df_pubhealth_train_expunged[
    df_pubhealth_train_expunged["answer"].isin(["false", 'mixture', 'true'])
]
df_pubhealth_train_expunged.to_json("./train_data/llama3_truncated/pubhealth_monitor_teacher_expunge.jsonl")