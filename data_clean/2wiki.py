import pandas as pd


df_2wiki_eval = pd.read_json("eval_data/2wiki_dev.json")
df_2wiki_eval["ctxs"] = df_2wiki_eval["context"].apply(
    lambda ctxs: [{"title": ctx[0], "text": '...'.join(ctx[1])} for ctx in ctxs]
)
df_2wiki_eval.drop(columns=["context"], inplace=True)

df_2wiki_eval.to_json("./eval_data/2wiki_dev_processed.jsonl", orient="records", lines=True)

df_2wiki_test = pd.read_json("eval_data/2wiki_test.json")
df_2wiki_test["ctxs"] = df_2wiki_test["context"].apply(
    lambda ctxs: [{"title": ctx[0], "text": '...'.join(ctx[1])} for ctx in ctxs]
)
df_2wiki_test.drop(columns=["context"], inplace=True)

df_2wiki_test.to_json("./eval_data/2wiki_test_processed.jsonl", orient="records", lines=True)

df_2wiki_train = pd.read_json("train_data/2wiki_train.json")
df_2wiki_train["ctxs"] = df_2wiki_train["context"].apply(
    lambda ctxs: [{"title": ctx[0], "text": '...'.join(ctx[1])} for ctx in ctxs]
)
df_2wiki_train.drop(columns=["context"], inplace=True)

df_2wiki_train.to_json("./train_data/2wiki_train_processed.jsonl", orient="records", lines=True)