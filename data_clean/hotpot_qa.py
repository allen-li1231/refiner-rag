import pandas as pd


df_hotpot = pd.read_json("./train_data/hotpot_train_v1.1.json")
df_hotpot["ctxs"] = df_hotpot["context"].apply(lambda lst_ctx: [{"title": ctx[0], "text": '...'.join(ctx[1])} for ctx in lst_ctx])
df_hotpot[["question", "ctxs", "answer"]].to_json("./train_data/hotpotqa_train_processed.json", orient="records")

df_hotpot = pd.read_json("./eval_data/hotpot_dev_distractor_v1.json")
df_hotpot["ctxs"] = df_hotpot["context"].apply(lambda lst_ctx: [{"title": ctx[0], "text": '...'.join(ctx[1])} for ctx in lst_ctx])
df_hotpot[["question", "ctxs", "answer"]].to_json("./eval_data/hotpotqa_dev_distractor_processed.json", orient="records")

df_hotpot = pd.read_json("./eval_data/hotpot_dev_fullwiki_v1.json")
df_hotpot["ctxs"] = df_hotpot["context"].apply(lambda lst_ctx: [{"title": ctx[0], "text": '...'.join(ctx[1])} for ctx in lst_ctx])
df_hotpot[["question", "ctxs", "answer"]].to_json("./eval_data/hotpotqa_dev_fullwiki_processed.json", orient="records")

df_hotpot = pd.read_json("./eval_data/hotpot_test_fullwiki_v1.json")
df_hotpot["ctxs"] = df_hotpot["context"].apply(lambda lst_ctx: [{"title": ctx[0], "text": '...'.join(ctx[1])} for ctx in lst_ctx])
df_hotpot[["question", "ctxs"]].to_json("./eval_data/hotpotqa_test_processed.json", orient="records")
