import random
from tqdm.auto import tqdm
import pandas as pd


def extract_contexts(paragraphs, top_n=10):
    assert len(paragraphs) >= 10, f"number of paragraph insufficient from which to retrieve top {top_n} contexts."
    lst_contexts = []
    for context in paragraphs:
        if context["is_supporting"]:
            lst_contexts.append(context)

    for _ in range(len(lst_contexts), top_n):
        if not context["is_supporting"]:
            lst_contexts.append(context)

    random.shuffle(lst_contexts)
    return lst_contexts


if __name__ == '__main__':
    tqdm.pandas(desc="Applying")

    for file_path in ("eval_data/musique_ans_v1.0_dev.jsonl",
                    #   "eval_data/musique_ans_v1.0_test.jsonl",
                      "train_data/musique_ans_v1.0_train.jsonl"):
        df_musique: pd.DataFrame = pd.read_json(file_path, lines=True, orient="records")
        df_musique["ctxs"] = df_musique["paragraphs"].progress_apply(extract_contexts)
        df_musique.to_json(f"{file_path.replace('_ans_v1.0', '').replace('.jsonl', '_processed.jsonl')}",
                           lines=True, orient="records")