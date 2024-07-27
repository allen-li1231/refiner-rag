import re
import pylcs
import pandas as pd
from tqdm.auto import tqdm
from utils import regex_section
from submit_get_refiner_teacher_data import downstream_inference_name

tqdm.pandas(desc="Applying")


def common_str_idx(s1: str, s2: str):
    return pylcs.lcs_string_idx(s1, s2)


def parse_quotes(teacher_answer: str):
    lst_teacher_quotes = teacher_answer.split('\n\n')
    if re.match(regex_section, lst_teacher_quotes[-1]) is None:
        teacher_answer = '\n\n'.join(lst_teacher_quotes[:-1])

    lst_quotes = re.findall(regex_section, teacher_answer.rstrip("</s>").strip())
    if len(lst_quotes) == 0 or isinstance(lst_quotes[0], str) and len(lst_quotes[0]) == 0:
        # raise LookupError("Cannot extract quotes from:", context)
        # print("---\nCannot extract quotes from:", teacher_answer)
        return teacher_answer

    return lst_quotes


def vote_quotes(lst_context: list, lst_quotes: list, voter_name: str, min_valid_str_len=3):
    if len(lst_quotes) == 0 or isinstance(lst_quotes[0], str) and len(lst_quotes[0]) == 0:
        return

    for d_context in lst_context:
        if "ballot" not in d_context:
            d_context["ballot"] = {}

        for section, _, _, title, _, quote in lst_quotes:
            if len(title) < min_valid_str_len \
                or len(quote) < min_valid_str_len \
                or title not in d_context["title"]:
                continue

            lst_common_idx = common_str_idx(d_context["text"], quote)
            current_section = int(section.split('.')[0])

            for i, idx in enumerate(lst_common_idx):
                if idx == -1:
                    continue

                if i not in d_context["ballot"]:
                    d_context["ballot"][i] = {voter_name: current_section}
                else:
                    d_context["ballot"][i][voter_name] = current_section


def extract_major_quote(text, ballots, min_context_votes, return_voters=False):
    ballots = dict(sorted(ballots.items()))

    i_context_start, i_context_end, max_voters = -1, -1, None
    for i, ballot in ballots.items():
        if max_voters is None and len(ballot) >= min_context_votes:
            i_context_start = i
            max_voters = ballot
        elif max_voters is not None and len(max_voters) < len(ballot):
            max_voters = ballot
    
    if i_context_start == -1:
        return '', max_voters if return_voters else ''

    for i, ballot in reversed(ballots.items()):
        if len(ballot) >= min_context_votes:
            i_context_end = i
            break

    quote = text[i_context_start: i_context_end + 1]
    if return_voters:
        return quote, max_voters

    return quote


def reassign_section(d_contexts: list, min_section_votes: int):
    for context in d_contexts:
        if "voters" not in context:
            continue

        context["section_ballot"] = {}

        for voter_name, section in context["voters"].items():
            for i, other_context in enumerate(d_contexts):
                if other_context is context \
                    or "voters" not in other_context \
                    or voter_name not in other_context["voters"] \
                    or section != other_context["voters"][voter_name]:
                    continue

                if i not in context["section_ballot"]:
                    context["section_ballot"][i] = 1
                else:
                    context["section_ballot"][i] += 1

    minor_section, major_section = 1, 1
    set_visited = set()
    for i, context in enumerate(d_contexts):
        if "voters" not in context or i in set_visited:
            continue

        set_visited.add(i)
        context["section"] = f"{major_section}.{minor_section}."

        for i_quote_in_same_section, n_occurrence in context["section_ballot"].items():
            if n_occurrence < min_section_votes:
                continue

            minor_section += 1
            d_contexts[i_quote_in_same_section]["section"] = f"{major_section}.{minor_section}."
            set_visited.add(i_quote_in_same_section)

        major_section += 1
        minor_section = 1


def generate_exemplar(df, teacher_names):
    n_teachers = len(teacher_names)
    n_majority = (n_teachers + 1) // 2
    n_empties = 0

    d_contexts = df["ctxs"]
    for teacher_name in teacher_names:
        lst_quotes = parse_quotes(df[teacher_name])
        if lst_quotes == '':
            n_empties += 1
            if n_empties >= n_majority:
                return ''

        elif isinstance(lst_quotes, str):
            n_teachers -= 1
            n_majority = (n_teachers + 1) // 2
            if n_teachers <= 2:
                # skip data as number of valid outputs are insufficient
                return None
        else:
            vote_quotes(d_contexts, lst_quotes, teacher_name)

    for context in d_contexts:
        quote, voters = extract_major_quote(
            context["text"], context["ballot"], n_majority, return_voters=True)

        quote = quote.strip()
        if len(quote) < 3 or voters is None:
            continue

        context["quote"] = quote
        context["voters"] = voters

    reassign_section(d_contexts, n_majority)

    lst_exemplar = []
    for context in d_contexts:
        if "quote" not in context:
            continue

        lst_exemplar.append(f"{context['section']} {context['title']}\n{context['quote']}")
    
    return '\n'.join(lst_exemplar)


df_arc_c: pd.DataFrame = pd.read_json("train_data/arc_c_train_teacher_models.jsonl", lines=True)
df_hotpotqa = pd.read_json("train_data/hotpotqa_train_teacher_models.jsonl", lines=True)


downstream_inference_name = ("Llama_3_8b", "Mixtral_8x7B", "Llama_2_70b", "Llama_3_70b", "Qwen2_72B")

df_arc_c["exemplar"] = df_arc_c.progress_apply(generate_exemplar, args=(downstream_inference_name,), axis=1)
df_arc_c["exemplar"][df_arc_c["exemplar"].isna()]
df_arc_c["exemplar"][~df_arc_c["exemplar"].isna() & (df_arc_c["exemplar"].str.len() == 0)]