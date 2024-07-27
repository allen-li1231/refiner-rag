import re
import pylcs
import pandas as pd
from tqdm.auto import tqdm
from utils import regex_section
from submit_get_refiner_teacher_data import TASKS, downstream_inference_name


def common_str_idx(s1: str, s2: str):
    # find longest common substring and return corresponding index
    return pylcs.lcs_string_idx(s1, s2)


def parse_quotes(teacher_answer: str):
    # parse quotes from a single teacher's answer
    lst_teacher_quotes = teacher_answer.split('\n\n')
    if re.match(regex_section, lst_teacher_quotes[-1]) is None:
        teacher_answer = '\n\n'.join(lst_teacher_quotes[:-1])

    lst_quotes = re.findall(regex_section, teacher_answer.rstrip("</s>").strip())
    # return original teacher_answer if parsing fails
    if len(lst_quotes) == 0 or isinstance(lst_quotes[0], str) and len(lst_quotes[0]) == 0:
        # raise LookupError("Cannot extract quotes from:", context)
        # print("---\nCannot extract quotes from:", teacher_answer)
        return teacher_answer

    return lst_quotes


def vote_quotes(lst_context: list, lst_quotes: list, voter_name: str, min_valid_str_len=3):
    # record quote poll with the following steps:
    # 1. retrieve verbatim quotes and sections parsed from a teacher answer
    # 2. calculate poll for each quote by accumulating the occurrence of quotes under word level
    # 3. the name of voter (a.k.a, name of teacher model) will be also recorded along with
    #    the section with respective to the quote, this is for serving reassigning section afterward
    
    # quick exit on empty quotes
    if len(lst_quotes) == 0 or isinstance(lst_quotes[0], str) and len(lst_quotes[0]) == 0:
        return

    for d_context in lst_context:
        if "ballot" not in d_context:
            # initialize quote ballot for multi-teachers
            d_context["ballot"] = {}

        for section, _, _, title, _, quote in lst_quotes:
            if len(title) < min_valid_str_len \
                or len(quote) < min_valid_str_len \
                or title not in d_context["title"]:
                # skip quotes until it matches with the title of a context
                continue

            # word-level context matching
            lst_common_idx = common_str_idx(d_context["text"], quote)
            # parse number of main section
            current_section = int(section.split('.')[0])

            for i, idx in enumerate(lst_common_idx):
                if idx == -1:
                    # idx=1 means the word in the original context respective to the index
                    # does not occur in the parsed quote
                    continue

                # record name of voter along with the designated section number
                if i not in d_context["ballot"]:
                    d_context["ballot"][i] = {voter_name: current_section}
                else:
                    d_context["ballot"][i][voter_name] = current_section


def extract_major_quote(text, ballots, min_context_votes, return_voters=False):
    # sort ballot on index,
    # so we can iterate through it and find the majority voted text from context
    ballots = dict(sorted(ballots.items()))

    i_context_start, i_context_end, max_voters = -1, -1, None
    for i, ballot in ballots.items():
        if max_voters is None and len(ballot) >= min_context_votes:
            i_context_start = i
            max_voters = ballot
        elif max_voters is not None and len(max_voters) < len(ballot):
            max_voters = ballot

    # exit if number of actual votes does not meet the min_context_votes
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
        # skip unselected contexts
        if "voters" not in context:
            continue

        context["section_ballot"] = {}

        # we want to find combinations of quotes voted by a majority of teachers
        # and put them into the same section subsequently
        for voter_name, section in context["voters"].items():
            for i, other_context in enumerate(d_contexts):
                if other_context is context \
                    or "voters" not in other_context \
                    or voter_name not in other_context["voters"] \
                    or section != other_context["voters"][voter_name]:
                    # only accumulate concurrence when number of section matches
                    continue

                # record number of concurrence on context level
                if i not in context["section_ballot"]:
                    context["section_ballot"][i] = 1
                else:
                    context["section_ballot"][i] += 1

    # it's time to reassign the sections
    minor_section, major_section = 1, 1
    set_visited = set()
    # iterate through all context and assign sections based on calculated section_ballot
    for i, context in enumerate(d_contexts):
        # once all quotes in a previous visited section are assigned with section
        # skip them in the following iterations to avoid duplication
        if "voters" not in context or i in set_visited:
            continue

        set_visited.add(i)
        context["section"] = f"{major_section}.{minor_section}."

        # assignment under one section
        for i_quote_in_same_section, n_occurrence in context["section_ballot"].items():
            if n_occurrence < min_section_votes:
                continue

            minor_section += 1
            d_contexts[i_quote_in_same_section]["section"] = f"{major_section}.{minor_section}."
            set_visited.add(i_quote_in_same_section)

        # assignments across sections
        major_section += 1
        minor_section = 1


def generate_exemplar(df, teacher_names):
    # generate Refiner's exemplar output

    n_teachers = len(teacher_names)
    # the number of majority votes
    n_majority = (n_teachers + 1) // 2
    n_empties = 0

    d_contexts = df["ctxs"]
    for teacher_name in teacher_names:
        lst_quotes = parse_quotes(df[teacher_name])
        if lst_quotes == '':
            n_empties += 1
            # handle majority of empty answers
            if n_empties >= n_majority:
                return ''

        elif isinstance(lst_quotes, str):
            # the lst_quotes is malformed, thus not successfully parsed
            # the corresponding teacher's answer should be disregarded
            n_teachers -= 1
            n_majority = (n_teachers + 1) // 2
            if n_teachers <= 2:
                # skip data as number of valid outputs are insufficient
                return None
        else:
            vote_quotes(d_contexts, lst_quotes, teacher_name)

    # get most voted quote text and the corresponding voters
    for context in d_contexts:
        quote, voters = extract_major_quote(
            context["text"], context["ballot"], n_majority, return_voters=True)

        quote = quote.strip()
        # skip empty quotes
        if len(quote) < 3 or voters is None:
            continue

        context["quote"] = quote
        context["voters"] = voters

    reassign_section(d_contexts, n_majority)

    # concatenate quotes, sections along with the titles
    lst_exemplar = []
    for context in d_contexts:
        if "quote" not in context:
            continue

        lst_exemplar.append(f"{context['section']} {context['title']}\n{context['quote']}")

        del context["ballot"]
        if "voters" in context:
            del context["voters"]
            del context["quote"]
        if "section_ballot" in context:
            del context["section_ballot"]

    return '\n'.join(lst_exemplar)


if __name__ == '__main__':
    tqdm.pandas(desc="Applying")

    for task in TASKS:
        df_train_data: pd.DataFrame = pd.read_json(f"train_data/{task}_teacher_models.jsonl", lines=True)

        df_train_data["exemplar"] = df_train_data.progress_apply(generate_exemplar, args=(downstream_inference_name,), axis=1)
        # remove dirty exemplars
        df_train_data = df_train_data[~df_train_data["exemplar"].isna()]

        df_train_data.to_json(f"train_data/{task}_teacher_models.jsonl", lines=True, orient="records")
        # df_train_data["exemplar"][~df_train_data["exemplar"].isna() & (df_train_data["exemplar"].str.len() == 0)]