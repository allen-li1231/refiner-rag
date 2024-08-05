import jsonlines
import json
import copy
import re
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft.peft_model import PeftModel


regex_section = re.compile(r"((\d+\.)+|\*)\s+(##\s)?([^\n]+?)(:\s|\n+)(.*)")
regex_quote = re.compile(r'^["\'].*["\']$')
regex_dummy = re.compile(r"([A-Z])\1{2}")

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_recomp_abstractive": 'Question: {question}\n Document: {context}\n Summary: ',
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
    "prompt_refiner_lora": "[INST]<<SYS>>[MONITOR]{context}<</SYS>>{question}[/INST] ",
    "prompt_refiner_prefix": "[INST]<<SYS>>{context}<</SYS>>{question}[/INST] ",
    "prompt_downstream_lora": "[INST]<<SYS>>[EXECUTOR]{refiner}<</SYS>>{question}[/INST] ",
    "prompt_downstream_prefix": "[INST]<<SYS>>{refiner}<</SYS>>{question}[/INST] ",
}

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}

rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]",
                          "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                        "[Utility:3]", "[Utility:4]", "[Utility:5]"]
ground_tokens_names = ["[Fully supported]",
                       "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

TRAIN_DATASET = {
    # self-rag
    "train": "wiki_retrieve.jsonl",
    "triviaqa_train": "triviaqa_train_retrieve.jsonl",
    "hotpotqa_train": "hotpotqa_train_processed.json",
    "2wiki_train": "2wiki_train_processed.json",
    "arc_c_train": "arc_c_train_retrieve.jsonl",
    "pubhealth_train": "pubhealth_train_processed.jsonl",
    "train_truncated": "arc_c_hotpotqa_triviaqa_truncated.jsonl",
}

TRAIN_DATASET_REFINER_OUTPUT = {
    # self-rag
    "train": "wiki_refiner_teacher_expunge.jsonl",
    "triviaqa_train": "triviaqa_refiner_teacher_expunge.jsonl",
    "hotpotqa_train": "hotpotqa_refiner_teacher_expunge.jsonl",
    "2wiki_train": "2wiki_refiner_teacher_expunge.jsonl",
    "arc_c_train": "arc_c_refiner_teacher_expunge.jsonl",
    "pubhealth_train": "pubhealth_refiner_teacher_expunge.jsonl",
    "train_truncated": "llama3_truncated/arc_c_hotpotqa_triviaqa_truncated.jsonl",
}

EVAL_DATASET = {
    # short-form
    "arc_c": "arc_challenge_processed.jsonl",
    "triviaqa": "triviaqa_test.jsonl",
    "popqa": "popqa_longtail.jsonl",
    # multi-hop
    "hotpotqa_dev_fullwiki": "hotpotqa_dev_fullwiki_processed.json",
    "hotpotqa_dev_distractor": "hotpotqa_dev_distractor_processed.json",
    "hotpotqa_test": "hotpotqa_test_processed.json",
    "2wiki_dev": "2wiki_dev_processed.jsonl",
    "2wiki_test": "2wiki_test_processed.jsonl",
    "musique_dev": "musique_dev_processed.jsonl"
}

EVAL_DATASET_REFINER_OUTPUT = {
    # short-form
    "arc_c": "arc_c_refiner_teacher_expunge.jsonl",
    "triviaqa": "triviaqa_refiner_teacher_expunge.jsonl",
    "popqa": "popqa_refiner_teacher_expunge.jsonl",
    # multi-hop
    "hotpotqa_dev_fullwiki": "hotpotqa_dev_fullwiki_refiner_teacher_expunge.jsonl",
    "hotpotqa_dev_distractor": "hotpotqa_dev_distractor_refiner_teacher_expunge.jsonl",
    "hotpotqa_test": "hotpotqa_test_refiner_teacher_expunge.jsonl",
    "2wiki_dev": "2wiki_dev_refiner_teacher_expunge.jsonl",
    "2wiki_test": "2wiki_test_refiner_teacher_expunge.jsonl",
    "musique_dev": "musique_dev_refiner_teacher_expunge.jsonl"
}

DATASET_TYPE = {
    "arc_c": "question", 
    "fever": "statement",
    "triviaqa": "question",
    "popqa": "question",
    "popqa_w_gs": "question",
    "asqa": "question",
    "factscore": "question",
    "hotpotqa_dev_fullwiki": "question",
    "hotpotqa_dev_distractor": "question",
    "hotpotqa_test": "question",
    "2wiki_dev": "question",
    "2wiki_test": "question"
}


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text


def postprocess(pred):
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                      "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")

    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        with open(input_fp) as f:
            input_data = json.load(f)
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp, mode='w'):
    with jsonlines.open(fp, mode=mode) as writer:
        writer.write_all(data)


def preprocess_input(input_data, task):
    if task == "factscore":
        for item in input_data:
            item["instruction"] = item["input"]
            item["output"] = [item["output"]
                              ] if "output" in item else [item["topic"]]
        return input_data

    elif task == "qa":
        for item in input_data:
            if "instruction" not in item:
                item["instruction"] = item["question"]
            if "answers" not in item and "output" in item:
                item["answers"] = "output"
        return input_data

    elif task in ["asqa", "eli5"]:
        processed_input_data = []
        for instance_idx, item in enumerate(input_data["data"]):
            prompt = item["question"]
            instructions = TASK_INST[task]
            prompt = instructions + "## Input:\n\n" + prompt
            entry = copy.deepcopy(item)
            entry["instruction"] = prompt
            processed_input_data.append(entry)
        return processed_input_data


def postprocess_output(input_instance, prediction, task, intermediate_results=None):
    if task == "factscore":
        return {"input": input_instance["input"], "output": prediction, "topic": input_instance["topic"], "cat": input_instance["cat"]}

    elif task == "qa":
        input_instance["pred"] = prediction
        return input_instance

    elif task in ["asqa", "eli5"]:
        # ALCE datasets require additional postprocessing to compute citation accuracy.
        final_output = ""
        docs = []
        if "splitted_sentences" not in intermediate_results:
            input_instance["output"] = postprocess(prediction)

        else:
            for idx, (sent, doc) in enumerate(zip(intermediate_results["splitted_sentences"][0], intermediate_results["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = postprocess(sent)
                final_output += postprocessed_result[:-
                                                     1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if final_output[-1] == " ":
                final_output = final_output[:-1]
            input_instance["output"] = final_output
        input_instance["docs"] = docs
        return input_instance


def postprocess_summarization(
        context: str,
        sep='\n',
        section_type="origin",
        title_type="origin",
        content_type="origin"
):
    lst_contexts = context.split('\n\n')
    if re.match(regex_section, lst_contexts[-1]) is None:
        context = '\n\n'.join(lst_contexts[:-1])

    lst_quotes = re.findall(regex_section, context.rstrip("</s>").strip())
    if len(lst_quotes) == 0 or isinstance(lst_quotes[0], str) and len(lst_quotes[0]) == 0:
        # raise LookupError("Cannot extract quotes from:", context)
        print("---\nCannot extract quotes from:", context)
        return context

    # try removing duplicated quotes
    lst_quotes.reverse()
    # remove duplicated quotes with larger section number using hash table
    dict_quotes = {quote.strip(" \"'\n") if re.match(regex_quote, quote.strip()) else quote: 
                   ['.'.join(section.split('.')[:2]) , title]
                   for section, _, _, title, _, quote in lst_quotes}

    lst_quotes = list(dict_quotes.keys())
    for i, quote in enumerate(lst_quotes):
        if quote == -1 or len(quote) < 3 or (quote == quote[0] * 3):
            lst_quotes[i] = -1
            if quote in dict_quotes:
                del dict_quotes[quote]
            continue

        for j, q in enumerate(lst_quotes[i + 1:]):
            if q != -1 and q in quote:
                if q in dict_quotes:
                    del dict_quotes[q]
                lst_quotes[i + j + 1] = -1

    # reverse quotes on sections
    lst_quotes.sort(key=lambda x: float(dict_quotes[x][0]
                        if isinstance(x, str) and dict_quotes[x][0] != '*'
                        else "inf"))
    # correct numeric sections
    minor_section = 1
    major_section = 1
    last_major_section = 1
    for quote in lst_quotes:
        if quote == -1:
            continue

        sections = dict_quotes[quote][0].split('.')
        cur_major_section = int(sections[0]) if sections[0].isnumeric() else major_section

        if (last_major_section == cur_major_section) or (major_section == 1) and (minor_section == 1):
            dict_quotes[quote][0] = f"{major_section}.{minor_section}."
        else:
            major_section += 1
            minor_section = 1
            dict_quotes[quote][0] = f"{major_section}.{minor_section}."

        minor_section += 1
        last_major_section = cur_major_section

    # concat back section and quote
    lst_cleaned_quotes = []
    for quote in lst_quotes:
        if not isinstance(quote, str):
            continue
        
        lst_structure = []
        if section_type is not None:
            section = dict_quotes[quote][0]
            if section_type == "star":
                section = '*'
            elif section_type == "number":
                section = f'{len(lst_cleaned_quotes) + 1}.'
            lst_structure.append(section)
        if title_type is not None:
            title = dict_quotes[quote][1]
            if title_type == "quote":
                title = f'"{title}"'
            if title_type == "md":
                title = f'## {title}'
            lst_structure.append(title)

        if content_type == "quote":
            quote = f'''"{quote.replace('"', "'")}"'''

        if len(lst_structure) == 0:
            lst_cleaned_quotes.append(quote)
        elif title_type is not None:
            lst_cleaned_quotes.append(
                f"{' '.join(lst_structure)}\n{quote}"
            )
        else:
            lst_cleaned_quotes.append(
                f"{' '.join(lst_structure)} {quote}"
            )

    all_quotes = re.sub(regex_dummy, '(No Title)', sep.join(lst_cleaned_quotes))
    return all_quotes


def process_arc_instruction(item, instruction):
    choices = item["choices"]
    answer_labels = {}
    for i in range(len(choices["label"])):
        answer_key = choices["label"][i]
        text = choices["text"][i]
        if answer_key == "1":
            answer_labels["A"] = text
        if answer_key == "2":
            answer_labels["B"] = text
        if answer_key == "3":
            answer_labels["C"] = text
        if answer_key == "4":
            answer_labels["D"] = text
        if answer_key == "5":
            answer_labels["E"] = text
        if answer_key in ["A", "B", "C", "D", "E"]:
            answer_labels[answer_key] = text

    choices = '\n'.join([f"{k}: {v}" for k, v in answer_labels.items()])
    processed_instruction = ('' if instruction is None else f"{instruction}\n") \
                            + f'{item.get("instruction", item["question"])}\n{choices}'
    return processed_instruction


def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output


def _concat_passages_by_id(df_passages: pd.DataFrame, n_contexts=2):
    last_i = None
    s_res = ''
    df_passages.sort_values("score", ascending=False, inplace=True)
    df_passages = df_passages.iloc[:n_contexts]
    df_passages.sort_values("id", inplace=True)
    for t, psg, i in zip(df_passages.title, df_passages.passage, df_passages.id):
        if last_i is None:
            s_res = f"## {t}\n{psg}"
        elif last_i + 1 == i:
            s_res += f" {psg}"
        else:
            s_res += f"...{psg}"

        last_i = i

    return s_res


def process_retriever_passage(data,
                              instruction: str = None,
                              max_len: int = None,
                              n_docs: int = 10,
                              n_contexts: int = 2,
                              highlight_keyword: bool = False,
                              sort: bool = True):
    if "choices" in data:
        prompt = process_arc_instruction(data, instruction)
    else:
        prompt = ('' 
                  if instruction is None or data["question"].startswith(instruction)
                  else f'{instruction}\n') \
            + data.get("input", data["question"])

    relevant_passages = {}
    ctxs = data["ctxs"]

    cur_len = 0
    # deduplicate context using dict
    for i, ctx in enumerate(ctxs["ctxs"]) if "ctxs" in ctxs else enumerate(ctxs):
        context: str = ctx["text"].strip()
        if len(context) < 5 \
            or ctx["title"].startswith("Category:") \
            or context.find(" ; ") > 0:
            continue

        if highlight_keyword:
            idx_keyword = ctx["text"].lower().find(k.lower())
            if idx_keyword >= 0:
                # markdown bold keyword
                context = f"{context[:idx_keyword]}**{k}**{context[:idx_keyword + len(k)]}"

        # assume ctxs is sorted by score in descending order
        if isinstance(max_len, (int, float)) and cur_len + len(context) + len(ctx["title"]) > max_len:
            break

        cur_len += len(context) + len(ctx["title"])
        relevant_passages[context] = float(ctx.get('score', 0.)), ctx["title"], int(ctx.get('id', i))

        # relevant_passages.insert(0, f"## Hint: {k.capitalize()}")

    df = pd.DataFrame({
        "passage": relevant_passages.keys(),
        "score": [relevant_passages[x][0] for x in relevant_passages.keys()],
        "title": [relevant_passages[x][1] for x in relevant_passages.keys()],
        "id": [relevant_passages[x][2] for x in relevant_passages.keys()]
    })
    df["max_score"] = df.groupby("title")["score"].transform("max")
    srs_passages = (df.groupby(["title", "max_score"], sort=False)[["title", "passage", "id", "score"]]
                    .apply(lambda x: _concat_passages_by_id(x, n_contexts=n_contexts))
                   ).sort_index(level=1, ascending=False)
    
    srs_passages = srs_passages.iloc[:n_docs]
    if not sort:
        srs_passages = srs_passages.sample(frac=1.)
    return prompt, "\n---\n".join(srs_passages)


def model_generate(prompt,
                   model,
                   tokenizer=None,
                   temperature=0.8,
                   max_new_tokens=2048,
                   top_k=1,
                   top_p=1.,
                   beam_width=1,
                   do_sample=False,
                   num_return_sequences=1,
                   **kwargs
                   ):

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if hasattr(model, "device") and model.device.type != "cuda" or isinstance(model, (AutoModel, PeftModel)):
        tokenizer = kwargs.get("tokenizer", tokenizer)
        if tokenizer is None:
            raise ValueError("tokenizer must present if not using cuda")

        inputs = tokenizer(prompt, padding="longest", return_tensors="pt")

        with torch.no_grad():
            if temperature is None or temperature <= 0.:
                preds = model.generate(
                    **inputs.to(model.device),
                    top_p=top_p,
                    num_beams=beam_width,
                    temperature=None,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    **kwargs)
            else:
                preds = model.generate(
                    **inputs.to(model.device),
                    temperature=0. if temperature is None else temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=beam_width,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    **kwargs)

        pred_token_ids = preds.sequences[:, inputs.input_ids.shape[1]:]
        pred_text = tokenizer.batch_decode(pred_token_ids)
        pred_log_probs = F.log_softmax(torch.stack(preds.scores), dim=2)
        pred_log_probs = torch.swapaxes(pred_log_probs, 0, 1).to("cpu").numpy()

    else:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
            use_beam_search=do_sample, n=num_return_sequences, logprobs=5)
        preds = model.generate(prompt, sampling_params, **kwargs)
        pred_token_ids = [[output.token_ids for output in p.outputs[: num_return_sequences]] for p in preds]
        pred_text = [[output.text for output in p.outputs[: num_return_sequences]] for p in preds]
        pred_log_probs = [[output.logprobs for output in p.outputs[: num_return_sequences]] for p in preds]

    return pred_text, pred_token_ids, pred_log_probs