from utils import load_file, save_file_jsonl


hotpot_expunge = load_file("./train_data/hotpotqa_extract_expunge.jsonl")
arc_c_expunge = load_file("./train_data/arc_c_extract_expunge.json")
trivia_expunge = load_file("./train_data/triviaqa_extract_expunge.jsonl")

for data in hotpot_expunge:
    del data["answer"]
    
for data in arc_c_expunge:
    data["question"] = data["instruction"]
    del data["choices"], data["answerKey"], data["instruction"]

train_data = hotpot_expunge + arc_c_expunge + trivia_expunge
save_file_jsonl(train_data, "./train_data/llama3_extract_expunge.jsonl")