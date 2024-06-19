## Refiner: Restructure Retrieved Content Efficiently to Advance Question-Answering Capabilities

## TL;DR
_Refiner_ is an end-to-end extract-and-restructure paradigm that incorporates query-relevant contents, contexts, and sectionalizes interconnected information, ensuring information distinction and alignment with the original context. Refiner achieves a 80.5% tokens reduction and a 1.6-7.0% improvement margin in
multi-hop tasks, rivaling with next best solution, [LongLLMLingua](https://arxiv.org/abs/2310.06839).

## How Refiner Works
Refiner integrates seamlessly with RAG systems, leveraging a single decoder-only LLM to:

* **Adaptively extract query-relevant contents**: Verbatim extraction of necessary context and sectioning of interconnected contents.
* **Preserve information distinction**: Highlights contextual relationships, ensuring effective representation of original context.

## Benefits
* **Improved answer accuracy**: Significant gain in downstream LLM performance.
* **Efficient compression**: Up to 80%+ token reduction.


## Get Started

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
```python
!pip install -qU transformers accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel


base_model = "meta-llama/Llama-2-7b-chat-hf"
adapter = "al1231/Refiner-7B"
TEMPLATE = "[INST]<<SYS>>[MONITOR]{context}<</SYS>>{question}[/INST] "

tokenizer = AutoTokenizer.from_pretrained(base_model)
base_model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
model.eval()

question = "What is John Finlay's occupation?"
context = "## John Finlay (footballer)\nJohn Finlay (16 February 1919 – 5 March 1985) was an English professional footballer who played as an inside forward for Sunderland. John Finlay made his debut on the 11th of September 1946 as a substitute for Sunderland AFC on their 4th match of the season in Division 1 against Charlton Athletic. 27,425 attended the match witnessing John's Debut. The Match which started at 6:00pm at Charlton Athletic's stadium “The Valley” was refereed by W.H.E Evans\n---\n## Andrew Finlay\nAndrew Finlay (born 10 February 1901; date of death unknown) was a Scottish footballer who played as a forward for Port Vale, Airdrieonians, Manchester City, Crewe Alexandra, Third Lanark, Dundee United and Hibernian in the 1920s....Source:\n---\n## John Finlay (poet)\nJohn Finlay (1782–1810) was a Scottish poet. Finlay was born in Glasgow in December 1782. He was educated in one of the academies at Glasgow, and at the age of fourteen entered the university, where he had as a classmate John Wilson (alias 'Christopher North'), who states that he was distinguished \"above most of his contemporaries\". The prospect of obtaining a situation in one of the public offices led him to visit London in 1807, and while there he contributed to the magazines some articles on antiquarian subjects. Not finding suitable employment he returned to Glasgow in 1808. He began to collect materials for a continuation of Warton's History of Poetry, but in 1810 he left Glasgow to visit Professor Wilson at Ellerlay, Westmoreland; on the way he fell ill at Moffat, and died there on 8 December.\n---\n## John Finlay (Canadian politician)\nJohn Finlay (April 22, 1837 &ndash; November 13, 1910) was a Canadian politician. Born in Dummer Township, Peterborough County, Upper Canada, Finlay was educated in the Public Schools of Dummer. A manufacturer, Finlay was Councillor and Reeve of the Village of Norwood and County Councillor. He was elected to the House of Commons of Canada for the electoral district of Peterborough East in the general elections of 1904. A Liberal, he did not run in the 1908 elections.\n---\n## John Finlay (fur trader)\nJohn Finlay (1774 – December 19, 1833) was a fur trader and explorer with the North West Company. He is best remembered for establishing the first fur trading post in what is now British Columbia, Canada and for his exploration of the Finlay River, one of the two major rivers forming the Peace River. Finlay was born in Montreal, the son of James Finlay, who himself was a significant player in the western Canadian fur trade. Finlay was apprenticed as a clerk in the North West Company in 1789 at the age of 15. He accompanied Alexander Mackenzie on his historic trip across the Rocky Mountains to the Pacific Ocean in 1792-93 becoming, with him, the first European to traverse North America. He was placed...Finlay in 1824, noting that \"he had studied Finlay’s chart.\" Nonetheless, it would appear from the information Black had that Finlay had only made it as far as the Ingenika River, about 130 km north of the Finlay River's confluence with the Peace. Indeed, Black's journal makes clear that the northern branch, far from being less complicated, was all but impassable in many parts, perhaps explaining Finlay's reluctance to travel more than about one-quarter of the river's actual length. Finlay remained in the North West Company's Athabasca Department, becoming a partner of the company in 1799. He retired from the fur trade in 1804 and returned to Montreal. Little is known of his life there, except that he obtained an appointment as deputy commissary-general."

prompt = TEMPLATE.format(question=question, context=context)

inputs = tokenizer(prompt, return_tensors="pt")

preds = model.generate(
    **inputs.to(model.device),
    top_p=1,
    temperature=None,
    do_sample=False,
    max_new_tokens=2048,
    num_return_sequences=1,
    output_scores=True,
    return_dict_in_generate=True,
    use_cache=True)
pred_token_ids = preds.sequences[:, inputs.input_ids.shape[1]:]
pred_text = tokenizer.batch_decode(pred_token_ids)
print(pred_text)
```

To reproduce paper experiment, first revise the GPU number in your environment in ```submit_evaluation_accelerate.py```
Then in the root directory, run the following code to evaluate _Refiner_:
```sh
python ./submit_evaluation_accelerate.py --adapter_name refiner --top_n 10 --eval_refiner
```

Then you can run the following code to evaluate downstream language models:
```sh
python ./submit_evaluation_accelerate.py --adapter_name refiner --top_n 10 --eval_downstream
```
To evaluate GPT 3.5 Turbo under our default retriever setting, first provide your OpenAI token in ```get_executor_data.py```, line 15, then run:
```sh
python ./submit_evaluation_accelerate.py --adapter_name refiner --use_openai --top_n 10 --eval_baseline
```
For RECOMP Abstractive Compressor:
```sh
python ./submit_evaluation_recomp.py --top_n 10 --eval_executor
```

For LongLLMLingua:
```sh
python ./submit_evaluation_longllmlingua.py --top_n 10 --inference_name longllmlingua --eval_executor --rate 0.5 --dynamic_context_compression_ratio 0.3 --output_dir "../eval_data/longllmlingua/top_10/"
```

## Citation
```cite
@misc{li2024textitrefiner,
      title={$\textit{Refiner}$: Restructure Retrieval Content Efficiently to Advance Question-Answering Capabilities}, 
      author={Zhonghao Li and Xuming Hu and Aiwei Liu and Kening Zheng and Sirui Huang and Hui Xiong},
      year={2024},
      eprint={2406.11357},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}