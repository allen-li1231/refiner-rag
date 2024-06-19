## Refiner: Restructure Retrieved Content Efficiently to Advance Question-Answering Capabilities

## TL;DR
_Refiner_ is an end-to-end extract-and-restructure paradigm that incorporates query-relevant contents, contexts, and sectionalizes interconnected information, ensuring information distinction and alignment with the original context. Refiner achieves a 80.5% tokens reduction and a 1.6-7.0% improvement margin in
multi-hop tasks, rivaling with next best solution, [LongLLMLingua](https://arxiv.org/abs/2310.06839).

**How Refiner Works**
Refiner integrates seamlessly with RAG systems, leveraging a single decoder-only LLM to:

* **Adaptively extract query-relevant contents**: Verbatim extraction of necessary context and sectioning of interconnected contents.
* **Preserve information distinction**: Highlights contextual relationships, ensuring effective representation of original context.

**Benefits**
Refiner offers:
* **Improved answer accuracy**: Significant gain in downstream LLM performance.
* **Efficient compression**: Up to 80%+ token reduction.


**Get Started**
To start, first revise the GPU number in your environment in ```submit_evaluation_accelerate.py```
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