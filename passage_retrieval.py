# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import json
import jsonlines
import pickle
import time
import glob

import numpy as np
import pandas as pd
import torch
import faiss

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
import src.normalize_text
from src.utils import DEVICE
from utils import save_file_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Retriever:
    def __init__(self,
        model_name_or_path: str,
        passages: str,
        passage_embeddings: str,
        no_fp16=False,
        save_or_load_index=False,
        indexing_batch_size=1000000,
        lowercase=False,
        normalize_text=True,
        per_gpu_batch_size=64,
        question_maxlength=512,
        projection_size=768,
        n_subquantizers=0,
        n_bits=8,
        index_device="cpu"
    ):
        self.model_name_or_path = model_name_or_path
        self.passages = passages
        self.passage_embeddings = passage_embeddings
        self.no_fp16 = no_fp16
        self.save_or_load_index = save_or_load_index
        self.indexing_batch_size = indexing_batch_size
        self.lowercase = lowercase
        self.normalize_text = normalize_text
        self.per_gpu_batch_size = per_gpu_batch_size
        self.question_maxlength = question_maxlength
        self.projection_size = projection_size
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.index_device = index_device

        self.setup_retriever()

    def embed_queries(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if self.lowercase:
                    q = q.lower()
                if self.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == self.per_gpu_batch_size or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=self.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.to(DEVICE) for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.to(self.index_device))

                    batch_question.clear()
                    getattr(torch, DEVICE).empty_cache()

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings

    def embed_queries_demo(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                batch_question.append(q)

                if len(batch_question) == 16 or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=200,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.to(DEVICE) for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()

    def index_encoded_data(self, embedding_files, indexing_batch_size):
        all_ids = []
        all_embeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            all_embeddings = np.vstack((all_embeddings, embeddings)) if all_embeddings.size else embeddings
            all_ids.extend(ids)
            while all_embeddings.shape[0] > indexing_batch_size:
                all_embeddings, all_ids = self.add_embeddings(all_embeddings, all_ids, indexing_batch_size)

        while all_embeddings.shape[0] > 0:
            all_embeddings, all_ids = self.add_embeddings(all_embeddings, all_ids, indexing_batch_size)

        print("Data indexing completed.")

    def add_embeddings(self, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        self.index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        lst_docs = []
        for passage_ids, scores in top_passages_and_scores:
            lst_doc = []
            for p_id, score in zip(passage_ids, scores):
                doc = passages[p_id].copy()
                doc["score"] = float(score)
                lst_doc.append(doc)
            
            lst_docs.append(lst_doc)

        return lst_docs

    def setup_retriever(self):
        print(f"Loading model from: {self.model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(self.model_name_or_path)
        self.model.eval()
        self.model = self.model.to(DEVICE)
        if not self.no_fp16:
            self.model = self.model.half()

        self.index = src.index.Indexer(self.projection_size, self.n_subquantizers, self.n_bits)
        if self.index_device.startswith("cuda"):
            if src.slurm.is_distributed:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), src.slurm.local_rank, self.index)
            else:
                n_gpus = faiss.get_num_gpus()
                if n_gpus <= 0:
                    raise LookupError("Fiass cannot detect a gpu")

                if n_gpus == 1:
                    self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                else:
                    self.index = faiss.index_cpu_to_all_gpus(self.index)

        # index all passages
        input_paths = glob.glob(self.passage_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(input_paths, self.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.save_or_load_index:
                self.index.serialize(embeddings_dir)

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(self.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    def search_document(self, query, top_n=10, index_batch_size=2048):
        queries = [query] if isinstance(query, str) else query

        questions_embedding = self.embed_queries(queries)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, top_n, index_batch_size)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        lst_passages = self.add_passages(self.passage_id_map, top_ids_and_scores)
        return lst_passages[0] if isinstance(query, str) else lst_passages

    def search_document_demo(self, query, n_docs=10):
        questions_embedding = self.embed_queries_demo([query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:n_docs]

    def setup_retriever_demo(self, model_name_or_path, passages, passages_embeddings, n_docs=5, save_or_load_index=False):
        print(f"Loading model from: {model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(model_name_or_path)
        self.model.eval()
        self.model = self.model.to(DEVICE)

        self.index = src.index.Indexer(768, 0, 8)

        # index all passages
        input_paths = glob.glob(passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(input_paths, 1000000)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        with jsonlines.open(data_path, 'r') as jsonl_f:
            data = [obj for obj in jsonl_f]
    elif data_path.endswith(".parquet"):
        data = pd.read_parquet(data_path).to_dict("records")
    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passage_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--output", type=str, help="dir path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--n_docs", type=int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    args = parser.parse_args()

    if args.num_shards > 1:
        src.slurm.init_distributed_mode(args)

    # for debugging
    # data_paths = glob.glob(args.data)
    retriever = Retriever(
        model_name_or_path=args.model_name_or_path,
        passages=args.passages,
        passage_embeddings=args.passage_embeddings,
        no_fp16=args.no_fp16,
        save_or_load_index=args.save_or_load_index,
        indexing_batch_size=args.indexing_batch_size,
        lowercase=args.lowercase,
        normalize_text=args.normalize_text,
        per_gpu_batch_size=args.per_gpu_batch_size,
        question_maxlength=args.question_maxlength,
        projection_size=args.projection_size,
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits
    )

    query = args.query
    if os.path.exists(query):
        query = load_data(query)

        shard_size = len(query) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size
        if args.shard_id == args.num_shards - 1:
            end_idx = len(query)

        query = query[start_idx: end_idx]
        print("query length:", len(query))

    retrieved_documents = retriever.search_document(query, args.n_docs)
    if isinstance(args.output, str):
        if isinstance(query, str):
            data = [{"question": query, "ctxs": retrieved_documents}]
        else:
            data = [{"question": question, "ctxs": ctx}
                    for question, ctx in zip(query, retrieved_documents)]
        save_file_jsonl(data, args.output)
    else:
        print(retrieved_documents)


if __name__ == "__main__":
    # --query "What is the occupation of Obama?" --passages ./wikipedia_data/psgs_w100.tsv --passage_embeddings "./wikipedia_data/embedding_contriever-msmarco/*" --model_name_or_path "facebook/contriever-msmarco" --output ./train_data/extractor_retrieve_wiki.jsonl
    main()
    # retriever = Retriever(
    #     "facebook/contriever-msmarco",
    #     "../wikipedia_data/psgs_w100.tsv",
    #     "../wikipedia_data/embedding_contriever-msmarco/*"
    # )
    # lst_questions = load_data("../train_data/retrieval_question.jsonl")

    # n_docs = 10
    # retrieved_documents = retriever.search_document(lst_questions, n_docs)
    # data = [{"question": question, "ctxs": ctx} for question, ctx in zip(lst_questions, retrieved_documents)]
    # save_file_jsonl(data, f"../train_data/extractor_retrieve_{n_docs}docs.jsonl")