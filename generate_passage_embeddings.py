# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import argparse
import pickle
from tqdm.auto import tqdm

import torch

import src.slurm
import src.contriever
import src.utils
import src.data
import src.normalize_text

from src.utils import DEVICE


def embed_passages(args, passages, model, tokenizer):
    total = 0
    all_ids, all_embeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in enumerate(tqdm(passages)):
            batch_ids.append(p["id"])
            if args.no_title or not "title" in p:
                text = p["text"]
            else:
                text = p["title"] + "\n" + p["text"]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.to(DEVICE) for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                all_ids.extend(batch_ids)
                all_embeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                getattr(torch, DEVICE).empty_cache()

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_ids, all_embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="prefix path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_maxlength", type=int, default=2048, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")

    args = parser.parse_args()

    if args.num_shards > 1:
        src.slurm.init_signal_handler()
        src.slurm.init_distributed_mode(args)

    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    print(f"Model loaded from {args.model_name_or_path}.", flush=True)
    model.eval()
    model = model.to(DEVICE)
    if not args.no_fp16:
        model = model.half()

    passages = src.data.load_passages(args.passages)

    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    print(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.")

    all_ids, all_embeddings = embed_passages(args, passages, model, tokenizer)

    save_file = os.path.join(args.output_dir, args.prefix + f"_{args.shard_id:02d}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(all_ids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((all_ids, all_embeddings), f)

    print(f"Total passages processed {len(all_ids)}. Written to {save_file}.")


if __name__ == "__main__":
    # ipython generate_passage_embeddings.py --model_name_or_path BAAI/bge-m3 --output_dir ../wikipedia_data/embedding_bge-m3 --passages ../wikipedia_data/psgs_w100.tsv
    main()
