import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Seq2SeqDataset_MetaQA, TestDataset_MetaQA
from model import TransformerModel


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained MetaQA checkpoints.")
    parser.add_argument("--embedding-dim", default=256, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--test-batch-size", default=64, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--save-dir", default="metaqa_model")
    parser.add_argument("--ckpt", default="best_model.pt")
    parser.add_argument("--dataset", default="metaqa")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--label-smooth", default=0.5, type=float)
    parser.add_argument("--encoder", default=False, action="store_true")
    parser.add_argument("--smart-filter", default=False, action="store_true")
    parser.add_argument(
        "--question-file",
        default="qa.csv",
        type=str,
        help="fallback question CSV used for both train dictionary loading and evaluation",
    )
    parser.add_argument("--train-question-file", default=None, type=str, help="MetaQA CSV used to load the training dictionary/entities")
    parser.add_argument("--eval-question-file", default=None, type=str, help="MetaQA CSV used for test evaluation")
    parser.add_argument("--max-q-len", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--test-paraphrased", default=False, action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def normalize_gold_answer_ids(gold_answers, fallback_target_id):
    values = []
    if gold_answers is not None:
        if torch.is_tensor(gold_answers):
            values = gold_answers.detach().cpu().view(-1).tolist()
        else:
            values = list(gold_answers)

    normalized = []
    seen = set()
    for value in values:
        if value is None:
            continue
        value = int(value)
        if value < 0 or value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    fallback_target_id = int(fallback_target_id)
    if not normalized:
        normalized.append(fallback_target_id)
    return normalized


def first_correct_candidate_rank(candidate_ids, gold_answer_ids):
    gold_answer_set = {int(answer_id) for answer_id in gold_answer_ids}
    for idx, candidate_id in enumerate(candidate_ids):
        if int(candidate_id) in gold_answer_set:
            return idx
    return None


def row_to_hop_count(row):
    if row is None:
        return None
    for column_name in ("Hops", "Num-Hops", "N-Hop", "hop", "num_hops"):
        if column_name not in row.index:
            continue
        hop_value = row.get(column_name)
        if hop_value is None:
            continue
        if isinstance(hop_value, (int, np.integer)):
            return int(hop_value)
        if isinstance(hop_value, float):
            if np.isnan(hop_value):
                continue
            return int(hop_value)
        hop_digits = "".join(ch for ch in str(hop_value) if ch.isdigit())
        if hop_digits:
            return int(hop_digits)
    return None


def load_dataset(dataset_cls, data_path, vocab_file, device, args, split_candidates):
    last_dataset = None
    for split in split_candidates:
        dataset = dataset_cls(
            data_path=data_path,
            vocab_file=vocab_file,
            device=device,
            split=split,
            args=args,
        )
        last_dataset = dataset
        if split is None or len(dataset) > 0:
            return dataset
    return last_dataset


def build_entity_candidate_ids(*datasets):
    candidate_ids = set()
    for dataset in datasets:
        if dataset is None:
            continue
        entity_candidate_ids = getattr(dataset, "entity_candidate_ids", None)
        if entity_candidate_ids is not None:
            values = entity_candidate_ids.view(-1).tolist()
            candidate_ids.update(int(candidate_id) for candidate_id in values if int(candidate_id) >= 0)
            continue

        entity2id = getattr(dataset, "entity2id", {})
        dictionary = getattr(dataset, "dictionary", None)
        if dictionary is None:
            continue
        for entity_token, mapped_token in entity2id.items():
            dict_id = dictionary.indices.get(str(mapped_token))
            if dict_id is None:
                dict_id = dictionary.indices.get(str(entity_token))
            if dict_id is not None:
                candidate_ids.add(int(dict_id))

    if not candidate_ids:
        raise ValueError("Unable to build MetaQA entity candidate ids from the provided datasets.")

    return torch.tensor(sorted(candidate_ids), dtype=torch.long)


def build_entity_candidate_mask(vocab_size, entity_candidate_ids, device):
    entity_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    entity_mask[entity_candidate_ids.to(device)] = True
    return entity_mask


def mask_entity_logits(logits, entity_mask):
    masked_logits = logits.masked_fill(~entity_mask.unsqueeze(0), float("-inf"))
    if torch.isinf(masked_logits).all(dim=-1).any():
        raise ValueError("Entity candidate masking removed every candidate. Check entity2id.txt and MetaQA answer mappings.")
    return masked_logits


class MetaQAEndpointModel(nn.Module):
    def __init__(self, args, dictionary):
        super().__init__()
        self.base_model = TransformerModel(args, dictionary)
        self.head_norm = nn.LayerNorm(args.embedding_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.dictionary = dictionary

    def encode_question(self, input_ids, attention_mask):
        encoded_question = self.base_model.encode_question(input_ids, attention_mask).transpose(0, 1)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded_question * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def answer_logits(self, input_ids, attention_mask, head_id=None):
        question_repr = self.encode_question(input_ids, attention_mask)

        if head_id is not None:
            valid_head = head_id.ge(0)
            if valid_head.any():
                head_repr = torch.zeros_like(question_repr)
                head_repr[valid_head] = self.base_model.encoder(head_id[valid_head])
                question_repr = self.head_norm(question_repr + self.dropout(head_repr))

        hidden = self.base_model.glue(self.base_model.fc(self.dropout(question_repr)))
        return torch.matmul(hidden, self.base_model.encoder.weight.transpose(0, 1))

    def forward(self, input_ids, attention_mask, head_id=None):
        return self.answer_logits(input_ids, attention_mask, head_id=head_id)


def evaluate(model, dataloader, device, args, entity_candidate_ids, entity_mask, split_name="eval"):
    model.eval()
    candidate_ids_device = entity_candidate_ids.to(device)
    metric_totals = {
        "mrr": 0.0,
        "hit1": 0.0,
        "count": 0,
    }
    split_label = split_name.title()
    dataset = getattr(dataloader, "dataset", None)
    hop_metrics = {}

    with tqdm(dataloader, desc=f"{split_label} Eval") as pbar:
        for samples in pbar:
            samples = move_batch_to_device(samples, device)
            logits = model(
                input_ids=samples["input_ids"],
                attention_mask=samples["attention_mask"],
                head_id=samples.get("head_id"),
            )
            masked_logits = mask_entity_logits(logits, entity_mask)
            candidate_scores = masked_logits.index_select(dim=1, index=candidate_ids_device)
            candidate_order = torch.argsort(candidate_scores, dim=-1, descending=True)
            ranked_candidate_ids = candidate_ids_device[candidate_order]

            batch_size = samples["input_ids"].size(0)
            for row_idx in range(batch_size):
                target_id = int(samples["target"][row_idx].item())
                gold_answers = normalize_gold_answer_ids(samples["gold_answers"][row_idx], target_id)
                candidate_ids = ranked_candidate_ids[row_idx].detach().cpu().tolist()
                rank_idx = first_correct_candidate_rank(candidate_ids, gold_answers)
                row = None
                if dataset is not None and hasattr(dataset, "data"):
                    row = dataset.data.iloc[int(samples["ids"][row_idx].item())]
                hop_count = row_to_hop_count(row)

                metric_totals["count"] += 1
                sample_mrr = 0.0
                sample_hit1 = 0
                if rank_idx is not None:
                    ranking_value = rank_idx + 1
                    sample_mrr = 1.0 / ranking_value
                    metric_totals["mrr"] += sample_mrr
                    if ranking_value == 1:
                        metric_totals["hit1"] += 1
                        sample_hit1 = 1

                if hop_count is not None:
                    if hop_count not in hop_metrics:
                        hop_metrics[hop_count] = {"mrr": 0.0, "hit1": 0.0, "count": 0}
                    hop_metric = hop_metrics[hop_count]
                    hop_metric["count"] += 1
                    hop_metric["mrr"] += sample_mrr
                    hop_metric["hit1"] += sample_hit1

            denominator = max(1, metric_totals["count"])
            pbar.set_description(
                "%s Eval | MRR: %.6f, Hit@1: %.6f"
                % (
                    split_label,
                    metric_totals["mrr"] / denominator,
                    metric_totals["hit1"] / denominator,
                )
            )

    denominator = max(1, metric_totals["count"])
    metrics = {
        "mrr": metric_totals["mrr"] / denominator,
        "hit1": metric_totals["hit1"] / denominator,
    }
    summary = "[%s] MRR: %.6f, Hit@1: %.6f" % (
        split_name.upper(),
        metrics["mrr"],
        metrics["hit1"],
    )
    tqdm.write(summary)
    logging.info(summary)
    for hop in sorted(hop_metrics):
        hop_metric = hop_metrics[hop]
        if hop_metric["count"] == 0:
            continue
        hop_denominator = hop_metric["count"]
        hop_summary = "[%s %d-hop] MRR: %.6f, Hit@1: %.6f" % (
            split_name.upper(),
            hop,
            hop_metric["mrr"] / hop_denominator,
            hop_metric["hit1"] / hop_denominator,
        )
        tqdm.write(hop_summary)
        logging.info(hop_summary)
    return metrics


def build_loader(dataset, batch_size, shuffle, seed, device, num_workers):
    loader_kwargs = {
        "num_workers": max(0, num_workers),
        "pin_memory": device == "cuda",
        "worker_init_fn": seed_worker,
    }
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        generator=build_dataloader_generator(seed),
        **loader_kwargs,
    )


def checkpoint(args):
    args.dataset = os.path.join("data", args.dataset)
    save_path = os.path.join("models", args.save_dir)
    ckpt_path = os.path.join(save_path, "checkpoint")
    if not os.path.exists(ckpt_path):
        print("Invalid path!")
        return
    if not getattr(args, "train_question_file", None) and not getattr(args, "question_file", None):
        args.train_question_file = getattr(args, "eval_question_file", None)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(save_path, "test_metaqa.log"),
        filemode="w",
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    )
    logging.info(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = args.dataset + "/"
    vocab_path = os.path.join(args.dataset, "vocab.txt")

    train_set = load_dataset(
        Seq2SeqDataset_MetaQA,
        dataset_path,
        vocab_path,
        device,
        args,
        ("train", None),
    )
    test_set = load_dataset(
        TestDataset_MetaQA,
        dataset_path,
        vocab_path,
        device,
        args,
        ("test", None),
    )
    test_loader = build_loader(test_set, args.test_batch_size, False, args.seed + 3, device, args.num_workers)

    entity_candidate_ids = build_entity_candidate_ids(train_set, test_set)
    entity_mask = build_entity_candidate_mask(len(train_set.dictionary), entity_candidate_ids, device)

    model = MetaQAEndpointModel(args, train_set.dictionary)
    state_dict = torch.load(os.path.join(ckpt_path, args.ckpt), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    with torch.no_grad():
        evaluate(model, test_loader, device, args, entity_candidate_ids, entity_mask, split_name="test")


def main():
    args = get_args()
    set_seed(args.seed)
    checkpoint(args)


if __name__ == "__main__":
    main()
