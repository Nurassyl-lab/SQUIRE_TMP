import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import Seq2SeqDataset, TestDataset
from model import TransformerModel
import argparse
import csv
import json
import numpy as np
import os
import random
from tqdm import tqdm
import logging
import ast
import transformers
import math
import matplotlib
from typing import Set, Tuple, Sequence, Dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", default=256, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--test-batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=10, type=int)
    parser.add_argument("--save-dir", default="model_1")
    parser.add_argument("--ckpt", default="ckpt_30.pt")
    parser.add_argument("--dataset", default="FB15K237")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--label-smooth", default=0.5, type=float)
    parser.add_argument("--l-punish", default=False, action="store_true") # during generation, add punishment for length
    parser.add_argument("--beam-size", default=128, type=int) # during generation, beam size
    parser.add_argument("--no-filter-gen", default=False, action="store_true") # during generation, not filter unreachable next token
    parser.add_argument("--test", default=False, action="store_true") # for test mode
    parser.add_argument("--mode", choices=("train", "eval"), default="train", help="run training or evaluation; --test remains supported as an alias for eval")
    parser.add_argument("--encoder", default=False, action="store_true") # only use TransformerEncoder
    parser.add_argument("--trainset", default="6_rev_rule")
    parser.add_argument("--loop", default=False, action="store_true") # add self-loop instead of <eos>
    parser.add_argument("--prob", default=0, type=float) # ratio of replaced token
    parser.add_argument("--max-len", default=3, type=int) # maximum number of hops considered
    parser.add_argument("--iter", default=False, action="store_true") # switch for iterative training
    parser.add_argument("--iter-batch-size", default=128, type=int)
    parser.add_argument("--smart-filter", default=False, action="store_true") # more space consumed, less time; switch on when --filter-gen
    parser.add_argument("--warmup", default=3, type=float) # warmup steps ratio
    parser.add_argument("--self-consistency", default=False, action="store_true") # self-consistency
    parser.add_argument("--output-path", default=False, action="store_true") # output top correct path in a file (for interpretability evaluation)
    parser.add_argument("--validate-during-training", dest="validate_during_training", action="store_true", help="run train/valid evaluation during training")
    parser.add_argument("--validate-interval", default=5, type=int, help="run train/valid evaluation every N epochs when validation is enabled")
    
    # question input related
    parser.add_argument("--question-file", default="kinship_hinton_qa_nhop.csv", type=str, help="fallback question CSV used for both training and evaluation when separate files are not provided")
    parser.add_argument("--train-question-file", default=None, type=str, help="question CSV used by Seq2SeqDataset for training")
    parser.add_argument("--eval-question-file", default=None, type=str, help="question CSV used by TestDataset for validation and test evaluation")
    parser.add_argument("--max-q-len", default=32, type=int, help="maximum number of tokens for the question") # used for Bert
    parser.add_argument("--num-workers", default=0, type=int, help="number of DataLoader worker processes, CPU-only when > 0; set to 0 to disable multiprocessing")
    parser.add_argument("--answer-set-topk", default=1, type=int, help="number of top ranked endpoints used for AnswerF1")
    ###
    # train paraphrased isn't used
    parser.add_argument("--train-paraphrased", default=False, action="store_true", help="Use Paraphrased questions for training.")
    parser.add_argument("--test-paraphrased", default=False, action="store_true", help="Use Paraphrased questions for testing.")

    args = parser.parse_args()
    return args

def safe_lookup(x, rev_dict=None):
    return rev_dict[x] if x in rev_dict else str(x)

def parse_optional_literal(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    return value

def relation_edit_distance(
    pred_relations: Sequence[int],
    gt_relations: Sequence[int],
    special_tokens: Set[int],
    inverse_mapping: Dict[int, int],
) -> int:
    """
    Raw relation-sequence edit distance.

    - removes special relation tokens
    - canonicalizes inverse relation tokens
    - computes Levenshtein distance
    - does NOT normalize
    """
    pred_rels = [
        canon_rel(r, inverse_mapping)
        for r in pred_relations
        if r not in special_tokens
    ]
    gt_rels = list(gt_relations)

    dist, _, _ = edit_distance(pred_rels, gt_rels)
    return dist

def relation_f1(
    pred_relations: Sequence[int],
    gt_relations: Sequence[int],
    special_tokens: Set[int],
    inverse_mapping: Dict[int, int],
) -> Tuple[float, float, float]:
    pred_rels = {
        canon_rel(r, inverse_mapping)
        for r in pred_relations
        if r not in special_tokens
    }
    gt_rels = set(gt_relations)
    return compute_precision_recall_f1(pred_rels, gt_rels)

def canon_rel(r: int, inverse_mapping: Dict[int, int]) -> int:
    return inverse_mapping.get(r, r)

def edit_distance(pred_seq: Sequence[int], gt_seq: Sequence[int]):
    m, n = len(pred_seq), len(gt_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_seq[i - 1] == gt_seq[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n], None, None

def answer_set_f1(predicted_endpoints, gold_answers, eps=1e-8):
    pred_set = set(predicted_endpoints)
    gold_set = set(gold_answers)

    tp = len(pred_set & gold_set)
    precision = tp / (len(pred_set) + eps)
    recall = tp / (len(gold_set) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def _normalize_text_list(value):
    parsed = parse_optional_literal(value)
    if parsed is None:
        return []
    if isinstance(parsed, (list, tuple, set)):
        values = []
        for item in parsed:
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    text = str(parsed).strip()
    return [text] if text else []


def _parse_relation_sequence_value(value):
    parsed = parse_optional_literal(value)
    if parsed is None:
        return []
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    text = str(parsed).strip()
    if not text:
        return []
    if "->" in text:
        return [part.strip() for part in text.split("->") if part.strip()]
    return [text]


def get_row_relation_sequence(row, columns):
    if row is None:
        return []
    for column in columns:
        if column not in row.index:
            continue
        relations = _parse_relation_sequence_value(row.get(column))
        if relations:
            return relations
    return []


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

def get_row_text(row, key, default="N/A"):

    if row is None or key not in row:
        return default
    value = row[key]
    if value is None:
        return default

    if key == "Question-Paraphrased":
        value = ast.literal_eval(str(value))[-1]

    if isinstance(value, float) and math.isnan(value):
        return default

    text = str(value).strip()
    return text if text else default

def decode_symbol(symbol, dataset=None):
    if dataset is not None:
        if hasattr(dataset, "id2entity") and symbol in dataset.id2entity:
            return dataset.id2entity[symbol]
        if hasattr(dataset, "id2relation") and symbol in dataset.id2relation:
            return dataset.id2relation[symbol]
    return symbol

def decode_token(token_id, rev_dict, dataset=None):
    symbol = safe_lookup(int(token_id), rev_dict)
    return decode_symbol(symbol, dataset)


def is_relation_symbol(symbol, dataset=None):
    if dataset is not None and hasattr(dataset, "id2relation") and symbol in dataset.id2relation:
        return True
    return isinstance(symbol, str) and symbol.startswith("R")

def format_generated_path(head_label, path_tokens, rev_dict, dataset, eos, bos):
    if path_tokens is None:
        return "N/A"

    parts = [head_label]
    pending_relation = None

    for token in path_tokens[1:]:
        token_id = int(token)
        if token_id == eos:
            break
        if token_id == bos:
            continue

        symbol = safe_lookup(token_id, rev_dict)
        label = decode_symbol(symbol, dataset)
        if is_relation_symbol(symbol, dataset):
            pending_relation = label
            continue
        if pending_relation is None:
            continue

        if pending_relation.endswith(" (reverse)"):
            relation_name = pending_relation[: -len(" (reverse)")]
            parts.append(f"<-{relation_name}- {label}")
        else:
            parts.append(f"--{pending_relation}--> {label}")
        pending_relation = None

    if pending_relation is not None:
        parts.append(f"--{pending_relation}--> ?")

    return " ".join(parts)

def compute_precision_recall_f1(
    pred: Set,
    gt: Set,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

def gt_edge_overlap_f1(
    pred_path: Sequence[Tuple[int, int, int]],
    gt_path: Sequence[Tuple[int, int, int]],
    special_tokens: Set[int],
    inverse_mapping: Dict[int, int],
) -> Tuple[float, float, float]:
    """
    Permutation-invariant edge overlap between predicted and gold paths.

    Mirrors the repo behavior:
    - remove special tokens such as NO_OP / STOP / RESTART
    - canonicalize inverse edges back into forward edges
    - compare edge sets
    """
    pred_edges = {
        canon_edge(h, r, t, inverse_mapping)
        for h, r, t in pred_path
        if r not in special_tokens
    }
    gt_edges = {(h, r, t) for h, r, t in gt_path}
    return compute_precision_recall_f1(pred_edges, gt_edges)

def canon_edge(h: int, r: int, t: int, inverse_mapping: Dict[int, int]) -> Tuple[int, int, int]:
    if r in inverse_mapping:
        return (t, inverse_mapping[r], h)
    return (h, r, t)

def path_tokens_to_edges(path_tokens, eos, bos):
    clean = []
    for token in path_tokens:
        token = int(token)
        if token == eos:
            break
        if token == bos:
            continue
        clean.append(token)

    edges = []
    for idx in range(0, len(clean) - 2, 2):
        edges.append((clean[idx], clean[idx + 1], clean[idx + 2]))

    return edges

def path_tokens_to_relations(path_tokens, eos, bos):
    clean = []
    for token in path_tokens:
        token = int(token)
        if token == eos:
            break
        if token == bos:
            continue
        clean.append(token)

    return [clean[idx] for idx in range(1, len(clean), 2)]

def format_gold_answers(row, gold_answer_ids, rev_dict, dataset):
    gold_texts = []
    if row is not None:
        gold_texts = _normalize_text_list(row.get("Answers"))
        if not gold_texts:
            gold_texts = _normalize_text_list(row.get("Answer"))
    if gold_texts:
        return gold_texts
    return [decode_token(answer_id, rev_dict, dataset) for answer_id in gold_answer_ids]

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

def count_parameters(model, exclude_bert=False):
    """Count total and trainable parameters in a PyTorch model."""
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if exclude_bert and 'bert' in name:
            continue
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return total_params, trainable_params

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

def compute_train_metrics(model, samples):
    """
    Compute train batch accuracy metrics in eval mode so reporting does not
    perturb dropout/stateful layers used by optimization.
    """
    was_training = model.training
    if was_training:
        model.eval()
    try:
        with torch.no_grad():
            logits = model.logits(
                samples["input_ids"],
                samples["attention_mask"],
                samples["prev_outputs"]
            )
            pred = logits.argmax(dim=-1)

            target = samples["target"]
            mask = samples["mask"]

            correct = (pred == target) & mask.bool()
            token_acc = correct.sum().float() / mask.sum().float()

            lengths = mask.sum(dim=1).long()
            last_idx = lengths - 2
            batch_indices = torch.arange(pred.size(0), device=pred.device)

            pred_last = pred[batch_indices, last_idx]
            target_last = target[batch_indices, last_idx]
            last_acc = (pred_last == target_last).float().mean()
    finally:
        if was_training:
            model.train()

    return token_acc, last_acc

def build_relation_name_to_id(dataset, rev_dict):
    relation_name_to_id = {}
    if dataset is None or not hasattr(dataset, "id2relation"):
        return relation_name_to_id

    for token_id, symbol in rev_dict.items():
        if symbol in dataset.id2relation:
            relation_name_to_id[dataset.id2relation[symbol]] = token_id
    return relation_name_to_id

def build_inverse_relation_mapping(relation_name_to_id):
    inverse_mapping = {}
    for relation_name, token_id in relation_name_to_id.items():
        if relation_name.endswith(" (reverse)"):
            base_name = relation_name[: -len(" (reverse)")]
            if base_name in relation_name_to_id:
                inverse_mapping[token_id] = relation_name_to_id[base_name]
    return inverse_mapping

def row_path_token_to_id(token, is_relation, dictionary_indices, dataset, relation_name_to_id):
    token = str(token)
    if token in dictionary_indices:
        return dictionary_indices[token]
    if dataset is None:
        return None
    if is_relation and token in relation_name_to_id:
        return relation_name_to_id[token]
    if is_relation and hasattr(dataset, "relation2id") and token in dataset.relation2id:
        mapped = dataset.relation2id[token]
        return dictionary_indices.get(mapped)
    if (not is_relation) and hasattr(dataset, "entity2id") and token in dataset.entity2id:
        mapped = dataset.entity2id[token]
        return dictionary_indices.get(mapped)
    return None

def row_to_gt_edges(row, dictionary_indices, dataset, relation_name_to_id, bos, eos):
    if row is None:
        return []
    gt_paths = parse_optional_literal(row.get("Paths"))
    if not isinstance(gt_paths, list) or not gt_paths:
        gt_paths = parse_optional_literal(row.get("Paths-Label"))
    if not isinstance(gt_paths, list) or not gt_paths:
        return []

    gt_path_tokens = [bos]
    for hop_idx, hop in enumerate(gt_paths):
        if not isinstance(hop, (list, tuple)) or len(hop) < 3:
            continue
        h_id = row_path_token_to_id(hop[0], is_relation=False, dictionary_indices=dictionary_indices, dataset=dataset, relation_name_to_id=relation_name_to_id)
        r_id = row_path_token_to_id(hop[1], is_relation=True, dictionary_indices=dictionary_indices, dataset=dataset, relation_name_to_id=relation_name_to_id)
        t_id = row_path_token_to_id(hop[2], is_relation=False, dictionary_indices=dictionary_indices, dataset=dataset, relation_name_to_id=relation_name_to_id)
        if h_id is None or r_id is None or t_id is None:
            continue
        if hop_idx == 0:
            gt_path_tokens.extend([h_id, r_id, t_id])
        else:
            gt_path_tokens.extend([r_id, t_id])
    gt_path_tokens.append(eos)
    return path_tokens_to_edges(gt_path_tokens, eos, bos)

def row_to_gt_relations(row, dictionary_indices, dataset, relation_name_to_id):
    if row is None:
        return []
    relation_tokens = get_row_relation_sequence(
        row,
        ("Path-Key", "Path_Key", "Query-Relations", "Query-Relation"),
    )
    if not relation_tokens:
        gt_paths = parse_optional_literal(row.get("Paths"))
        if not isinstance(gt_paths, list) or not gt_paths:
            gt_paths = parse_optional_literal(row.get("Paths-Label"))
        if isinstance(gt_paths, list):
            relation_tokens = [
                str(hop[1]).strip()
                for hop in gt_paths
                if isinstance(hop, (list, tuple)) and len(hop) >= 2 and str(hop[1]).strip()
            ]
    relation_ids = []
    for relation in relation_tokens:
        relation_id = row_path_token_to_id(
            relation,
            is_relation=True,
            dictionary_indices=dictionary_indices,
            dataset=dataset,
            relation_name_to_id=relation_name_to_id,
        )
        if relation_id is not None:
            relation_ids.append(relation_id)
    return relation_ids

def row_to_hop_count(row):
    if row is None:
        return None
    for column_name in ("Hops", "Num-Hops", "N-Hop", "hop", "num_hops"):
        if column_name not in row.index:
            continue
        hop_value = row.get(column_name)
        if hop_value is None or (isinstance(hop_value, float) and math.isnan(hop_value)):
            continue
        if isinstance(hop_value, (int, np.integer)):
            return int(hop_value)
        if isinstance(hop_value, float):
            return int(hop_value)
        hop_digits = "".join(ch for ch in str(hop_value) if ch.isdigit())
        if hop_digits:
            return int(hop_digits)
    gt_paths = parse_optional_literal(row.get("Paths"))
    if not isinstance(gt_paths, list) or not gt_paths:
        gt_paths = parse_optional_literal(row.get("Paths-Label"))
    if isinstance(gt_paths, list) and gt_paths:
        return len(gt_paths)
    return None

def evaluate(model, dataloader, device, args, true_triples=None, valid_triples=None, split_name="eval"):
    model.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 2
    restricted_punish = -30
    mrr, hit1, hit3, hit5, hit10, f1_sg, red_sum, relation_f1_sum, path_edit_distance_sum, answer_f1_sum, count = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    vocab_size = len(model.dictionary)
    eos = model.dictionary.eos()
    bos = model.dictionary.bos()
    dictionary_indices = model.dictionary.indices
    rev_dict = {v: k for k, v in model.dictionary.indices.items()}
    lines = []
    dataset = getattr(dataloader, "dataset", None)
    split_label = split_name.title()
    hop_metrics = {
        hop: {
            "count": 0,
            "mrr": 0,
            "hit1": 0,
            "hit3": 0,
            "hit5": 0,
            "hit10": 0,
            "f1_sg": 0,
            "red": 0,
            "f1_rel": 0,
            "ped": 0,
            "answer_f1": 0,
        }
        for hop in (2, 3, 4)
    }
    special_tokens = {bos, eos, model.dictionary.pad()}
    loop_token = dictionary_indices.get("LOOP")
    if loop_token is not None:
        special_tokens.add(loop_token)
    relation_name_to_id = build_relation_name_to_id(dataset, rev_dict)
    inverse_mapping = build_inverse_relation_mapping(relation_name_to_id)

    with tqdm(dataloader, desc=f"{split_label} Eval") as pbar:
        for samples in pbar:
            samples = move_batch_to_device(samples, device)
            pbar.set_description(
                "%s Eval | MRR: %f, Hit@1: %f, Hit@3: %f, Hit@5: %f, Hit@10: %f, F1_SG: %f, RED: %f, F1_REL: %f, PED: %f, AnswerF1: %f"
                % (split_label, mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit5/max(1, count), hit10/max(1, count), f1_sg/max(1, count), red_sum/max(1, count), relation_f1_sum/max(1, count), path_edit_distance_sum/max(1, count), answer_f1_sum/max(1, count))
            )
            batch_size = samples["input_ids"].size(0)

            candidates = [dict() for _ in range(batch_size)]
            candidates_path = [dict() for _ in range(batch_size)]
            input_ids = samples["input_ids"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            attention_mask = samples["attention_mask"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            # The question encoder input is identical for every beam in the
            # batch, so compute it once and expand the cached states instead of
            # re-running BERT for every beam step.
            question_source = model.encode_question(samples["input_ids"], samples["attention_mask"])
            beam_question_source = question_source.unsqueeze(2).repeat(1, 1, beam_size, 1).reshape(
                question_source.size(0),
                batch_size * beam_size,
                question_source.size(-1),
            )
            prefix = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix[:, :, 0].fill_(model.dictionary.bos())
            lprob = torch.zeros([batch_size, beam_size]).to(device)
            clen = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            # first token after BOS predicts head_0
            tmp_input_ids = samples["input_ids"]
            tmp_attention_mask = samples["attention_mask"]
            tmp_prefix = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix[:, 0].fill_(model.dictionary.bos())
            logits = model.logits(
                tmp_input_ids,
                tmp_attention_mask,
                tmp_prefix,
                encoded_source=question_source,
            ).squeeze(1)
            logits = F.log_softmax(logits, dim=-1)
            logits = logits.view(-1, vocab_size)
            argsort = torch.argsort(logits, dim=-1, descending=True)[:, :beam_size]
            prefix[:, :, 1] = argsort[:, :]
            lprob += torch.gather(input=logits, dim=-1, index=argsort)
            clen += 1
            for l in range(2, max_len):
                tmp_prefix = prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob = lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)    
                tmp_clen = clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                all_logits = model.logits(
                    input_ids.view(bb, -1),
                    attention_mask.view(bb, -1),
                    prefix.view(bb, -1),
                    encoded_source=beam_question_source,
                ).view(batch_size, beam_size, max_len, -1)
                logits = torch.gather(input=all_logits, dim=2, index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                # relation slots use the previously predicted head; head slots use (head, relation)
                if args.no_filter_gen:
                    logits = F.log_softmax(logits, dim=-1)
                else:
                    restricted = torch.ones([batch_size, beam_size, vocab_size]) * restricted_punish
                    if l % 2 == 0:
                        index = prefix[:, :, l - 1]
                    else:
                        hid = prefix[:, :, l - 2]
                        rid = prefix[:, :, l - 1]
                        index = vocab_size * rid + hid
                    index = index.cpu().numpy()
                    for i in range(batch_size):
                        for j in range(beam_size):
                            if index[i][j] in true_triples:
                                if args.smart_filter:
                                    restricted[i][j] = true_triples[index[i][j]]
                                else:
                                    idx = torch.LongTensor(true_triples[index[i][j]]).unsqueeze(0)
                                    restricted[i][j] = -restricted_punish * torch.zeros(1, vocab_size).scatter_(1, idx, 1) + restricted_punish
                    logits = F.log_softmax(logits+restricted.to(device), dim=-1)
                argsort = torch.argsort(logits, dim=-1, descending=True)[:, :, :beam_size]
                tmp_clen = tmp_clen + 1
                tmp_prefix = tmp_prefix.scatter_(dim=-1, index=tmp_clen.unsqueeze(-1), src=argsort.unsqueeze(-1))
                tmp_lprob += torch.gather(input=logits, dim=-1, index=argsort)
                tmp_prefix, tmp_lprob, tmp_clen = tmp_prefix.view(batch_size, -1, max_len), tmp_lprob.view(batch_size, -1), tmp_clen.view(batch_size, -1)
                if l == max_len-1:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :(2*beam_size)]
                else:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :beam_size]
                prefix = torch.gather(input=tmp_prefix, dim=1, index=argsort.unsqueeze(-1).repeat(1, 1, max_len))
                lprob = torch.gather(input=tmp_lprob, dim=1, index=argsort)
                clen = torch.gather(input=tmp_clen, dim=1, index=argsort)
                # filter out next token after <end>, add to candidates
                for i in range(batch_size):
                    for j in range(beam_size):
                        if l % 2 == 0 and prefix[i][j][l].item() == eos:
                            candidate_pos = l - 1
                            candidate = prefix[i][j][candidate_pos].item()
                            if l_punish:
                                prob = lprob[i][j].item() / max(1, l // 2)
                            else:
                                prob = lprob[i][j].item()
                            path_array = prefix[i][j, :l + 1].detach().cpu().numpy()
                            lprob[i][j] -= 10000
                            if candidate not in candidates[i]:
                                if args.self_consistency:
                                    candidates[i][candidate] = math.exp(prob)
                                else:
                                    candidates[i][candidate] = prob
                                candidates_path[i][candidate] = path_array
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = path_array
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)
                # no </s> but reach max_len
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate_pos = l if l % 2 == 1 else l - 1
                            candidate = prefix[i][j][candidate_pos].item()
                            if l_punish:
                                prob = lprob[i][j].item() / max(1, (l - 1) // 2)
                            else:
                                prob = lprob[i][j].item()
                            path_array = prefix[i][j, :candidate_pos + 1].detach().cpu().numpy()
                            if candidate not in candidates[i]:
                                if args.self_consistency:
                                    candidates[i][candidate] = math.exp(prob)
                                else:
                                    candidates[i][candidate] = prob
                                candidates_path[i][candidate] = path_array
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = path_array
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:                             
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)
            target = samples["target"].cpu()
            for i in range(batch_size):
                hid = samples["head_id"][i].item()
                index = None
                if index is not None and index in valid_triples:
                    mask = valid_triples[index]
                    for tid in candidates[i].keys():
                        if tid == target[i].item():
                            continue
                        elif args.smart_filter:
                            if mask[tid].item() == 0:
                                candidates[i][tid] -= 100000
                        else:
                            if tid in mask:
                                candidates[i][tid] -= 100000
                count += 1
                candidate_ = sorted(zip(candidates[i].items(), candidates_path[i].items()), key=lambda x:x[0][1], reverse=True)
                candidate_ids = [pair[0][0] for pair in candidate_]
                candidate_path = [pair[1][1] for pair in candidate_]
                target_id = target[i].item()
                gold_answers = normalize_gold_answer_ids(
                    samples["gold_answers"][i] if "gold_answers" in samples else None,
                    target_id,
                )
                rank_idx = first_correct_candidate_rank(candidate_ids, gold_answers)
                row = dataset.data.iloc[int(samples["ids"][i].item())] if dataset is not None and hasattr(dataset, "data") else None
                hop_count = row_to_hop_count(row)
                pred_edges = path_tokens_to_edges(candidate_path[0], eos, bos) if candidate_path else []
                pred_relations = path_tokens_to_relations(candidate_path[0], eos, bos) if candidate_path else []
                gt_edges = row_to_gt_edges(row, dictionary_indices, dataset, relation_name_to_id, bos, eos)
                gt_relations = row_to_gt_relations(row, dictionary_indices, dataset, relation_name_to_id)
                answer_set_topk = max(1, args.answer_set_topk)
                predicted_endpoints = candidate_ids[:answer_set_topk]
                _, _, answer_f1 = answer_set_f1(predicted_endpoints, gold_answers)
                answer_f1_sum += answer_f1
                sample_f1_sg = 0.0
                if gt_edges:
                    _, _, sample_f1_sg = gt_edge_overlap_f1(pred_edges, gt_edges, special_tokens, inverse_mapping)
                    f1_sg += sample_f1_sg
                sample_ped = 0.0
                if gt_edges:
                    pred_edges_seq = pred_edges
                    gt_edges_seq = gt_edges
                    dist, _, _ = edit_distance(pred_edges_seq, gt_edges_seq)
                    sample_ped = dist
                path_edit_distance_sum += sample_ped
                # Multi-answer rows may omit a gold path entirely. In that case
                # there is no reliable gold relation sequence to compare against.
                sample_relation_edit_distance = 0.0
                if gt_relations:
                    sample_relation_edit_distance = relation_edit_distance(pred_relations, gt_relations, special_tokens, inverse_mapping)
                red_sum += sample_relation_edit_distance
                sample_f1_rel = 0.0
                if gt_relations:
                    _, _, sample_f1_rel = relation_f1(pred_relations, gt_relations, special_tokens, inverse_mapping)
                    relation_f1_sum += sample_f1_rel
                if args.test_paraphrased:
                    question_text = get_row_text(row, "Question-Paraphrased")
                else:
                    question_text = get_row_text(row, "Question")

                head_label = get_row_text(row, "Source", decode_token(hid, rev_dict, dataset))
                target_label = " | ".join(format_gold_answers(row, gold_answers, rev_dict, dataset))
                path_token = f"{question_text}\t{head_label} | {target_label}\t"
                sample_mrr = 0.0
                sample_hit1 = 0
                sample_hit3 = 0
                sample_hit5 = 0
                sample_hit10 = 0

                if rank_idx is not None:
                    path = candidate_path[rank_idx]
                    path_token += format_generated_path(head_label, path, rev_dict, dataset, eos, bos) + '\t'
                    path_token += str(rank_idx)
                    ranking_value = 1 + rank_idx
                    sample_mrr = 1 / ranking_value
                    mrr += sample_mrr
                    if ranking_value <= 1:
                        hit1 += 1
                        sample_hit1 = 1
                    if ranking_value <= 3:
                        hit3 += 1
                        sample_hit3 = 1
                    if ranking_value <= 5:
                        hit5 += 1
                        sample_hit5 = 1
                    if ranking_value <= 10:
                        hit10 += 1
                        sample_hit10 = 1
                else:
                    path_token += "wrong"
                if hop_count in hop_metrics:
                    hop_metric = hop_metrics[hop_count]
                    hop_metric["count"] += 1
                    hop_metric["mrr"] += sample_mrr
                    hop_metric["hit1"] += sample_hit1
                    hop_metric["hit3"] += sample_hit3
                    hop_metric["hit5"] += sample_hit5
                    hop_metric["hit10"] += sample_hit10
                    hop_metric["f1_sg"] += sample_f1_sg
                    hop_metric["red"] += sample_relation_edit_distance
                    hop_metric["f1_rel"] += sample_f1_rel
                    hop_metric["ped"] += sample_ped
                    hop_metric["answer_f1"] += answer_f1
                lines.append(path_token+'\n')
    
    if args.output_path and split_name=="test":
        with open(os.path.join(args.save_dir,"test_output_squire.txt"), "w") as f:
            f.writelines(lines)
    metric_denominator = max(1, count)
    summary = "[%s] MRR: %.6f, Hit@1: %.6f, Hit@3: %.6f, Hit@5: %.6f, Hit@10: %.6f, F1_SG: %.6f, RED: %.6f, F1_REL: %.6f, PED: %.6f, AnswerF1: %.6f" % (
        split_name.upper(),
        mrr/metric_denominator,
        hit1/metric_denominator,
        hit3/metric_denominator,
        hit5/metric_denominator,
        hit10/metric_denominator,
        f1_sg/metric_denominator,
        red_sum/metric_denominator,
        relation_f1_sum/metric_denominator,
        path_edit_distance_sum/metric_denominator,
        answer_f1_sum/metric_denominator,
    )
    tqdm.write(summary)
    logging.info(summary)
    for hop in (2, 3, 4):
        hop_metric = hop_metrics[hop]
        if hop_metric["count"] == 0:
            continue
        hop_denominator = hop_metric["count"]
        hop_summary = "[%s %d-hop] MRR: %.6f, Hit@1: %.6f, Hit@3: %.6f, Hit@5: %.6f, Hit@10: %.6f, F1_SG: %.6f, RED: %.6f, F1_REL: %.6f, PED: %.6f, AnswerF1: %.6f" % (
            split_name.upper(),
            hop,
            hop_metric["mrr"] / hop_denominator,
            hop_metric["hit1"] / hop_denominator,
            hop_metric["hit3"] / hop_denominator,
            hop_metric["hit5"] / hop_denominator,
            hop_metric["hit10"] / hop_denominator,
            hop_metric["f1_sg"] / hop_denominator,
            hop_metric["red"] / hop_denominator,
            hop_metric["f1_rel"] / hop_denominator,
            hop_metric["ped"] / hop_denominator,
            hop_metric["answer_f1"] / hop_denominator,
        )
        tqdm.write(hop_summary)
        logging.info(hop_summary)
    return mrr/metric_denominator, hit1/metric_denominator, hit3/metric_denominator, hit5/metric_denominator, hit10/metric_denominator


def save_metric_history(metric_history, save_dir, filename_prefix="metrics"):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{filename_prefix}.json")
    csv_path = os.path.join(save_dir, f"{filename_prefix}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metric_history, f, indent=2)

    keys = list(metric_history.keys())
    rows = []
    length = len(metric_history.get("epoch", []))
    for i in range(length):
        rows.append({key: metric_history[key][i] for key in keys})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_epoch_metrics(metric_history, save_dir):
    epochs = metric_history["epoch"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    metric_specs = [
        ("mrr", "MRR"),
        ("hit1", "Hit@1"),
        ("hit3", "Hit@3"),
        ("hit5", "Hit@5"),
        ("hit10", "Hit@10"),
    ]

    for ax, (key, title) in zip(axes.flat, metric_specs):
        ax.plot(epochs, metric_history[f"train_{key}"], label="Train", linewidth=2)
        ax.plot(epochs, metric_history[f"valid_{key}"], label="Valid", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes.flat[len(metric_specs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_metrics.png"), dpi=200)
    plt.close(fig)

def train(args):
    args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('models', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/train.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.info(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_kwargs = {
        "num_workers": max(0, args.num_workers),
        "pin_memory": device == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, split="train", args=args)
    if not getattr(train_set, "has_paths", True):
        raise ValueError(
            "SQUIRE path-supervised training requires a Paths column. "
            "This question file appears to be a multi-answer evaluation file without Paths."
        )
    valid_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="valid_triples.txt", split="dev", args=args)
    train_eval_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="train_triples.txt", split="train", args=args)
    train_valid, eval_valid = train_set.get_next_valid()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True, generator=build_dataloader_generator(args.seed), **loader_kwargs)
    valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size, collate_fn=valid_set.collate_fn, shuffle=True, generator=build_dataloader_generator(args.seed + 1), **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_set, batch_size=args.test_batch_size, collate_fn=valid_set.collate_fn, shuffle=False, generator=build_dataloader_generator(args.seed + 2), **loader_kwargs)

    model = TransformerModel(args, train_set.dictionary).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_step_num = len(train_loader) * args.num_epoch
    warmup_steps = total_step_num / args.warmup
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_step_num)
    
    best_hit1 = -float("inf")
    best_epoch = -1
    metric_history = {
        "epoch": [],
        "train_mrr": [],
        "valid_mrr": [],
        "train_hit1": [],
        "valid_hit1": [],
        "train_hit3": [],
        "valid_hit3": [],
        "train_hit5": [],
        "valid_hit5": [],
        "train_hit10": [],
        "valid_hit10": [],
    }
    for epoch in range(args.num_epoch):

        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            token_accs = []
            last_accs = []
            for samples in pbar:
                samples = move_batch_to_device(samples, device)
                optimizer.zero_grad()
                loss = model.get_loss(**samples)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

                token_acc, last_acc = compute_train_metrics(model, samples)
                token_accs.append(token_acc.item())
                last_accs.append(last_acc.item())
                pbar.set_description(
                    f"Epoch: {epoch+1}, Loss: {np.mean(losses):.4f}, TokenAcc: {token_acc:.4f}, LastAcc: {last_acc:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}"
                )
        logging.info(
                "[Epoch %d/%d] [train loss: %f] [token acc: %f] [last acc: %f]"
                % (epoch + 1, args.num_epoch, np.mean(losses), np.mean(token_accs), np.mean(last_accs))
                )
        validate_interval = max(1, args.validate_interval)
        should_validate = args.validate_during_training and ((epoch + 1) % validate_interval == 0)
        if should_validate:
            with torch.no_grad():
                train_mrr, train_hit1, train_hit3, train_hit5, train_hit10 = evaluate(model, train_eval_loader, device, args, train_valid, eval_valid, split_name="train")
                valid_mrr, valid_hit1, valid_hit3, valid_hit5, valid_hit10 = evaluate(model, valid_loader, device, args, train_valid, eval_valid, split_name="valid")

            metric_history["epoch"].append(epoch + 1)
            metric_history["train_mrr"].append(train_mrr)
            metric_history["valid_mrr"].append(valid_mrr)
            metric_history["train_hit1"].append(train_hit1)
            metric_history["valid_hit1"].append(valid_hit1)
            metric_history["train_hit3"].append(train_hit3)
            metric_history["valid_hit3"].append(valid_hit3)
            metric_history["train_hit5"].append(train_hit5)
            metric_history["valid_hit5"].append(valid_hit5)
            metric_history["train_hit10"].append(train_hit10)
            metric_history["valid_hit10"].append(valid_hit10)

            logging.info(
                "[Epoch %d Metrics] [Train MRR: %.6f Hit@1: %.6f Hit@3: %.6f Hit@5: %.6f Hit@10: %.6f] "
                "[Valid MRR: %.6f Hit@1: %.6f Hit@3: %.6f Hit@5: %.6f Hit@10: %.6f]",
                epoch + 1,
                train_mrr,
                train_hit1,
                train_hit3,
                train_hit5,
                train_hit10,
                valid_mrr,
                valid_hit1,
                valid_hit3,
                valid_hit5,
                valid_hit10,
            )

            if valid_hit1 > best_hit1:
                best_hit1 = valid_hit1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), ckpt_path + "/best_model.pt".format(best_epoch))
                logging.info("[Checkpoint Saved] [Epoch: %d] [Best Hit@1: %f]", best_epoch, best_hit1)
            else:
                logging.info(
                    "[Checkpoint Skipped] [Epoch: %d] [Hit@1: %f] [Best Epoch: %d] [Best Hit@1: %f]",
                    epoch + 1,
                    valid_hit1,
                    best_epoch,
                    best_hit1,
                )

            plot_epoch_metrics(metric_history, save_path)
            save_metric_history(metric_history, save_path)
        else:
            logging.info(
                "[Epoch %d/%d] train/valid evaluation skipped. validate_during_training=%s validate_interval=%d",
                epoch + 1,
                args.num_epoch,
                args.validate_during_training,
                validate_interval,
            )

def checkpoint(args):
    args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('models', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(ckpt_path):
        print("Invalid path!")
        return
    if not getattr(args, "train_question_file", None) and not getattr(args, "question_file", None):
        args.train_question_file = getattr(args, "eval_question_file", None)
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/test.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_kwargs = {
        "num_workers": max(0, args.num_workers),
        "pin_memory": device == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, split="train", args=args)
    test_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="test_triples.txt", split="test", args=args)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True, generator=build_dataloader_generator(args.seed + 3), **loader_kwargs)
    train_valid, eval_valid = train_set.get_next_valid()
    model = TransformerModel(args, train_set.dictionary)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, args.ckpt)))
    model.args = args
    model = model.to(device)
    
    # Count and print model parameters
    total_params, trainable_params = count_parameters(model, exclude_bert=False)
    total_params_no_bert, trainable_params_no_bert = count_parameters(model, exclude_bert=True)
    print(f"Total parameters (with BERT): {total_params:,}")
    print(f"Trainable parameters (with BERT): {trainable_params:,}")
    print(f"Total parameters (without BERT): {total_params_no_bert:,}")
    print(f"Trainable parameters (without BERT): {trainable_params_no_bert:,}")
    
    with torch.no_grad():
        evaluate(model, test_loader, device, args, train_valid, eval_valid, split_name="test")
    

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if args.test or args.mode == "eval":
        checkpoint(args)
    else:
        train(args)
