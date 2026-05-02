import json
import os
import copy
from torch.utils.data import Dataset
from dictionary import Dictionary
import torch
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
import pandas as pd
from transformers import BertTokenizer
import ast
import re


DIRECT_REVERSE_SUFFIX = "_reverse"
ANSWER_ENTITY_COLUMNS = ("Answer-Entities", "Answers-Entities", "Answer-Entity")
ANSWER_TEXT_COLUMNS = ("Answers", "Answer")


def _parse_literal_cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
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


def _parse_paths_cell(value):
    parsed = _parse_literal_cell(value)
    return [] if parsed is None else parsed


def _dedupe_preserve_order(values):
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _flatten_scalar_values(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_scalar_values(item))
        return flattened
    text = _normalize_label_text(value)
    return [text] if text is not None else []


def _normalize_list_like_cell(value):
    parsed = _parse_literal_cell(value)
    return _dedupe_preserve_order(_flatten_scalar_values(parsed))


def _get_first_nonempty_row_values(row, columns):
    for column in columns:
        if column not in row.index:
            continue
        values = _normalize_list_like_cell(row.get(column))
        if values:
            return values
    return []


def _get_row_answer_texts(row):
    return _get_first_nonempty_row_values(row, ANSWER_TEXT_COLUMNS)


def _get_row_answer_entities(row):
    entity_values = _get_first_nonempty_row_values(row, ANSWER_ENTITY_COLUMNS)
    if entity_values:
        return entity_values
    return _get_row_answer_texts(row)


def _resolve_entity_token_to_dictionary_id(container, token):
    if token is None:
        return None
    token = str(token)

    if hasattr(container, "entity2id") and token in container.entity2id:
        mapped_token = container.entity2id[token]
        dict_id = container.dictionary.indices.get(mapped_token)
        if dict_id is not None:
            return dict_id

    dict_id = container.dictionary.indices.get(token)
    if dict_id is not None:
        return dict_id

    label2entity_ids = getattr(container, "label2entity_ids", {})
    if token in label2entity_ids:
        for entity_token in label2entity_ids[token]:
            dict_id = _resolve_entity_token_to_dictionary_id(container, entity_token)
            if dict_id is not None:
                return dict_id

    return None


def _iter_row_entity_label_pairs(row):
    source_entities = _normalize_list_like_cell(row.get("Source-Entity"))
    source_labels = _normalize_list_like_cell(row.get("Source"))
    for idx, entity in enumerate(source_entities):
        label = source_labels[idx] if idx < len(source_labels) else None
        yield entity, label

    answer_entities = _get_row_answer_entities(row)
    answer_labels = _get_row_answer_texts(row)
    for idx, entity in enumerate(answer_entities):
        label = answer_labels[idx] if idx < len(answer_labels) else None
        yield entity, label


def _flatten_path_hops(paths):
    tgt_line = []
    for i, hop in enumerate(paths):
        hop = [str(token) for token in hop]
        if i == 0:
            tgt_line.extend(hop)
        else:
            tgt_line.extend(hop[1:])
    return tgt_line


def _is_wikidata_entity(token):
    return isinstance(token, str) and re.fullmatch(r"Q\d+", token) is not None


def _is_wikidata_relation(token):
    return isinstance(token, str) and re.fullmatch(r"P\d+", token) is not None


def _reverse_relation_token(relation):
    relation = str(relation)
    if relation.endswith(DIRECT_REVERSE_SUFFIX):
        return relation
    return f"{relation}{DIRECT_REVERSE_SUFFIX}"


def _normalize_label_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _parse_paraphrased_questions(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = text

    if isinstance(parsed, (list, tuple, set)):
        questions = []
        for question in parsed:
            question_text = _normalize_label_text(question)
            if question_text is not None:
                questions.append(question_text)
        return questions

    question_text = _normalize_label_text(parsed)
    return [question_text] if question_text is not None else []


def _expand_test_paraphrased_rows(dataframe):
    if "Question-Paraphrased" not in dataframe.columns:
        return dataframe

    expanded_rows = []
    for _, row in dataframe.iterrows():
        paraphrased_questions = _parse_paraphrased_questions(row.get("Question-Paraphrased"))
        if not paraphrased_questions:
            expanded_rows.append(row.copy())
            continue
        for question in paraphrased_questions:
            expanded_row = row.copy()
            expanded_row["Question-Paraphrased"] = repr([question])
            expanded_rows.append(expanded_row)

    if not expanded_rows:
        return dataframe.iloc[0:0].copy()
    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def _iter_triple_file(path):
    if not os.path.exists(path):
        return
    with open(path) as fin:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            yield tuple(str(part) for part in parts)


def _resolve_question_csv_path(args, data_path, prefer_eval=False):
    preferred_attr = "eval_question_file" if prefer_eval else "train_question_file"
    csv_file = getattr(args, preferred_attr, None)
    if not csv_file:
        csv_file = getattr(args, "question_file", None)
    if not csv_file:
        raise ValueError(f"args.{preferred_attr} or args.question_file is required")
    return os.path.join(data_path, csv_file)


def _infer_direct_id_mode(dataframe):
    if "Paths" not in dataframe.columns:
        for _, row in dataframe.iterrows():
            source_entities = _normalize_list_like_cell(row.get("Source-Entity"))
            answer_entities = _get_row_answer_entities(row)
            if any(_is_wikidata_entity(token) for token in source_entities + answer_entities):
                return True
        return False
    for value in dataframe["Paths"]:
        try:
            paths = _parse_paths_cell(value)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(paths, list) or not paths:
            continue
        tgt_line = _flatten_path_hops(paths)
        if len(tgt_line) < 3:
            continue
        return (
            _is_wikidata_entity(tgt_line[0])
            and _is_wikidata_relation(tgt_line[1])
            and _is_wikidata_entity(tgt_line[2])
        )
    for _, row in dataframe.iterrows():
        source_entities = _normalize_list_like_cell(row.get("Source-Entity"))
        answer_entities = _get_row_answer_entities(row)
        if any(_is_wikidata_entity(token) for token in source_entities + answer_entities):
            return True
    return False


def _direct_vocab_is_compatible(dictionary, dataframe):
    sample_tokens = []
    for _, row in dataframe.iterrows():
        row_entities = _normalize_list_like_cell(row.get("Source-Entity")) + _get_row_answer_entities(row)
        for token in row_entities:
            if _is_wikidata_entity(token):
                sample_tokens.append(token)
                break
        if sample_tokens:
            break
    if "Paths" in dataframe.columns:
        for value in dataframe["Paths"]:
            try:
                paths = _parse_paths_cell(value)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(paths, list) or not paths:
                continue
            tgt_line = _flatten_path_hops(paths)
            if len(tgt_line) < 3:
                continue
            sample_tokens.extend([tgt_line[0], tgt_line[1], tgt_line[2], _reverse_relation_token(tgt_line[1])])
            break
    return all(token in dictionary.indices for token in sample_tokens)


def _collect_direct_vocab_tokens(data_path, dataframe):
    entities = set()
    relations = set()

    for _, row in dataframe.iterrows():
        for token in _normalize_list_like_cell(row.get("Source-Entity")) + _get_row_answer_entities(row):
            if _is_wikidata_entity(token):
                entities.add(token)

    if "Paths" in dataframe.columns:
        for value in dataframe["Paths"]:
            try:
                paths = _parse_paths_cell(value)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(paths, list):
                continue
            tgt_line = _flatten_path_hops(paths)
            for i, token in enumerate(tgt_line):
                if i % 2 == 0 and _is_wikidata_entity(token):
                    entities.add(token)
                elif i % 2 == 1 and _is_wikidata_relation(token):
                    relations.add(token)

    for file_name in ("train.txt", "valid.txt", "test.txt", "triplets.txt"):
        for h, r, t in _iter_triple_file(os.path.join(data_path, file_name)):
            if _is_wikidata_entity(h):
                entities.add(h)
            if _is_wikidata_relation(r):
                relations.add(r)
            if _is_wikidata_entity(t):
                entities.add(t)

    reverse_relations = {_reverse_relation_token(relation) for relation in relations}
    return entities, relations, reverse_relations


def _add_entity_label_mapping(container, entity_token, label):
    entity_token = str(entity_token)
    label_text = _normalize_label_text(label)
    if label_text is None:
        return
    container.label2entity_ids.setdefault(label_text, [])
    if entity_token not in container.label2entity_ids[label_text]:
        container.label2entity_ids[label_text].append(entity_token)


def _add_direct_id_entity(container, token, label=None):
    token = str(token)
    if not _is_wikidata_entity(token):
        return
    container.entity2id[token] = token
    label_text = _normalize_label_text(label)
    if label_text is not None:
        container.id2entity[token] = label_text
    else:
        container.id2entity.setdefault(token, token)
    _add_entity_label_mapping(container, token, label_text)


def _add_direct_id_relation(container, token, label=None):
    token = str(token)
    if not _is_wikidata_relation(token):
        return
    container.relation2id[token] = token
    label_text = _normalize_label_text(label)
    if label_text is not None:
        container.id2relation[token] = label_text
    else:
        container.id2relation.setdefault(token, token)

    rev_token = _reverse_relation_token(token)
    container.relation2id.setdefault(rev_token, rev_token)
    reverse_label = f"{container.id2relation[token]} (reverse)"
    if label_text is not None:
        container.id2relation[rev_token] = reverse_label
    else:
        container.id2relation.setdefault(rev_token, reverse_label)


def _load_direct_id_mappings(container):
    container.relation2id = {}
    container.id2relation = {}

    for _, row in container.data.iterrows():
        for entity_token, label in _iter_row_entity_label_pairs(row):
            _add_direct_id_entity(container, entity_token, label)

        try:
            paths = _parse_paths_cell(row.get("Paths"))
        except (ValueError, SyntaxError):
            paths = []
        try:
            path_labels = _parse_paths_cell(row.get("Paths-Label"))
        except (ValueError, SyntaxError):
            path_labels = []

        if not isinstance(paths, list):
            continue
        if not isinstance(path_labels, list):
            path_labels = []

        for i, hop in enumerate(paths):
            if not isinstance(hop, (list, tuple)) or len(hop) < 3:
                continue
            label_hop = path_labels[i] if i < len(path_labels) else ()
            if not isinstance(label_hop, (list, tuple)):
                label_hop = ()
            _add_direct_id_entity(container, hop[0], label_hop[0] if len(label_hop) > 0 else None)
            _add_direct_id_relation(container, hop[1], label_hop[1] if len(label_hop) > 1 else None)
            _add_direct_id_entity(container, hop[2], label_hop[2] if len(label_hop) > 2 else None)

    for _, row in container.data.iterrows():
        for token in _normalize_list_like_cell(row.get("Source-Entity")) + _get_row_answer_entities(row):
            _add_direct_id_entity(container, token)

    if "Paths" not in container.data.columns:
        return

    for value in container.data["Paths"]:
        try:
            paths = _parse_paths_cell(value)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(paths, list):
            continue
        tgt_line = _flatten_path_hops(paths)
        for i, token in enumerate(tgt_line):
            if i % 2 == 0:
                _add_direct_id_entity(container, token)
            else:
                _add_direct_id_relation(container, token)


def _load_standard_mappings(container):
    with open(container.data_path + "entity2id.txt") as f:
        for line in f:
            e, eid = line.strip().split('\t')
            container.entity2id[e] = eid
            container.id2entity[eid] = e

    container.relation2id = {}
    container.id2relation = {}
    relation_rows = []
    with open(container.data_path + "relation2id.txt") as f:
        for line in f:
            r, rid = line.strip().split('\t')
            relation_rows.append((r, rid))
    num_relations = len(relation_rows)
    for r, rid in relation_rows:
        container.relation2id[r] = 'R' + rid
        container.id2relation["R" + rid] = r
        container.id2relation["R" + str(int(rid) + num_relations)] = f"{r} (reverse)"

    for _, row in container.data.iterrows():
        for entity_token, label in _iter_row_entity_label_pairs(row):
            if entity_token in container.entity2id:
                _add_entity_label_mapping(container, entity_token, label)


def _load_raw_triples_into_valid_dict(container, file_name, valid_dict, vocab_size, eos):
    path = os.path.join(container.data_path, file_name)
    if not os.path.exists(path):
        return
    with open(path, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            h, r, t = (str(part) for part in parts)
            if h not in container.dictionary.indices or r not in container.dictionary.indices or t not in container.dictionary.indices:
                continue
            container._add_valid_triple(valid_dict, h, r, t, vocab_size, eos)
            rev_r = _reverse_relation_token(r)
            if rev_r in container.dictionary.indices:
                container._add_valid_triple(valid_dict, t, rev_r, h, vocab_size, eos)

class Seq2SeqDataset(Dataset):
    _tokenizer = None

    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", args=None, split: str = None):
        self.data_path = data_path

        self.csv_file = _resolve_question_csv_path(args, data_path, prefer_eval=False)

        self.data = pd.read_csv(self.csv_file)
        if split is not None:
            self.data = self.data[self.data["SplitLabel"] == split].reset_index(drop=True)
        self.has_paths = "Paths" in self.data.columns
        self.direct_id_mode = _infer_direct_id_mode(self.data)

        if Seq2SeqDataset._tokenizer is None:
            Seq2SeqDataset._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = Seq2SeqDataset._tokenizer

        self.max_q_len = getattr(args, "max_q_len", 32)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
            if self.direct_id_mode and not _direct_vocab_is_compatible(self.dictionary, self.data):
                raise ValueError("Existing vocab file does not match direct QID/PID tokens")
        except (FileNotFoundError, ValueError):
            self.dictionary = Dictionary()
            self._init_vocab()
        self._load_mappings()

        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
        self.smart_filter = args.smart_filter
        self.args = args
    
    def __len__(self):
        return len(self.data)

    def _init_vocab(self):
        self.dictionary.add_symbol('LOOP')
        if self.direct_id_mode:
            entities, relations, reverse_relations = _collect_direct_vocab_tokens(self.data_path, self.data)
            for relation in sorted(relations):
                self.dictionary.add_symbol(relation)
            for relation in sorted(reverse_relations):
                self.dictionary.add_symbol(relation)
            for entity in sorted(entities):
                self.dictionary.add_symbol(entity)
            self.dictionary.save(self.vocab_file)
            return
        N = 0
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                N += 1
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                r, rid = line.strip().split('\t')
                rev_rid = int(rid) + N # adding reverse relations IDs
                self.dictionary.add_symbol('R'+rid)
                self.dictionary.add_symbol('R'+str(rev_rid))
        with open(self.data_path+'entity2id.txt') as fin:
            for line in fin:
                e, eid = line.strip().split('\t')
                self.dictionary.add_symbol(eid)
        self.dictionary.save(self.vocab_file)
    
    def _load_mappings(self):
        self.entity2id = {}
        self.id2entity = {}
        self.label2entity_ids = {}

        if self.direct_id_mode:
            _load_direct_id_mappings(self)
            return

        _load_standard_mappings(self)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        if self.args.train_paraphrased:
            set_question = str(row["Question-Paraphrased"])
            question = ast.literal_eval(set_question)[-1]
        else:
            question = str(row["Question"])

        if "Paths" not in row.index:
            raise ValueError(
                "SQUIRE path-supervised training requires a Paths column. "
                "This question file appears to be a multi-answer evaluation file without Paths."
            )
        paths = _parse_paths_cell(row["Paths"])
        tgt_line = _flatten_path_hops(paths)
        tgt_line_ids = []
        for i, token in enumerate(tgt_line):
            if i % 2 == 0:  # entity
                try:
                    tgt_line_ids.append(self.entity2id[token])
                except KeyError:
                    raise ValueError(f"Unknown entity: {token}")
            else:  # relation
                try:
                    tgt_line_ids.append(self.relation2id[token])
                except KeyError:
                    raise ValueError(f"Unknown relation: {token}")

        encoded_question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_q_len,
            return_tensors='pt'
        )

        target_id = self.dictionary.encode_line(tgt_line_ids)
        l = len(target_id)
        mask = torch.ones_like(target_id)
        for i in range(0, l-2):
            if i % 2 == 0:
                continue
            if random.random() < self.args.prob: # randomly replace with prob
                target_id[i] = random.randint(0, self.len_vocab - 1)
                mask[i] = 0
        return {
            "id": index,
            "tgt_length": len(target_id),
            "input_ids": encoded_question["input_ids"].squeeze(0),
            "attention_mask": encoded_question["attention_mask"].squeeze(0),
            "target": target_id,
            "mask": mask,
        }

    def collate_fn(self, samples):
        lens = [sample["tgt_length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens)

        input_ids = torch.stack([s["input_ids"] for s in samples])
        attention_mask = torch.stack([s["attention_mask"] for s in samples])

        prev_outputs = torch.LongTensor(bsz, max_len)
        mask = torch.zeros(bsz, max_len)
 
        prev_outputs.fill_(self.dictionary.pad())
        prev_outputs[:, 0].fill_(self.dictionary.bos())
        target = copy.deepcopy(prev_outputs)

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            target_ids = sample["target"]
            input_ids[idx] = sample["input_ids"]
            attention_mask[idx] = sample["attention_mask"]
            prev_outputs[idx, 1:sample["tgt_length"]] = target_ids[: -1]
            target[idx, 0: sample["tgt_length"]] = target_ids
            mask[idx, 0: sample["tgt_length"]] = sample["mask"]

        # Keep worker processes CPU-only. The training loop moves tensor batches
        # to the target device after DataLoader returns them.
        return {
            "ids": torch.LongTensor(ids),
            "lengths": torch.LongTensor(lens),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prev_outputs": prev_outputs,
            "target": target,
            "mask": mask,
        }

    def _add_valid_triple(self, valid_dict, h, r, t, vocab_size, eos):
        hid = self.dictionary.indices[h]
        rid = self.dictionary.indices[r]
        tid = self.dictionary.indices[t]
        e = hid
        er = vocab_size * rid + hid
        if e not in valid_dict:
            if self.smart_filter:
                valid_dict[e] = -30 * torch.ones([vocab_size])
            else:
                valid_dict[e] = [eos, ]
        if er not in valid_dict:
            if self.smart_filter:
                valid_dict[er] = -30 * torch.ones([vocab_size])
            else:
                valid_dict[er] = []
        if self.smart_filter:
            valid_dict[e][rid] = 0
            valid_dict[e][eos] = 0
            valid_dict[er][tid] = 0
        else:
            valid_dict[e].append(rid)
            valid_dict[er].append(tid)

    def get_next_valid(self):
        if self.direct_id_mode:
            train_valid = dict()
            eval_valid = dict()
            vocab_size = len(self.dictionary)
            eos = self.dictionary.eos()

            train_file = "train.txt" if os.path.exists(os.path.join(self.data_path, "train.txt")) else "triplets.txt"
            _load_raw_triples_into_valid_dict(self, train_file, train_valid, vocab_size, eos)
            _load_raw_triples_into_valid_dict(self, train_file, eval_valid, vocab_size, eos)
            _load_raw_triples_into_valid_dict(self, "valid.txt", eval_valid, vocab_size, eos)
            _load_raw_triples_into_valid_dict(self, "test.txt", eval_valid, vocab_size, eos)
            return train_valid, eval_valid

        train_valid = dict()
        eval_valid = dict()
        vocab_size = len(self.dictionary)
        eos = self.dictionary.eos()
        with open(self.data_path+'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in train_valid:
                    if self.smart_filter:
                        train_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[e] = [eos, ]
                if er not in train_valid:
                    if self.smart_filter:
                        train_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[er] = []
                if self.smart_filter:
                    train_valid[e][rid] = 0
                    train_valid[e][eos] = 0
                    train_valid[er][tid] = 0
                else:
                    train_valid[e].append(rid)
                    train_valid[er].append(tid)
        with open(self.data_path+'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in eval_valid:
                    if self.smart_filter:
                        eval_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[e] = [eos, ]
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[e][rid] = 0
                    eval_valid[e][eos] = 0
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[e].append(rid)
                    eval_valid[er].append(tid)
        with open(self.data_path+'valid_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)
        with open(self.data_path+'test_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)
        return train_valid, eval_valid
                
class TestDataset(Dataset):
    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", src_file=None, args=None, split: str = None):
        self.data_path = data_path
        self.csv_file = _resolve_question_csv_path(args, data_path, prefer_eval=True)

        self.data = pd.read_csv(self.csv_file)
        if split is not None:
            self.data = self.data[self.data["SplitLabel"] == split].reset_index(drop=True)
        if getattr(args, "test_paraphrased", False):
            self.data = _expand_test_paraphrased_rows(self.data)
        self.direct_id_mode = _infer_direct_id_mode(self.data)
        if Seq2SeqDataset._tokenizer is None:
            Seq2SeqDataset._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = Seq2SeqDataset._tokenizer
        self.max_q_len = getattr(args, "max_q_len", 32)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
            if self.direct_id_mode and not _direct_vocab_is_compatible(self.dictionary, self.data):
                raise ValueError("Existing vocab file does not match direct QID/PID tokens")
        except (FileNotFoundError, ValueError):
            self.dictionary = Dictionary()
            Seq2SeqDataset._init_vocab(self)
        Seq2SeqDataset._load_mappings(self)
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        if self.args.test_paraphrased:
            paraphrased_questions = _parse_paraphrased_questions(row.get("Question-Paraphrased"))
            question = paraphrased_questions[0] if paraphrased_questions else str(row["Question"])
        else:
            question = str(row["Question"])
        answer_tokens = _get_row_answer_entities(row)
        if not answer_tokens:
            raise ValueError(f"No answer entity found for row {index} in {self.csv_file}")
        head_tokens = _normalize_list_like_cell(row.get("Source-Entity"))
        if not head_tokens:
            raise ValueError(f"No source entity found for row {index} in {self.csv_file}")
        head = head_tokens[0]
        encoded_question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_q_len,
            return_tensors='pt'
        )
        gold_answer_ids = []
        for answer in answer_tokens:
            answer_id = _resolve_entity_token_to_dictionary_id(self, answer)
            if answer_id is None:
                continue
            gold_answer_ids.append(answer_id)
        gold_answer_ids = _dedupe_preserve_order(gold_answer_ids)
        if not gold_answer_ids:
            raise ValueError(f"Unknown answer entities for row {index}: {answer_tokens}")

        head_id = _resolve_entity_token_to_dictionary_id(self, head)
        if head_id is None:
            raise ValueError(f"Unknown head entity: {head}")

        target_id = torch.tensor([gold_answer_ids[0]], dtype=torch.long)
        return {
            "id": index,
            "input_ids": encoded_question["input_ids"].squeeze(0),
            "attention_mask": encoded_question["attention_mask"].squeeze(0),
            "target": target_id,
            "head_id": torch.tensor(head_id, dtype=torch.long),
            "gold_answers": torch.tensor(gold_answer_ids, dtype=torch.long),
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        input_ids = torch.stack([sample["input_ids"] for sample in samples])
        attention_mask = torch.stack([sample["attention_mask"] for sample in samples])
        target = torch.LongTensor(bsz, 1)
        head_id = torch.LongTensor(bsz)
        max_gold_answers = max(max(sample["gold_answers"].numel(), 1) for sample in samples)
        gold_answers = torch.full((bsz, max_gold_answers), -1, dtype=torch.long)

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            target_ids = sample["target"]
            input_ids[idx] = sample["input_ids"]
            attention_mask[idx] = sample["attention_mask"]
            target[idx, 0] = target_ids[0]
            head_id[idx] = sample["head_id"]
            gold_answers[idx, : sample["gold_answers"].numel()] = sample["gold_answers"]
        
        # Keep worker processes CPU-only. The evaluation loop moves tensor
        # batches to the target device after DataLoader returns them.
        return {
            "ids": torch.LongTensor(ids),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
            "head_id": head_id,
            "gold_answers": gold_answers,
        }

class Seq2SeqDataset_MetaQA(Dataset):
    _tokenizer = None

    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", args=None, split: str = None):
        self.data_path = data_path

        self.csv_file = _resolve_question_csv_path(args, data_path, prefer_eval=False)

        self.data = pd.read_csv(self.csv_file)
        if split is not None and "SplitLabel" in self.data.columns:
            self.data = self.data[self.data["SplitLabel"] == split].reset_index(drop=True)
        self.has_paths = "Paths" in self.data.columns
        self.direct_id_mode = _infer_direct_id_mode(self.data)

        if Seq2SeqDataset._tokenizer is None:
            Seq2SeqDataset._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = Seq2SeqDataset._tokenizer

        self.max_q_len = getattr(args, "max_q_len", 32)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
            if self.direct_id_mode and not _direct_vocab_is_compatible(self.dictionary, self.data):
                raise ValueError("Existing vocab file does not match direct QID/PID tokens")
        except (FileNotFoundError, ValueError):
            self.dictionary = Dictionary()
            Seq2SeqDataset._init_vocab(self)
        Seq2SeqDataset._load_mappings(self)

        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
        self.smart_filter = args.smart_filter
        self.args = args
        self.entity_candidate_ids = self._build_entity_candidate_ids()
    
    def __len__(self):
        return len(self.data)

    def _build_entity_candidate_ids(self):
        candidate_ids = set()

        for entity_token, mapped_token in self.entity2id.items():
            dict_id = self.dictionary.indices.get(str(mapped_token))
            if dict_id is None:
                dict_id = self.dictionary.indices.get(str(entity_token))
            if dict_id is not None:
                candidate_ids.add(int(dict_id))

        if not candidate_ids:
            for mapped_token in self.id2entity.keys():
                dict_id = self.dictionary.indices.get(str(mapped_token))
                if dict_id is not None:
                    candidate_ids.add(int(dict_id))

        if not candidate_ids:
            raise ValueError(f"No entity candidates found for MetaQA dataset at {self.csv_file}")

        return torch.tensor(sorted(candidate_ids), dtype=torch.long)

    def _get_question_text(self, row, train=False):
        if train and getattr(self.args, "train_paraphrased", False):
            paraphrased_questions = _parse_paraphrased_questions(row.get("Question-Paraphrased"))
            if paraphrased_questions:
                return paraphrased_questions[-1]
        if (not train) and getattr(self.args, "test_paraphrased", False):
            paraphrased_questions = _parse_paraphrased_questions(row.get("Question-Paraphrased"))
            if paraphrased_questions:
                return paraphrased_questions[0]
        return str(row["Question"])

    def _resolve_gold_answer_ids(self, row, index):
        answer_tokens = _get_row_answer_entities(row)
        if not answer_tokens:
            raise ValueError(f"No answer entity found for row {index} in {self.csv_file}")

        gold_answer_ids = []
        for answer in answer_tokens:
            candidate_tokens = [answer]
            if isinstance(answer, str):
                stripped = answer.strip()
                if stripped and any(delimiter in stripped for delimiter in ("|", ";")):
                    candidate_tokens = [part.strip() for part in re.split(r"[|;]", stripped) if part.strip()]

            for candidate_token in candidate_tokens:
                answer_id = _resolve_entity_token_to_dictionary_id(self, candidate_token)
                if answer_id is not None:
                    gold_answer_ids.append(int(answer_id))

        gold_answer_ids = _dedupe_preserve_order(gold_answer_ids)
        if not gold_answer_ids:
            raise ValueError(f"Unknown answer entities for row {index}: {answer_tokens}")

        return gold_answer_ids

    def _resolve_head_id(self, row):
        head_tokens = _normalize_list_like_cell(row.get("Source-Entity"))
        if not head_tokens:
            head_tokens = _normalize_list_like_cell(row.get("Source"))

        for head_token in head_tokens:
            head_id = _resolve_entity_token_to_dictionary_id(self, head_token)
            if head_id is not None:
                return int(head_id)
        return -1

    def _init_vocab(self):
        self.dictionary.add_symbol('LOOP')
        if self.direct_id_mode:
            entities, relations, reverse_relations = _collect_direct_vocab_tokens(self.data_path, self.data)
            for relation in sorted(relations):
                self.dictionary.add_symbol(relation)
            for relation in sorted(reverse_relations):
                self.dictionary.add_symbol(relation)
            for entity in sorted(entities):
                self.dictionary.add_symbol(entity)
            self.dictionary.save(self.vocab_file)
            return
        N = 0
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                N += 1
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                r, rid = line.strip().split('\t')
                rev_rid = int(rid) + N # adding reverse relations IDs
                self.dictionary.add_symbol('R'+rid)
                self.dictionary.add_symbol('R'+str(rev_rid))
        with open(self.data_path+'entity2id.txt') as fin:
            for line in fin:
                e, eid = line.strip().split('\t')
                self.dictionary.add_symbol(eid)
        self.dictionary.save(self.vocab_file)
    
    def _load_mappings(self):
        self.entity2id = {}
        self.id2entity = {}
        self.label2entity_ids = {}

        if self.direct_id_mode:
            _load_direct_id_mappings(self)
            return

        _load_standard_mappings(self)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = self._get_question_text(row, train=True)
        gold_answer_ids = self._resolve_gold_answer_ids(row, index)
        head_id = self._resolve_head_id(row)

        encoded_question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_q_len,
            return_tensors='pt'
        )

        return {
            "id": index,
            "input_ids": encoded_question["input_ids"].squeeze(0),
            "attention_mask": encoded_question["attention_mask"].squeeze(0),
            "head_id": torch.tensor(head_id, dtype=torch.long),
            "gold_answers": torch.tensor(gold_answer_ids, dtype=torch.long),
            "target": torch.tensor(gold_answer_ids[0], dtype=torch.long),
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        input_ids = torch.stack([sample["input_ids"] for sample in samples])
        attention_mask = torch.stack([sample["attention_mask"] for sample in samples])
        head_id = torch.LongTensor(bsz)
        target = torch.LongTensor(bsz)
        max_gold_answers = max(max(sample["gold_answers"].numel(), 1) for sample in samples)
        gold_answers = torch.full((bsz, max_gold_answers), -1, dtype=torch.long)

        ids = []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            head_id[idx] = int(sample["head_id"].item())
            target[idx] = int(sample["target"].item())
            gold_answers[idx, : sample["gold_answers"].numel()] = sample["gold_answers"]

        return {
            "ids": torch.LongTensor(ids),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_id": head_id,
            "gold_answers": gold_answers,
            "target": target,
        }

    def _add_valid_triple(self, valid_dict, h, r, t, vocab_size, eos):
        hid = self.dictionary.indices[h]
        rid = self.dictionary.indices[r]
        tid = self.dictionary.indices[t]
        e = hid
        er = vocab_size * rid + hid
        if e not in valid_dict:
            if self.smart_filter:
                valid_dict[e] = -30 * torch.ones([vocab_size])
            else:
                valid_dict[e] = [eos, ]
        if er not in valid_dict:
            if self.smart_filter:
                valid_dict[er] = -30 * torch.ones([vocab_size])
            else:
                valid_dict[er] = []
        if self.smart_filter:
            valid_dict[e][rid] = 0
            valid_dict[e][eos] = 0
            valid_dict[er][tid] = 0
        else:
            valid_dict[e].append(rid)
            valid_dict[er].append(tid)

    def get_next_valid(self):
        if self.direct_id_mode:
            train_valid = dict()
            eval_valid = dict()
            vocab_size = len(self.dictionary)
            eos = self.dictionary.eos()

            train_file = "train.txt" if os.path.exists(os.path.join(self.data_path, "train.txt")) else "triplets.txt"
            _load_raw_triples_into_valid_dict(self, train_file, train_valid, vocab_size, eos)
            _load_raw_triples_into_valid_dict(self, train_file, eval_valid, vocab_size, eos)
            _load_raw_triples_into_valid_dict(self, "valid.txt", eval_valid, vocab_size, eos)
            _load_raw_triples_into_valid_dict(self, "test.txt", eval_valid, vocab_size, eos)
            return train_valid, eval_valid

        train_valid = dict()
        eval_valid = dict()
        vocab_size = len(self.dictionary)
        eos = self.dictionary.eos()
        with open(self.data_path+'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in train_valid:
                    if self.smart_filter:
                        train_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[e] = [eos, ]
                if er not in train_valid:
                    if self.smart_filter:
                        train_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[er] = []
                if self.smart_filter:
                    train_valid[e][rid] = 0
                    train_valid[e][eos] = 0
                    train_valid[er][tid] = 0
                else:
                    train_valid[e].append(rid)
                    train_valid[er].append(tid)
        with open(self.data_path+'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in eval_valid:
                    if self.smart_filter:
                        eval_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[e] = [eos, ]
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[e][rid] = 0
                    eval_valid[e][eos] = 0
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[e].append(rid)
                    eval_valid[er].append(tid)
        with open(self.data_path+'valid_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)
        with open(self.data_path+'test_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)
        return train_valid, eval_valid
                
class TestDataset_MetaQA(Dataset):
    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", src_file=None, args=None, split: str = None):
        self.data_path = data_path
        self.csv_file = _resolve_question_csv_path(args, data_path, prefer_eval=True)

        self.data = pd.read_csv(self.csv_file)
        if split is not None and "SplitLabel" in self.data.columns:
            self.data = self.data[self.data["SplitLabel"] == split].reset_index(drop=True)
        if getattr(args, "test_paraphrased", False):
            self.data = _expand_test_paraphrased_rows(self.data)
        self.direct_id_mode = _infer_direct_id_mode(self.data)
        if Seq2SeqDataset._tokenizer is None:
            Seq2SeqDataset._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = Seq2SeqDataset._tokenizer
        self.max_q_len = getattr(args, "max_q_len", 32)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
            if self.direct_id_mode and not _direct_vocab_is_compatible(self.dictionary, self.data):
                raise ValueError("Existing vocab file does not match direct QID/PID tokens")
        except (FileNotFoundError, ValueError):
            self.dictionary = Dictionary()
            Seq2SeqDataset._init_vocab(self)
        Seq2SeqDataset._load_mappings(self)
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
        self.args = args
        self.entity_candidate_ids = Seq2SeqDataset_MetaQA._build_entity_candidate_ids(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = Seq2SeqDataset_MetaQA._get_question_text(self, row, train=False)
        gold_answer_ids = Seq2SeqDataset_MetaQA._resolve_gold_answer_ids(self, row, index)
        head_id = Seq2SeqDataset_MetaQA._resolve_head_id(self, row)
        encoded_question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_q_len,
            return_tensors='pt'
        )
        return {
            "id": index,
            "input_ids": encoded_question["input_ids"].squeeze(0),
            "attention_mask": encoded_question["attention_mask"].squeeze(0),
            "target": torch.tensor(gold_answer_ids[0], dtype=torch.long),
            "head_id": torch.tensor(head_id, dtype=torch.long),
            "gold_answers": torch.tensor(gold_answer_ids, dtype=torch.long),
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        input_ids = torch.stack([sample["input_ids"] for sample in samples])
        attention_mask = torch.stack([sample["attention_mask"] for sample in samples])
        target = torch.LongTensor(bsz)
        head_id = torch.LongTensor(bsz)
        max_gold_answers = max(max(sample["gold_answers"].numel(), 1) for sample in samples)
        gold_answers = torch.full((bsz, max_gold_answers), -1, dtype=torch.long)

        ids = []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            target[idx] = int(sample["target"].item())
            head_id[idx] = int(sample["head_id"].item())
            gold_answers[idx, : sample["gold_answers"].numel()] = sample["gold_answers"]
        
        return {
            "ids": torch.LongTensor(ids),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
            "head_id": head_id,
            "gold_answers": gold_answers,
        }
