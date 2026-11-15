"""Lossless source-offset BIO inputs, shared by training and text-only inference.

Each Unicode word, whitespace run and punctuation character is tokenized on its
own. Boundaries depend ONLY on text, never on annotations. This intentionally
changes native BPE segmentation: it prevents a currency/number token from also
containing a preceding space outside the entity. No input is truncated.
"""
from functools import lru_cache
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

LABELS = ("PRIVATE_PRICE", "NEGOTIATION_LIMIT", "INDIVIDUAL_DISCOUNT",
          "RETRO_BONUS", "PAYMENT_TERMS", "INTERNAL_MARGIN")
BIO_LABELS = ("O",) + tuple(f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I"))
LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
IGNORE_INDEX = -100
SEPARATOR = "\n\n"
PREPROCESSING = "unicode-lexical-segments-v1/all-subtokens/final-copy-only"
SEGMENTS = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)


def read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str):
                raise ValueError(f"Invalid id/text in {path}:{lineno}")
            rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate row ID in {path}")
    return rows


@lru_cache(maxsize=32768)
def _segment_tokens(tokenizer, surface):
    encoded = tokenizer(surface, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True)
    ids = tuple(encoded["input_ids"])
    offsets = tuple(tuple(x) for x in encoded["offset_mapping"])
    if len(ids) != len(offsets):
        raise ValueError("Tokenizer changed offset count")
    covered = set()
    for a, b in offsets:
        if not 0 <= a < b <= len(surface):
            raise ValueError(f"Tokenizer returned an empty/invalid source offset for {surface!r}")
        covered.update(range(a, b))
    if covered != set(range(len(surface))):
        raise ValueError(f"Tokenizer dropped source characters in {surface!r}")
    return ids, offsets


@lru_cache(maxsize=8)
def _special_template(tokenizer):
    """Derive native BOS/EOS placement without relying on removed v4 methods."""
    plain = tokenizer("x", add_special_tokens=False, truncation=False)["input_ids"]
    wrapped = tokenizer("x", add_special_tokens=True, truncation=False,
                        return_special_tokens_mask=True)
    mask = wrapped["special_tokens_mask"]
    actual = [i for i, special in zip(wrapped["input_ids"], mask) if not special]
    positions = [i for i, special in enumerate(mask) if not special]
    if not positions or actual != plain or positions != list(range(positions[0], positions[-1]+1)):
        raise ValueError("Unsupported tokenizer special-token template")
    return tuple(wrapped["input_ids"][:positions[0]]), tuple(wrapped["input_ids"][positions[-1]+1:])


def _validate_gold(text, spans):
    last = 0
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        a, b = span["start"], span["end"]
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text)
                or a < last or span["label"] not in LABELS or text[a:b] != span["text"]):
            raise ValueError(f"Invalid or overlapping gold span: {span}")
        if text[a].isspace() or text[b-1].isspace():
            raise ValueError("Gold entities must not have external whitespace")
        last = b


def encode_row(row, tokenizer, repeat=2, max_length=None):
    """Return unpadded inputs plus offsets; absent ``spans`` means inference.

    Only final-copy source tokens receive labels (all -100 for inference).
    ``offsets`` maps final-copy tokens to ORIGINAL character positions; all other
    tokens map to None. Metadata contains text/id, never gold/protected spans.
    An overlength example raises instead of silently discarding any text.
    """
    if type(repeat) is not int or repeat not in (1, 2):
        raise ValueError("repeat must be 1 or 2")
    if max_length is not None and (type(max_length) is not int or max_length < 1):
        raise ValueError("max_length must be a positive integer or None")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with source offsets is required")
    text = row["text"]
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    supervised = "spans" in row
    gold = row.get("spans", [])
    _validate_gold(text, gold)
    source_ids, source_offsets = [], []
    for match in SEGMENTS.finditer(text):
        ids, offsets = _segment_tokens(tokenizer, match.group())
        source_ids.extend(ids)
        source_offsets.extend((a+match.start(), b+match.start()) for a, b in offsets)
    source_labels = [0] * len(source_ids)
    assigned = set()
    for span in gold:
        a, b = span["start"], span["end"]
        indices = [i for i, (x, y) in enumerate(source_offsets) if x < b and a < y]
        if (not indices or min(source_offsets[i][0] for i in indices) != a
                or max(source_offsets[i][1] for i in indices) != b
                or any(not a <= source_offsets[i][0] < source_offsets[i][1] <= b for i in indices)):
            raise ValueError(f"Gold is not exactly representable by text-only tokenization: {row.get('id')}: {span}")
        if assigned.intersection(indices):
            raise ValueError("One token belongs to multiple gold entities")
        assigned.update(indices)
        for n, i in enumerate(indices):
            source_labels[i] = LABEL2ID[f"{'B' if n == 0 else 'I'}-{span['label']}"]
    prefix, suffix = _special_template(tokenizer)
    separator = tuple(tokenizer(SEPARATOR, add_special_tokens=False, truncation=False)["input_ids"])
    ids, labels, offsets, prediction_mask = list(prefix), [IGNORE_INDEX]*len(prefix), [None]*len(prefix), [False]*len(prefix)
    for copy in range(repeat):
        if copy:
            ids.extend(separator); labels.extend([IGNORE_INDEX]*len(separator))
            offsets.extend([None]*len(separator)); prediction_mask.extend([False]*len(separator))
        final = copy == repeat-1
        ids.extend(source_ids)
        labels.extend(source_labels if final and supervised else [IGNORE_INDEX]*len(source_ids))
        offsets.extend(source_offsets if final else [None]*len(source_ids))
        prediction_mask.extend([final]*len(source_ids))
    ids.extend(suffix); labels.extend([IGNORE_INDEX]*len(suffix))
    offsets.extend([None]*len(suffix)); prediction_mask.extend([False]*len(suffix))
    if not ids:
        fallback = getattr(tokenizer, "eos_token_id", None)
        if fallback is None:
            raise ValueError("Empty input requires a native BOS/EOS token")
        ids, labels, offsets, prediction_mask = [fallback], [IGNORE_INDEX], [None], [False]
    if max_length is not None and len(ids) > max_length:
        raise ValueError(f"No truncation: {row.get('id')} needs {len(ids)} tokens, max_length={max_length}")
    return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels,
            "offsets": offsets, "prediction_mask": prediction_mask,
            "metadata": {"id": row.get("id"), "text": text, "repeat": repeat,
                         "source_tokens": len(source_ids), "sequence_tokens": len(ids),
                         "preprocessing": PREPROCESSING}}


class DataCollator:
    """Right-pad model fields only; source metadata never enters model.forward."""
    def __init__(self, tokenizer=None, pad_token_id=None, pad_to_multiple_of=None):
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Need a pad_token_id or tokenizer with native pad/EOS")
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of
        if pad_to_multiple_of is not None and (type(pad_to_multiple_of) is not int or pad_to_multiple_of < 1):
            raise ValueError("pad_to_multiple_of must be a positive integer")

    def __call__(self, features):
        import torch
        if not features:
            raise ValueError("Cannot collate an empty batch")
        length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            length = ((length+self.pad_to_multiple_of-1)//self.pad_to_multiple_of)*self.pad_to_multiple_of
        result = {}
        for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", IGNORE_INDEX)):
            result[key] = torch.tensor([list(f[key])+[pad]*(length-len(f[key])) for f in features], dtype=torch.long)
        return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(args):
    """Audit all original data on a real tokenizer, with an oracle BIO roundtrip."""
    from transformers import AutoTokenizer
    try:
        from .evaluate import decode_probabilities
    except ImportError:
        from evaluate import decode_probabilities
    import numpy as np
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    report = {"preprocessing": PREPROCESSING, "repeat": args.repeat, "max_length": args.max_length,
              "tokenizer": str(Path(args.tokenizer).resolve()), "model_revision": args.revision,
              "label_order": list(BIO_LABELS), "tokenizer_files": {}, "data_files": {}, "splits": {},
              "model_forward_calls": 0, "gold_used_for_segmentation": False, "truncated_documents": 0}
    for path in sorted(Path(args.tokenizer).glob("*")):
        if path.is_file() and ("tokenizer" in path.name or path.name in
                               ("special_tokens_map.json", "merges.txt", "vocab.json", "vocab.txt")):
            report["tokenizer_files"][path.name] = sha256(path)
    with (output/"rows.jsonl").open("x", encoding="utf-8") as details:
        for split, expected in (("train", 5000), ("dev", 500), ("test", 1000)):
            path = Path(args.data_dir)/f"{split}.jsonl"
            rows = read_rows(path)
            if len(rows) != expected or {r["split"] for r in rows} != {split}:
                raise ValueError(f"Expected original complete {split} split")
            report["data_files"][split] = sha256(path)
            lengths, gold_count, native_misses = [], 0, 0
            counts = Counter()
            for row in rows:
                encoded = encode_row(row, tokenizer, args.repeat, args.max_length)
                # Inference must have exactly the same input and offsets without gold.
                blind = encode_row({"id": row["id"], "text": row["text"]}, tokenizer, args.repeat, args.max_length)
                for key in ("input_ids", "attention_mask", "offsets", "prediction_mask", "metadata"):
                    if encoded[key] != blind[key]:
                        raise AssertionError(f"Gold leakage in {key}: {row['id']}")
                probabilities = np.zeros((len(encoded["labels"]), len(BIO_LABELS)), dtype=np.float32)
                for i, label in enumerate(encoded["labels"]):
                    probabilities[i, max(label, 0)] = 1.
                predicted = decode_probabilities(blind, probabilities, threshold=0.)
                key = lambda s: (s["label"], s["start"], s["end"])
                if sorted(map(key, predicted)) != sorted(map(key, row["spans"])):
                    raise AssertionError(f"Oracle roundtrip failed: {row['id']}")
                native = tokenizer(row["text"], add_special_tokens=False, truncation=False, return_offsets_mapping=True)
                for span in row["spans"]:
                    relevant = [(a,b) for a,b in native["offset_mapping"] if a < span["end"] and span["start"] < b]
                    native_misses += int(not relevant or min(a for a,b in relevant) != span["start"]
                                         or max(b for a,b in relevant) != span["end"])
                lengths.append(len(encoded["input_ids"])); gold_count += len(row["spans"])
                counts.update(s["label"] for s in row["spans"])
                details.write(json.dumps({"id": row["id"], "sequence_tokens": lengths[-1],
                    "source_tokens": encoded["metadata"]["source_tokens"], "gold_entities": len(row["spans"]),
                    "oracle_exact": True, "inference_input_identical": True}, ensure_ascii=False)+"\n")
            report["splits"][split] = {"documents": len(rows), "gold_entities": gold_count,
                "per_type": dict(counts), "sequence_tokens": {"p50": float(np.percentile(lengths,50)),
                "p95": float(np.percentile(lengths,95)), "max": max(lengths)},
                "native_tokenizer_unrepresentable_gold": native_misses,
                "oracle_exact_documents": len(rows), "inference_input_identical_documents": len(rows)}
            (output/"report.partial.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    report["code_files"] = {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("evaluate.py"))}
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repeat", type=int, choices=(1,2), default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    preflight(parser.parse_args())
