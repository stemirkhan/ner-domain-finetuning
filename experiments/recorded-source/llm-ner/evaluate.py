"""Gold-independent BIO decoding and strict CRM span/masking evaluation."""
from collections import Counter, defaultdict
import math

try:
    from .data import BIO_LABELS, LABELS
except ImportError:
    from data import BIO_LABELS, LABELS

THRESHOLDS = tuple(i/20 for i in range(1, 20))
DECODER = {
    "version": "llm-bio-source-v1",
    "labels": list(BIO_LABELS),
    "token_decision": "argmax; ties use the lower fixed label ID",
    "invalid_I": "start a new entity when no matching entity is open",
    "whitespace": "trim source whitespace at entity edges only",
    "score": "geometric mean of probabilities of selected BIO tags",
    "threshold": "span score >= threshold, after BIO assembly",
    "overlap": "descending score, descending length, ascending start/end/label; greedy nonoverlap",
}


def _matrix(values):
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    import numpy as np
    return np.asarray(values, dtype=np.float64)


def candidates_from_probabilities(encoded, probabilities):
    """Use only input text, source offsets and model scores; never read labels."""
    import numpy as np
    probabilities = _matrix(probabilities)
    if probabilities.shape != (len(encoded["input_ids"]), len(BIO_LABELS)):
        raise ValueError("Probability shape must equal [unpadded sequence length, 13]")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5)):
        raise ValueError("Probabilities must be finite normalized distributions")
    text = encoded["metadata"]["text"]
    offsets, mask = encoded["offsets"], encoded["prediction_mask"]
    if len(offsets) != len(probabilities) or len(mask) != len(probabilities):
        raise ValueError("Offset/mask length mismatch")
    candidates, current = [], None

    def flush():
        nonlocal current
        if current is None:
            return
        a, b = current["start"], current["end"]
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b-1].isspace():
            b -= 1
        if a < b:
            candidates.append({"start": a, "end": b, "text": text[a:b], "label": current["label"],
                               "score": math.exp(sum(current["log_scores"])/len(current["log_scores"]))})
        current = None

    for i, (offset, active) in enumerate(zip(offsets, mask)):
        if not active:
            flush()
            continue
        if offset is None or len(offset) != 2:
            raise ValueError("Every prediction token needs a source offset")
        a, b = offset
        if not 0 <= a < b <= len(text):
            raise ValueError("Invalid source offset")
        tag_id = int(probabilities[i].argmax())
        tag = BIO_LABELS[tag_id]
        if tag == "O":
            flush()
            continue
        prefix, label = tag.split("-", 1)
        # UTF-8 byte fallback can give successive tokens identical/overlapping
        # source offsets. A B on that same character must not create two masks.
        overlapping_same = current and current["label"] == label and a < current["end"]
        continuation = (current and current["label"] == label and
                        (prefix == "I" or overlapping_same) and a <= current["end"])
        if not continuation:
            flush()
            current = {"start": a, "end": b, "label": label, "log_scores": []}
        else:
            current["end"] = max(current["end"], b)
        current["log_scores"].append(math.log(max(float(probabilities[i, tag_id]), 1e-300)))
    flush()
    return candidates


def decode_probabilities(encoded, probabilities, threshold=0.5):
    if not math.isfinite(float(threshold)) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    candidates = candidates_from_probabilities(encoded, probabilities)
    ranked = sorted((s for s in candidates if s["score"] >= threshold),
                    key=lambda s: (-s["score"], -(s["end"]-s["start"]), s["start"], s["end"], s["label"]))
    kept = []
    for span in ranked:
        if not any(span["start"] < p["end"] and p["start"] < span["end"] for p in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: (s["start"], s["end"], s["label"]))


def decode_logits(encoded, logits, threshold=0.5):
    import numpy as np
    values = _matrix(logits)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Logits must be a finite matrix")
    exp = np.exp(values-values.max(axis=-1, keepdims=True))
    return decode_probabilities(encoded, exp/exp.sum(axis=-1, keepdims=True), threshold)


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": ratio(tp, tp+fp),
            "recall": ratio(tp, tp+fn), "f1": ratio(2*tp, 2*tp+fp+fn)}


def _validate_spans(text, spans, labels, scored=False):
    last, seen = 0, set()
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        a, b, label = span["start"], span["end"], span["label"]
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text)
                or a < last or label not in labels or text[a:b] != span["text"]
                or (label, a, b) in seen):
            raise ValueError(f"Invalid/overlapping span: {span}")
        if scored and (not isinstance(span.get("score"), (int, float))
                       or not math.isfinite(span["score"]) or not 0 <= span["score"] <= 1):
            raise ValueError("Invalid predicted confidence")
        last = b; seen.add((label, a, b))


def _chars(spans):
    return {i for span in spans for i in range(span["start"], span["end"])}


def evaluate(rows, predictions, labels=LABELS):
    """Match the GLiNER exact-span scorer and its 500+75 public denominator.

    Fixtures may omit contrast metadata; incomplete pairs are reported, not
    counted as successful. Complete frozen CRM splits contain only full pairs.
    """
    if len({r["id"] for r in rows}) != len(rows) or len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate row ID")
    by_id = {r["id"]: r for r in predictions}
    if set(by_id) != {r["id"] for r in rows}:
        raise ValueError("Gold and prediction IDs must match exactly")
    counts = {label: Counter() for label in labels}
    totals, pairs, details = Counter(), defaultdict(dict), []
    subtypes = {kind: Counter() for kind in ("money", "percent")}
    subtype_families = {kind: set() for kind in subtypes}
    errors, public_errors = [], []
    for row in rows:
        pred_row = by_id[row["id"]]
        if pred_row.get("text", row["text"]) != row["text"]:
            raise ValueError("Prediction text changed")
        gold, pred, text = row["spans"], pred_row["predictions"], row["text"]
        _validate_spans(text, gold, labels); _validate_spans(text, pred, labels, scored=True)
        gs = {(s["label"], s["start"], s["end"]) for s in gold}
        ps = {(s["label"], s["start"], s["end"]) for s in pred}
        for label in labels:
            g, p = {s for s in gs if s[0] == label}, {s for s in ps if s[0] == label}
            counts[label].update(tp=len(g&p), fp=len(p-g), fn=len(g-p))
        gc, pc = _chars(gold), _chars(pred)
        covered = sum(set(range(s["start"], s["end"])) <= pc for s in gold)
        totals.update(documents=1, gold_entities=len(gold), gold_fully_masked=covered,
                      gold_chars=len(gc), gold_chars_masked=len(gc&pc), pred_chars=len(pc),
                      excess_chars=len(pc-gc), nongold_chars=len(text)-len(gc),
                      negative_docs=int(not gold), negative_masked=int(not gold and bool(pred)))
        if gs != ps:
            errors.append(row["id"])
        for span in gold:
            if span["label"] == "NEGOTIATION_LIMIT":
                kind = "percent" if "%" in span["text"] else "money"
                positions = set(range(span["start"], span["end"]))
                full, any_hit = positions <= pc, bool(positions&pc)
                exact = (span["label"], span["start"], span["end"]) in ps
                subtypes[kind].update(gold=1, exact=int(exact), fully_masked=int(full), any_masked=int(any_hit),
                    unmasked=int(not any_hit), partially_masked=int(any_hit and not full),
                    fully_masked_wrong_type_or_boundary=int(full and not exact))
                subtype_families[kind].add(row.get("family", row["id"]))
        public, role, target = [], row.get("contrast_role"), row.get("contrast_target")
        target_hit = target_full = None
        if target is not None:
            a, b = target["start"], target["end"]
            if (role not in ("private", "public") or (role == "private") != target["should_mask"]
                    or not 0 <= a < b <= len(text) or text[a:b] != target["text"]):
                raise ValueError("Invalid contrast target")
            tc = set(range(a,b)); target_hit, target_full = bool(tc&pc), tc <= pc
            pair = pairs[row["contrast_id"]]
            if role in pair:
                raise ValueError("Duplicate contrast role")
            pair[role] = target_full if role == "private" else not target_hit
            if role == "public":
                public.append(("primary", target))
        public += [("repeat", s) for s in row.get("protected_spans", []) if s["label"] == "PUBLIC_REPEATED_VALUE"]
        for kind, span in public:
            a, b = span["start"], span["end"]
            if not 0 <= a < b <= len(text) or text[a:b] != span["text"]:
                raise ValueError("Invalid public source offset")
            positions = set(range(a,b)); hit = bool(positions&pc)
            totals.update(public_all=1, public_all_hit=int(hit), public_all_chars=len(positions),
                          public_all_chars_hit=len(positions&pc))
            prefix = "public_targets" if kind == "primary" else "public_repeated"
            totals[prefix] += 1; totals[prefix+"_hit"] += int(hit)
            if kind == "primary":
                totals.update(public_target_chars=len(positions), public_target_chars_hit=len(positions&pc))
            if hit:
                public_errors.append({"id": row["id"], "occurrence": kind, "text": span["text"], "start": a, "end": b})
        details.append({"id": row["id"], "tp": len(gs&ps), "fp": len(ps-gs), "fn": len(gs-ps),
                        "gold_entities_fully_masked": covered, "excess_characters": len(pc-gc),
                        "target_any_masked": target_hit, "target_fully_masked": target_full})
    total = {k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn")}
    complete_pairs = {key: p for key, p in pairs.items() if set(p) == {"private", "public"}}
    successful = sum(all(p.values()) for p in complete_pairs.values())
    return {"micro": prf(**total), "per_type": {label: prf(**{k: c[k] for k in ("tp","fp","fn")}) for label,c in counts.items()},
        "coverage": {"fully_masked_entities": totals["gold_fully_masked"], "gold_entities": totals["gold_entities"],
            "entity_rate": ratio(totals["gold_fully_masked"], totals["gold_entities"]),
            "character_rate": ratio(totals["gold_chars_masked"], totals["gold_chars"]),
            "masked_gold_characters": totals["gold_chars_masked"], "gold_characters": totals["gold_chars"],
            "not_fully_masked_entities": totals["gold_entities"]-totals["gold_fully_masked"]},
        "public_all": {"count": totals["public_all"], "masked": totals["public_all_hit"],
            "masked_rate": ratio(totals["public_all_hit"], totals["public_all"]),
            "primary_count": totals["public_targets"], "primary_masked": totals["public_targets_hit"],
            "repeated_count": totals["public_repeated"], "repeated_masked": totals["public_repeated_hit"]},
        "public_targets": {"count": totals["public_targets"], "masked": totals["public_targets_hit"],
            "masked_rate": ratio(totals["public_targets_hit"], totals["public_targets"]),
            "characters": totals["public_target_chars"], "masked_characters": totals["public_target_chars_hit"]},
        "overmasking": {"excess_characters": totals["excess_chars"], "predicted_characters": totals["pred_chars"],
            "excess_share_of_mask": ratio(totals["excess_chars"], totals["pred_chars"]),
            "nongold_character_rate": ratio(totals["excess_chars"], totals["nongold_chars"]),
            "negative_documents_masked": totals["negative_masked"], "negative_documents": totals["negative_docs"],
            "negative_document_rate": ratio(totals["negative_masked"], totals["negative_docs"])},
        "contrast_pairs": {"count": len(complete_pairs), "incomplete": len(pairs)-len(complete_pairs),
            "success": successful, "success_rate": ratio(successful, len(complete_pairs))},
        "negotiation_subtypes": {k: {**dict(v), "families": len(subtype_families[k]),
            "exact_recall": ratio(v["exact"],v["gold"]), "full_mask_rate": ratio(v["fully_masked"],v["gold"])} for k,v in subtypes.items()},
        "documents": len(rows), "families": len({r.get("family", r["id"]) for r in rows}),
        "totals": dict(totals), "error_document_ids": errors, "public_error_occurrences": public_errors,
        "document_details": details}


def selection_key(candidate):
    metrics = candidate["metrics"]
    return (metrics["micro"]["f1"] or 0., -metrics["public_all"]["masked"],
            -candidate.get("epoch", 0), candidate["threshold"])


def select_threshold(rows, cached, thresholds=THRESHOLDS):
    """Sweep an already computed dev cache; this function makes no model calls.

    Cache entries are {encoded: encode_row({id,text}, ...), probabilities: [L,13]}.
    Never call this on test. Tie break: fewer public masks, higher threshold.
    """
    if any(r.get("split") != "dev" for r in rows):
        raise ValueError("Threshold selection is restricted to the dev split")
    thresholds = tuple(thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("Threshold grid must be nonempty and unique")
    results = []
    for threshold in thresholds:
        predictions = [{"id": item["encoded"]["metadata"]["id"], "text": item["encoded"]["metadata"]["text"],
                        "predictions": decode_probabilities(item["encoded"], item["probabilities"], threshold)} for item in cached]
        results.append({"threshold": threshold, "metrics": evaluate(rows, predictions)})
    return {"selected": max(results, key=selection_key), "candidates": results,
            "criterion": "max strict micro-F1; fewer masked public occurrences; higher threshold",
            "decoder": DECODER, "model_forward_calls": 0}
