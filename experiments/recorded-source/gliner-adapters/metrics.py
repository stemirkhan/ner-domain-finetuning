"""Exact NER and character masking metrics, independent of model libraries."""
from collections import Counter, defaultdict
import math

DECODER = {
    "version": "crm-global-greedy-v1",
    "candidate_threshold": 0.0,
    "model_overlap_policy": "allow",
    "adaptive_threshold": False,
    "threshold_rule": "score >= selected threshold, before overlap resolution",
    "overlap_rule": "global greedy: descending score, descending length, ascending start/end/label",
    "abstention": "checkpoint default; retained and recorded in model configuration",
}


def validate_spans(text, spans, labels, scored=False):
    for span in spans:
        a, b = span["start"], span["end"]
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text):
            raise ValueError(f"Invalid span boundaries: {span}")
        if span["label"] not in labels or text[a:b] != span["text"]:
            raise ValueError(f"Invalid span label/surface: {span}")
        if scored and (not math.isfinite(float(span["score"])) or not 0 <= span["score"] <= 1):
            raise ValueError(f"Invalid confidence: {span}")


def decode(candidates, threshold):
    """One deterministic non-overlapping policy across all six labels."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    ranked = sorted((s for s in candidates if float(s["score"]) >= threshold),
                    key=lambda s: (-float(s["score"]), -(s["end"]-s["start"]), s["start"], s["end"], s["label"]))
    kept = []
    for span in ranked:
        if not any(span["start"] < p["end"] and p["start"] < span["end"] for p in kept):
            kept.append(dict(span))
    return sorted(kept, key=lambda s: (s["start"], s["end"], s["label"]))


def chars(spans):
    return {i for s in spans for i in range(s["start"], s["end"])}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
            "f1": ratio(2*tp, 2*tp+fp+fn)}


def mask_text(text, predictions):
    result = text
    last = len(text)
    for span in sorted(predictions, key=lambda s: s["start"], reverse=True):
        if span["end"] > last:
            raise ValueError("Mask requires non-overlapping predictions")
        result = result[:span["start"]] + "<" + span["label"] + ">" + result[span["end"]:]
        last = span["start"]
    return result


def evaluate(rows, predictions, labels):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate gold row ID")
    if len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction row ID")
    by_id = {p["id"]: p for p in predictions}
    if set(by_id) != {r["id"] for r in rows}:
        raise ValueError("Prediction and gold IDs must match exactly")
    counts = {label: Counter() for label in labels}
    totals = Counter()
    pairs = defaultdict(dict)
    document_details = []
    for row in rows:
        pred_row = by_id[row["id"]]
        if "text" in pred_row and pred_row["text"] != row["text"]:
            raise ValueError("Prediction text changed")
        gold, pred = row["spans"], pred_row["predictions"]
        validate_spans(row["text"], gold, labels)
        validate_spans(row["text"], pred, labels, scored=True)
        gold_set = {(s["label"], s["start"], s["end"]) for s in gold}
        pred_set = {(s["label"], s["start"], s["end"]) for s in pred}
        for label in labels:
            gs = {s for s in gold_set if s[0] == label}; ps = {s for s in pred_set if s[0] == label}
            counts[label].update(tp=len(gs & ps), fp=len(ps-gs), fn=len(gs-ps))
        gc, pc = chars(gold), chars(pred)
        covered = sum(set(range(s["start"], s["end"])) <= pc for s in gold)
        totals.update(documents=1, text_characters=len(row["text"]), gold_entities=len(gold),
                      gold_entities_fully_masked=covered, gold_characters=len(gc),
                      gold_characters_masked=len(gc & pc), predicted_characters=len(pc),
                      excess_characters=len(pc-gc), nongold_characters=len(row["text"])-len(gc))
        if not gold:
            totals["negative_documents"] += 1
            totals["negative_documents_masked"] += bool(pred)
        target = row["contrast_target"]
        tc = set(range(target["start"], target["end"]))
        target_masked = bool(tc & pc)
        target_fully_masked = tc <= pc
        role = row["contrast_role"]
        if role not in ("private", "public") or (role == "private") != target["should_mask"]:
            raise ValueError("Invalid contrast role")
        if role in pairs[row["contrast_id"]]:
            raise ValueError("Duplicate contrast role")
        pairs[row["contrast_id"]][role] = target_fully_masked if role == "private" else not target_masked
        if role == "public":
            totals.update(public_targets=1, public_targets_masked=int(target_masked),
                          public_target_characters=len(tc), public_target_characters_masked=len(tc & pc))
        protected = row.get("protected_spans", [])
        totals.update(protected_mentions=len(protected),
                      protected_mentions_masked=sum(bool(set(range(s["start"], s["end"])) & pc) for s in protected))
        document_details.append({"id": row["id"], "family": row["family"],
                                 "tp": len(gold_set & pred_set), "fp": len(pred_set-gold_set), "fn": len(gold_set-pred_set),
                                 "gold_entities_fully_masked": covered, "excess_characters": len(pc-gc),
                                 "contrast_role": role, "target_any_masked": target_masked,
                                 "target_fully_masked": target_fully_masked})
    if any(set(pair) != {"private", "public"} for pair in pairs.values()):
        raise ValueError("Evaluation requires complete contrast pairs")
    per_type = {label: prf(**{k: counts[label][k] for k in ("tp", "fp", "fn")}) for label in labels}
    total = {k: sum(counts[label][k] for label in labels) for k in ("tp", "fp", "fn")}
    return {
        "micro": prf(**total), "per_type": per_type,
        "coverage": {"fully_masked_entities": totals["gold_entities_fully_masked"], "gold_entities": totals["gold_entities"],
                     "entity_rate": ratio(totals["gold_entities_fully_masked"], totals["gold_entities"]),
                     "masked_gold_characters": totals["gold_characters_masked"], "gold_characters": totals["gold_characters"],
                     "character_rate": ratio(totals["gold_characters_masked"], totals["gold_characters"])},
        "overmasking": {"excess_characters": totals["excess_characters"], "predicted_characters": totals["predicted_characters"],
                        "excess_share_of_mask": ratio(totals["excess_characters"], totals["predicted_characters"]),
                        "nongold_character_rate": ratio(totals["excess_characters"], totals["nongold_characters"]),
                        "negative_documents_masked": totals["negative_documents_masked"], "negative_documents": totals["negative_documents"],
                        "negative_document_rate": ratio(totals["negative_documents_masked"], totals["negative_documents"])},
        "public_targets": {"count": totals["public_targets"], "masked": totals["public_targets_masked"],
                           "masked_rate": ratio(totals["public_targets_masked"], totals["public_targets"]),
                           "characters": totals["public_target_characters"], "masked_characters": totals["public_target_characters_masked"],
                           "masked_character_rate": ratio(totals["public_target_characters_masked"], totals["public_target_characters"])},
        "contrast_pairs": {"count": len(pairs), "success": sum(all(pair.values()) for pair in pairs.values()),
                           "success_rate": ratio(sum(all(pair.values()) for pair in pairs.values()), len(pairs))},
        "protected_all": {"count": totals["protected_mentions"], "masked": totals["protected_mentions_masked"],
                          "note": "Includes SKU/quantities; use public_targets separately."},
        "documents": totals["documents"], "families": len({r["family"] for r in rows}),
        "document_details": document_details,
    }


def selection_key(candidate):
    m = candidate["metrics"]
    return (m["micro"]["f1"] or 0.0, -m["public_targets"]["masked"], -candidate["epoch"], candidate["threshold"])
