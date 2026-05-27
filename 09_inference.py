"""
MultiClinNER-nl Inference Script
Runs fine-tuned per-type models on test texts and produces run.tsv.

Expected directory structure:
    MultiClinNER-nl-test_batch1/
        MultiClinNER-nl-test-disease_batch1/txt/*.txt
        MultiClinNER-nl-test-procedure_batch1/txt/*.txt
        MultiClinNER-nl-test-symptom_batch1/txt/*.txt

Produces:
    run.tsv — single TSV with columns: filename, label, start_span, end_span, text

Usage:
    python 09_inference.py
"""

import os
import csv
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

# =======================
# CONFIG
# =======================
MAX_LENGTH  = 512

# Fine-tuned model directories — one per entity type
MODEL_DIRS = {
    "DISEASE":   r"./finetuned_model_ce_layerwise_lr/disease",
    "PROCEDURE": r"./finetuned_model_ce_layerwise_lr/procedure",
    "SYMPTOM":   r"./finetuned_model_ce_layerwise_lr/symptom",
}

# Test JSON files — preprocessed MultiClinNER-nl test data (see README)
TEST_JSONS = {
    "DISEASE":   r"./data/multiclin/test_disease.json",
    "PROCEDURE": r"./data/multiclin/test_procedure.json",
    "SYMPTOM":   r"./data/multiclin/test_symptom.json",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =======================
# HELPERS
# =======================
def get_real_tokens(special_mask):
    return [i for i, sm in enumerate(special_mask) if sm == 0]


def tokenize_lines(text, tokenizer):
    """
    Tokenize each line separately with char offset tracking.
    Returns list of (input_ids, attention_mask, shifted_offsets, special_mask).
    """
    lines    = text.split('\n')
    char_pos = 0
    chunks   = []

    for line in lines:
        if line.strip():
            enc = tokenizer(
                line, truncation=True, max_length=MAX_LENGTH,
                return_offsets_mapping=True, return_special_tokens_mask=True,
                return_tensors="pt"
            )
            offsets      = enc.pop("offset_mapping")[0].tolist()
            special_mask = enc.pop("special_tokens_mask")[0].tolist()
            shifted      = [(s + char_pos, e + char_pos) for s, e in offsets]
            chunks.append((enc, shifted, special_mask))
        char_pos += len(line) + 1

    return chunks


def bio_to_spans(token_labels, offsets, special_mask, text, entity_type):
    """
    Convert BIO token labels to character-level spans.
    Returns list of (char_start, char_end, span_text).
    """
    real_tokens = get_real_tokens(special_mask)
    spans       = []
    current_start = None
    current_end   = None

    for rel_idx, tok_idx in enumerate(real_tokens):
        label    = token_labels[rel_idx]
        ts, te   = offsets[tok_idx]

        if label == f"B-{entity_type}":
            # Start new span
            if current_start is not None:
                span_text = text[current_start:current_end].strip()
                if span_text:
                    spans.append((current_start, current_end, span_text))
            current_start = ts
            current_end   = te

        elif label == f"I-{entity_type}" and current_start is not None:
            current_end = te

        else:
            # O or different entity — close current span
            if current_start is not None:
                span_text = text[current_start:current_end].strip()
                if span_text:
                    spans.append((current_start, current_end, span_text))
                current_start = None
                current_end   = None

    # Close final span
    if current_start is not None:
        span_text = text[current_start:current_end].strip()
        if span_text:
            spans.append((current_start, current_end, span_text))

    return spans


# =======================
# INFERENCE FOR ONE TYPE
# =======================
def run_inference(entity_type, model_dir, json_path):
    """
    Run inference on all texts for one entity type.
    Returns list of dicts with filename, label, start_span, end_span, text.
    """
    print(f"\n{'='*60}")
    print(f"Entity type: {entity_type}")
    print(f"Model: {model_dir}")
    print(f"Input: {json_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model     = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    label_list = ["O", f"B-{entity_type}", f"I-{entity_type}"]
    id2label   = {i: l for i, l in enumerate(label_list)}

    # Load records from JSON
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  {len(records)} records loaded")

    rows = []

    for record in records:
        fname = record["filename"]
        text  = record["text"]

        # Tokenize line by line
        chunks = tokenize_lines(text, tokenizer)

        # Run inference on each chunk
        all_token_labels = []
        all_offsets      = []
        all_special      = []

        for enc, shifted, special_mask in chunks:
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
            logits   = outputs.logits[0].cpu().numpy()
            pred_ids = np.argmax(logits, axis=-1)

            real_tokens = get_real_tokens(special_mask)
            chunk_labels = [id2label[pred_ids[ti]] for ti in real_tokens]

            all_token_labels.extend(chunk_labels)
            all_offsets.extend(shifted)
            all_special.extend(special_mask)

        # Convert BIO to spans
        # Rebuild real_tokens for the full document
        real_tokens = get_real_tokens(all_special)
        assert len(all_token_labels) == len(real_tokens)

        spans = bio_to_spans(all_token_labels, all_offsets, all_special, text, entity_type)

        for char_start, char_end, span_text in spans:
            rows.append({
                "filename":   fname,
                "label":      entity_type,
                "start_span": char_start,
                "end_span":   char_end,
                "text":       span_text,
            })

    print(f"  {len(rows)} entities predicted")
    return rows


# =======================
# MAIN
# =======================
def main():
    all_rows = []

    for entity_type in ["DISEASE", "PROCEDURE", "SYMPTOM"]:
        model_dir = MODEL_DIRS[entity_type]
        json_path = TEST_JSONS[entity_type]

        if not os.path.exists(model_dir):
            print(f"WARNING: Model dir not found: {model_dir}")
            continue
        if not os.path.exists(json_path):
            print(f"WARNING: JSON not found: {json_path} — preprocess the MultiClinNER-nl test data first (see README)")
            continue

        rows = run_inference(entity_type, model_dir, json_path)
        all_rows.extend(rows)

    # Write single run.tsv
    out_path = "run.tsv"
    print(f"\nWriting {len(all_rows)} predictions to {out_path}...")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label", "start_span", "end_span", "text"],
            delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done. {out_path} written.")
    for et in ["DISEASE", "PROCEDURE", "SYMPTOM"]:
        n = sum(1 for r in all_rows if r["label"] == et)
        print(f"  {et}: {n}")


if __name__ == "__main__":
    main()
