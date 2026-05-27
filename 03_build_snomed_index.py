"""
Build SNOMED FAISS index from data/snomed_nl.tsv.
Run once — saves index and metadata to disk for reuse.

Input:
    data/snomed_nl.tsv — concept_id \\t term \\t semantic_type
                 semantic_type: verrichting | bevinding | aandoening

Output:
    snomed.index     — FAISS index (IndexFlatIP, L2-normalized = cosine)
    snomed_meta.json — list of entity_type per FAISS index position

Usage:
    python 03_build_snomed_index.py
"""

import json
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

# =======================
# CONFIG
# =======================
MODEL_DIR    = r"./dokterbert"
#MODEL_DIR = r"path/to/medroberta"
SNOMED_TSV   = r"./data/snomed_nl.tsv"
OUTPUT_INDEX = r"./snomed.index"
OUTPUT_META  = r"./snomed_meta.json"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 64

SNOMED_TYPE_MAP = {
    "procedure": "PROCEDURE",
    "finding":   "SYMPTOM",
    "disorder":  "DISEASE",
}


def main():
    print(f"Loading model: {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model     = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    # Parse TSV
    print(f"Parsing {SNOMED_TSV}...")
    terms     = []  # (term, entity_type)
    skipped   = 0

    with open(SNOMED_TSV, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            _, term, sem_type = parts[0], parts[1], parts[2].lower().strip()
            entity_type = SNOMED_TYPE_MAP.get(sem_type)
            if entity_type is None:
                skipped += 1
                continue
            terms.append((term, entity_type))

    print(f"  Loaded {len(terms)} terms ({skipped} skipped — unknown semantic type)")

    # Encode all terms
    print(f"Encoding {len(terms)} terms with DokterBERT...")
    all_embs = []

    for i in range(0, len(terms), BATCH_SIZE):
        batch_terms = [t for t, _ in terms[i:i + BATCH_SIZE]]
        encoding = tokenizer(
            batch_terms, padding=True, truncation=True,
            max_length=64, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs   = model(**encoding, output_hidden_states=True)
            hidden    = outputs.hidden_states[-1]
            attn_mask = encoding["attention_mask"].unsqueeze(-1).float()
            embs      = (hidden * attn_mask).sum(1) / attn_mask.sum(1)
            embs      = embs.cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        all_embs.append(embs / norms)

        if (i // BATCH_SIZE) % 100 == 0:
            print(f"  {i}/{len(terms)}...")

    all_embs = np.vstack(all_embs)
    print(f"Encoded {len(all_embs)} terms → shape {all_embs.shape}")

    # Build and save FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(all_embs.shape[1])  # cosine on normalized vectors
    index.add(all_embs)
    faiss.write_index(index, OUTPUT_INDEX)
    print(f"Saved FAISS index → {OUTPUT_INDEX} ({index.ntotal} vectors)")

    # Save metadata (entity_type per index position)
    id2type = [et for _, et in terms]
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(id2type, f)
    print(f"Saved metadata → {OUTPUT_META}")

    print("\nDone. Index ready for 05_link_spans.py and 08_geometry_checks.py.")


if __name__ == "__main__":
    main()
