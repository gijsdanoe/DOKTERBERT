"""
Advanced Geometry Checks for DokterBERT
Evaluates representation quality beyond separability.

Checks:
  1. Nearest neighbor coherence — entity tokens near same-type tokens?
  2. Clustering purity — do clusters align with entity types?
  3. SNOMED concept recovery — are entity embeddings close to their SNOMED anchors?
  4. Cross/intra-type distances — inter-type far, intra-type close?
  5. Linear probe — can a linear classifier predict entity type from frozen embeddings?

Usage:
    python geometry_checks.py
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict
import faiss

# =======================
# CONFIG
# =======================
# MODEL_DIR = r"./dokterbert"   # example: evaluate DOKTERBERT
MODEL_DIR = r"path/to/model-to-evaluate"   # run once per model: dokterbert, medroberta, sapbert, robbert
TRAIN_JSON   = r"./data/multiclin/dev.json"
SNOMED_INDEX = r"./snomed.index"        # for SNOMED concept recovery check
SNOMED_META  = r"./snomed_meta.json"
SNOMED_TSV   = r"./data/snomed_nl.tsv"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH   = 512
MAX_TOKENS   = 5000   # max entity tokens to collect per entity type
K_NEIGHBORS  = 10     # for NN coherence
MIN_CONCEPT_TOKENS = 3  # min tokens per concept to include in concept-level checks
ENTITY_TYPES = ["DISEASE", "PROCEDURE", "SYMPTOM"]

SNOMED_TYPE_MAP = {
    "procedure": "PROCEDURE",
    "finding":   "SYMPTOM",
    "disorder":  "DISEASE",
}


# =======================
# HELPERS
# =======================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_real_tokens(special_mask):
    return [i for i, sm in enumerate(special_mask) if sm == 0]


# =======================
# BUILD SNOMED LOOKUP
# =======================
def build_snomed_lookup(tsv_path):
    """
    Build: term_lower -> (concept_id, entity_type)
    and:   concept_id -> (preferred_term, entity_type)
    """
    term2concept = {}
    concept2term = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            concept_id, term, sem_type = parts[0], parts[1], parts[2].lower().strip()
            et = SNOMED_TYPE_MAP.get(sem_type)
            if not et or not term.strip():
                continue
            term2concept[term.lower().strip()] = (concept_id, et)
            if concept_id not in concept2term:
                concept2term[concept_id] = (term, et)
    print(f"  SNOMED lookup: {len(term2concept)} terms, {len(concept2term)} concepts")
    return term2concept, concept2term


# =======================
# COLLECT ENTITY TOKEN EMBEDDINGS WITH SNOMED CONCEPT IDs
# =======================
def collect_entity_embeddings(records, model, tokenizer, device,
                               term2concept, max_per_type=MAX_TOKENS):
    """
    Collect contextual token embeddings for entity tokens.
    Link each entity span to its SNOMED concept ID via exact match.

    Returns:
        embs:        list of (768,) embeddings
        type_labels: list of entity type strings
        concept_ids: list of SNOMED concept ID strings (None if not matched)
    """
    embs        = []
    type_labels = []
    concept_ids = []
    type_counts = defaultdict(int)
    model.eval()

    for record in records:
        text     = record["text"]
        entities = record.get("entities", [])
        if not entities:
            continue

        if all(type_counts[et] >= max_per_type for et in ENTITY_TYPES):
            break

        lines    = text.split('\n')
        char_pos = 0

        for line in lines:
            if not line.strip():
                char_pos += len(line) + 1
                continue

            enc = tokenizer(
                line, truncation=True, max_length=MAX_LENGTH,
                return_offsets_mapping=True, return_special_tokens_mask=True,
                return_tensors="pt"
            )
            offsets      = enc.pop("offset_mapping")[0].tolist()
            special_mask = enc.pop("special_tokens_mask")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1][0].cpu().numpy()

            shifted = [(s + char_pos, e + char_pos) for s, e in offsets]

            for ent in entities:
                et = ent["label"]
                if type_counts[et] >= max_per_type:
                    continue
                cs, ce = ent["start"], ent["end"]
                touched = [i for i, (ts, te) in enumerate(shifted)
                           if te > cs and ts < ce and special_mask[i] == 0]
                if not touched:
                    continue

                # Mean pool span tokens
                span_emb = hidden[touched].mean(axis=0)
                span_text = text[cs:ce].lower().strip()

                # Link to SNOMED concept
                concept_id = None
                if span_text in term2concept:
                    cid, cet = term2concept[span_text]
                    if cet == et:
                        concept_id = cid

                embs.append(span_emb)
                type_labels.append(et)
                concept_ids.append(concept_id)
                type_counts[et] += 1

            char_pos += len(line) + 1

    print(f"  Collected {len(embs)} entity span embeddings")
    matched = sum(1 for c in concept_ids if c is not None)
    print(f"  SNOMED matched: {matched}/{len(embs)} ({100*matched/len(embs):.1f}%)")
    for et in ENTITY_TYPES:
        print(f"    {et}: {type_counts[et]} spans")

    return np.vstack(embs), type_labels, concept_ids


# =======================
# CHECK 1: NN COHERENCE AT CONCEPT LEVEL
# =======================
def check_nn_coherence(embs, type_labels, concept_ids, k=K_NEIGHBORS):
    """
    For each entity span with a known SNOMED concept, find k nearest neighbors.
    Measure:
      - same-type fraction (coarse)
      - same-concept fraction (fine-grained) — the key DokterBERT check
    """
    print("\n" + "="*60)
    print("CHECK 1 — Nearest Neighbor Coherence (concept level)")
    print("="*60)

    embs_norm   = normalize(embs)
    type_arr    = np.array(type_labels)
    concept_arr = np.array([c if c else "NONE" for c in concept_ids])

    index = faiss.IndexFlatIP(embs_norm.shape[1])
    index.add(embs_norm.astype(np.float32))
    distances, indices = index.search(embs_norm.astype(np.float32), k + 1)

    # Only evaluate spans with known concept
    has_concept = np.array([c is not None for c in concept_ids])

    type_coherences    = []
    concept_coherences = []

    for i in np.where(has_concept)[0][:2000]:
        neighbors     = indices[i][1:k+1]
        neighbor_types    = type_arr[neighbors]
        neighbor_concepts = concept_arr[neighbors]

        type_coherences.append((neighbor_types == type_arr[i]).mean())
        concept_coherences.append((neighbor_concepts == concept_arr[i]).mean())

    print(f"  Same-type NN coherence:    {np.mean(type_coherences):.4f} (coarse)")
    print(f"  Same-concept NN coherence: {np.mean(concept_coherences):.4f} (fine-grained)")
    print(f"  Random baseline (type):    {1/3:.4f}")
    print(f"  Random baseline (concept): ~0.000")

    return np.mean(type_coherences), np.mean(concept_coherences)


# =======================
# CHECK 2: CLUSTERING AT CONCEPT LEVEL
# =======================
def check_clustering(embs, type_labels, concept_ids):
    """
    Cluster embeddings and measure:
      - NMI/ARI against entity type labels (coarse)
      - NMI/ARI against SNOMED concept IDs (fine-grained)
    Only use spans with known SNOMED concept for concept-level check.
    """
    print("\n" + "="*60)
    print("CHECK 2 — Clustering Purity (type and concept level)")
    print("="*60)

    embs_norm = normalize(embs)
    type_ints = np.array([ENTITY_TYPES.index(t) for t in type_labels])

    # Coarse: cluster by entity type
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    pred3 = km3.fit_predict(embs_norm)
    nmi_type = normalized_mutual_info_score(type_ints, pred3)
    ari_type = adjusted_rand_score(type_ints, pred3)
    print(f"  Entity type clustering — NMI: {nmi_type:.4f}  ARI: {ari_type:.4f}")

    # Fine-grained: cluster by SNOMED concept
    has_concept = np.array([c is not None for c in concept_ids])
    concept_subset = [(i, c) for i, c in enumerate(concept_ids) if c is not None]

    if len(concept_subset) > 100:
        # Only concepts with enough tokens
        concept_counts = defaultdict(int)
        for _, c in concept_subset:
            concept_counts[c] += 1
        valid_concepts = {c for c, n in concept_counts.items() if n >= MIN_CONCEPT_TOKENS}

        valid_idx     = [i for i, c in concept_subset if c in valid_concepts]
        valid_concept_labels = [concept_ids[i] for i in valid_idx]

        # Encode concept IDs as integers
        unique_concepts = sorted(set(valid_concept_labels))
        concept2int     = {c: i for i, c in enumerate(unique_concepts)}
        concept_int_labels = np.array([concept2int[c] for c in valid_concept_labels])

        n_clusters = min(len(unique_concepts), 50)  # cap at 50 clusters
        km_c = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        pred_c = km_c.fit_predict(normalize(embs[valid_idx]))

        nmi_concept = normalized_mutual_info_score(concept_int_labels, pred_c)
        ari_concept = adjusted_rand_score(concept_int_labels, pred_c)
        print(f"  SNOMED concept clustering  — NMI: {nmi_concept:.4f}  ARI: {ari_concept:.4f}")
        print(f"  ({len(unique_concepts)} concepts, {len(valid_idx)} spans, k={n_clusters})")
    else:
        print(f"  Too few concept-matched spans for concept clustering")

    return nmi_type, ari_type


# =======================
# CHECK 3: SNOMED CONCEPT RECOVERY (SPECIFIC ANCHOR)
# =======================
def check_snomed_recovery(embs, concept_ids, concept2term, model, tokenizer, device):
    """
    For each entity span with known SNOMED concept:
      - Encode its specific SNOMED preferred term in isolation
      - Measure cosine similarity between contextual span embedding and isolated concept embedding
    This directly measures whether contrastive pretraining achieved its objective.
    """
    print("\n" + "="*60)
    print("CHECK 3 — SNOMED Concept Recovery (specific anchor)")
    print("="*60)

    # Get unique concepts to encode
    unique_concepts = {c: concept2term[c] for c in concept_ids
                       if c is not None and c in concept2term}

    if not unique_concepts:
        print("  No concept-matched spans found")
        return None

    print(f"  Encoding {len(unique_concepts)} unique SNOMED concept terms...")

    # Encode all unique concept terms in isolation
    concept_embs = {}
    model.eval()
    terms   = list(unique_concepts.items())
    batch_size = 32

    for i in range(0, len(terms), batch_size):
        batch = terms[i:i+batch_size]
        batch_ids   = [cid for cid, _ in batch]
        batch_texts = [term for _, (term, _) in batch]

        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=64, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden    = out.hidden_states[-1]
        attn_mask = enc["attention_mask"].unsqueeze(-1).float()
        batch_embs = (hidden * attn_mask).sum(1) / attn_mask.sum(1)
        batch_embs = batch_embs.cpu().numpy()

        for j, cid in enumerate(batch_ids):
            concept_embs[cid] = batch_embs[j]

    # For each span, measure similarity to its specific concept embedding
    sims_matched = []
    sims_random  = []

    for i, (emb, cid) in enumerate(zip(embs, concept_ids)):
        if cid is None or cid not in concept_embs:
            continue

        emb_norm     = emb / (np.linalg.norm(emb) + 1e-9)
        concept_emb  = concept_embs[cid]
        concept_norm = concept_emb / (np.linalg.norm(concept_emb) + 1e-9)

        # Similarity to correct concept
        sim = float(emb_norm @ concept_norm)
        sims_matched.append(sim)

        # Similarity to random other concept (same type)
        other_concepts = [c for c in concept_embs if c != cid]
        if other_concepts:
            rand_cid  = other_concepts[np.random.randint(len(other_concepts))]
            rand_emb  = concept_embs[rand_cid] / (np.linalg.norm(concept_embs[rand_cid]) + 1e-9)
            rand_sim  = float(emb_norm @ rand_emb)
            sims_random.append(rand_sim)

    if sims_matched:
        print(f"  Avg similarity to correct SNOMED concept: {np.mean(sims_matched):.4f}")
        print(f"  Avg similarity to random SNOMED concept:  {np.mean(sims_random):.4f}")
        print(f"  Discrimination gap: {np.mean(sims_matched) - np.mean(sims_random):+.4f}")
        print(f"  (evaluated on {len(sims_matched)} concept-matched spans)")

    return np.mean(sims_matched) if sims_matched else None


# =======================
# CHECK 4: INTRA/INTER-CONCEPT DISTANCES
# =======================
def check_concept_distances(embs, concept_ids):
    """
    Measure intra-concept vs inter-concept cosine similarity.
    DokterBERT should have high intra-concept similarity (same concept close)
    and low inter-concept similarity (different concepts far apart).
    """
    print("\n" + "="*60)
    print("CHECK 4 — Intra/Inter-concept Distances")
    print("="*60)

    embs_norm = normalize(embs)

    # Group by concept
    concept_groups = defaultdict(list)
    for i, cid in enumerate(concept_ids):
        if cid is not None:
            concept_groups[cid].append(i)

    # Filter to concepts with enough tokens
    valid = {c: idxs for c, idxs in concept_groups.items()
             if len(idxs) >= MIN_CONCEPT_TOKENS}

    if len(valid) < 2:
        print("  Too few concepts with sufficient tokens")
        return

    print(f"  {len(valid)} concepts with >={MIN_CONCEPT_TOKENS} tokens")

    # Intra-concept similarity
    intra_sims = []
    for cid, idxs in list(valid.items())[:50]:
        e = embs_norm[idxs]
        sims = e @ e.T
        mask = ~np.eye(len(e), dtype=bool)
        if mask.any():
            intra_sims.append(sims[mask].mean())

    # Inter-concept similarity
    inter_sims = []
    concept_list = list(valid.keys())[:50]
    for i in range(len(concept_list)):
        for j in range(i+1, min(i+5, len(concept_list))):
            e1 = embs_norm[valid[concept_list[i]][:10]]
            e2 = embs_norm[valid[concept_list[j]][:10]]
            inter_sims.append((e1 @ e2.T).mean())

    print(f"  Intra-concept similarity: {np.mean(intra_sims):.4f} (higher = more compact)")
    print(f"  Inter-concept similarity: {np.mean(inter_sims):.4f} (lower = better separated)")
    print(f"  Discrimination ratio: {np.mean(intra_sims)/np.mean(inter_sims):.4f} (higher = better)")


# =======================
# CHECK 5: LINEAR PROBE
# =======================
def check_linear_probe(embs, type_labels):
    """
    Train a linear classifier on frozen embeddings to predict entity type.
    Higher accuracy = more entity-type information encoded geometrically.
    """
    print("\n" + "="*60)
    print("CHECK 5 — Linear Probe (entity type classification)")
    print("="*60)

    embs_norm  = normalize(embs)
    label_ints = np.array([ENTITY_TYPES.index(t) for t in type_labels])

    idx   = np.random.permutation(len(embs_norm))
    split = int(0.8 * len(idx))
    X_train, X_test = embs_norm[idx[:split]], embs_norm[idx[split:]]
    y_train, y_test = label_ints[idx[:split]], label_ints[idx[split:]]

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"  Linear probe accuracy:   {acc:.4f}")
    print(f"  Random baseline:         {1/3:.4f}")
    print(f"  Improvement over random: +{acc - 1/3:.4f}")
    return acc


# =======================
# MAIN
# =======================
def main():
    print(f"Model: {MODEL_DIR}")
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model     = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    print(f"\nBuilding SNOMED lookup from {SNOMED_TSV}...")
    term2concept, concept2term = build_snomed_lookup(SNOMED_TSV)

    print(f"\nCollecting entity span embeddings from {TRAIN_JSON}...")
    train_records = load_json(TRAIN_JSON)
    embs, type_labels, concept_ids = collect_entity_embeddings(
        train_records, model, tokenizer, DEVICE, term2concept
    )

    # Run all checks
    check_nn_coherence(embs, type_labels, concept_ids)
    check_clustering(embs, type_labels, concept_ids)
    check_snomed_recovery(embs, concept_ids, concept2term, model, tokenizer, DEVICE)
    check_concept_distances(embs, concept_ids)
    check_linear_probe(embs, type_labels)

    print("\n" + "="*60)
    print("DONE — run on MedRoBERTa (CLTL/MedRoBERTa.nl) to compare")
    print("="*60)


if __name__ == "__main__":
    main()
