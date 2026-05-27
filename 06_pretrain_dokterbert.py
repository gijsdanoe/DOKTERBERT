"""
DOKTERBERT Contrastive Pretraining Script

Loss:
    L_total = L_mlm + alpha * (L_contrastive + L_orthogonal)

Routing:
    Medical spans  -> L_contrastive (graph-aware InfoNCE)
    Non-medical spans -> L_orthogonal (projection loss)

Usage:
    python dokterbert_train.py
"""

import json
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CORPUS_PATH       = r"path/to/corpus"        # raw text lines for MLM {"text": "..."}
SPANS_PATH        = "./data/spans.jsonl"         # span annotations
SENTENCES_PATH    = "./data/sentences.jsonl"     # sentence lookup by sentence_id
SNOMED_TSV        = "./data/snomed_nl.tsv"       # concept_id, description, semantic_type
DISTANCES_FILE    = "./data/snomed_distances.json"  # precomputed graph distances
MULTICLIN_TRAIN   = r"./data/multiclin/train.json"
MODEL_NAME        = r"path/to/medroberta"
SAVE_PATH         = "dokterbert"
DEVICE            = "cuda"

TAU               = 0.07
ALPHA             = 0.2
BETA_1            = 2.0
SIGMA             = 15.0
K_PER_TYPE        = 50
N_COMPONENTS      = 150
BATCH_SIZE        = 128
MED_SPAN_BATCH    = 32      # medical spans per step -> contrastive
NONMED_SPAN_BATCH = 0       # non-medical spans per step -> orthogonal
LEARNING_RATE     = 2e-5
WARMUP_STEPS      = 1000
EPOCHS            = 1
MLM_PROB          = 0.15
MAX_SEQ_LEN       = 128


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_sentences(path):
    """Load sentence id -> text into memory dict."""
    sentences = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sentences[obj["id"]] = obj["sentence"]
    print(f"  Loaded {len(sentences)} sentences")
    return sentences


def load_spans(path):
    medical, nonmedical = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("is_medical") and s.get("concept_id"):
                medical.append(s)
            elif not s.get("is_medical"):
                nonmedical.append(s)
    print(f"  Spans — medical: {len(medical)}, non-medical: {len(nonmedical)}")
    return medical, nonmedical


def load_snomed(path):
    concepts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                concepts[parts[0]] = {
                    "description": parts[1],
                    "semantic_type": parts[2]
                }
    return concepts


def load_distances(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# ANCHOR EMBEDDINGS (fixed)
# ─────────────────────────────────────────────

def initialize_anchors(snomed_concepts, model, tokenizer, device, spans_concept_ids):
    relevant_ids = [cid for cid in snomed_concepts if cid in spans_concept_ids]
    if not relevant_ids:
        raise ValueError("No matching concept IDs between spans and SNOMED TSV.")

    descriptions = [snomed_concepts[cid]["description"] for cid in relevant_ids]
    print(f"  Initializing {len(relevant_ids)} anchor embeddings (filtered from {len(snomed_concepts)} total)...")

    model.eval()
    all_embs = []
    for i in range(0, len(descriptions), 64):
        batch = descriptions[i:i+64]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        emb = out.hidden_states[-1][:, 0, :].cpu()
        all_embs.append(emb)

    init_matrix = torch.cat(all_embs, dim=0)
    anchor_embeddings = nn.Embedding.from_pretrained(init_matrix, freeze=True)
    concept_id_to_idx = {cid: i for i, cid in enumerate(relevant_ids)}
    return anchor_embeddings, concept_id_to_idx


# ─────────────────────────────────────────────
# MEDICAL SUBSPACE U
# ─────────────────────────────────────────────

def build_U(snomed_concepts, model, tokenizer, device):
    from collections import defaultdict
    type_embs = defaultdict(list)
    descriptions_by_type = defaultdict(list)

    for cid, info in snomed_concepts.items():
        descriptions_by_type[info["semantic_type"]].append(info["description"])

    print("  Encoding SNOMED by semantic type...")
    model.eval()
    for sem_type, descs in descriptions_by_type.items():
        if len(descs) > 5000:
            descs = random.sample(descs, 5000)
        for i in range(0, len(descs), 64):
            batch = descs[i:i+64]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=64).to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            emb = out.hidden_states[-1][:, 0, :].cpu().numpy()
            type_embs[sem_type].append(emb)

    all_centroids = []
    for sem_type, emb_list in type_embs.items():
        embs = np.vstack(emb_list)
        k = min(K_PER_TYPE, len(embs))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(embs)
        all_centroids.append(km.cluster_centers_)
        print(f"    {sem_type}: {len(embs)} concepts, k={k}")

    centroids = np.vstack(all_centroids)
    svd = TruncatedSVD(n_components=min(N_COMPONENTS, centroids.shape[0]-1), random_state=42)
    svd.fit(centroids)
    U = torch.tensor(svd.components_.T, dtype=torch.float32).to(device)
    return U


def build_U_from_multiclin(multiclin_path, model, tokenizer, device):
    """
    Build medical subspace U from actual labeled token embeddings in MultiClinNER training set.
    Uses tokens labeled as DISEASE, SYMPTOM, PROCEDURE in clinical context.
    More task-aligned than SNOMED description embeddings.
    """
    from collections import defaultdict

    print("  Building U from MultiClinNER training tokens...")
    with open(multiclin_path, encoding="utf-8") as f:
        records = json.load(f)

    type_token_embs = defaultdict(list)
    model.eval()

    for record in records:
        text = record["text"]
        entities = record.get("entities", [])
        if not entities:
            continue

        encoding = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN,
            return_offsets_mapping=True
        )
        offset_mapping = encoding["offset_mapping"][0]
        encoding.pop("offset_mapping")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]  # (seq_len, 768)

        for entity in entities:
            char_start = entity["start"]
            char_end = entity["end"]
            label = entity["label"]

            span_token_ids = [
                idx for idx, (start, end) in enumerate(offset_mapping)
                if end > char_start and start < char_end
            ]
            if not span_token_ids:
                continue

            # Collect individual token embeddings per entity type (cap at 5000)
            for token_id in span_token_ids:
                if len(type_token_embs[label]) < 5000:
                    type_token_embs[label].append(hidden[token_id].cpu().numpy())

    # KMeans per entity type then SVD
    all_centroids = []
    max_per_type = 5000
    for label, embs in type_token_embs.items():
        embs = np.vstack(embs)
        if len(embs) > max_per_type:
            idx = np.random.choice(len(embs), max_per_type, replace=False)
            embs = embs[idx]
        k = min(K_PER_TYPE, len(embs))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(embs)
        all_centroids.append(km.cluster_centers_)
        print(f"    {label}: {len(embs)} token embs, k={k}")

    centroids = np.vstack(all_centroids)
    svd = TruncatedSVD(n_components=min(N_COMPONENTS, centroids.shape[0]-1), random_state=42)
    svd.fit(centroids)
    U = torch.tensor(svd.components_.T, dtype=torch.float32).to(device)
    print(f"  U shape: {U.shape}")
    return U


# ─────────────────────────────────────────────
# WEIGHT FUNCTION
# ─────────────────────────────────────────────

def get_weight(c1, c2, distances, semantic_types):
    d = distances.get(c1, {}).get(c2, None)
    return math.exp(-d / SIGMA) if d is not None else 1.0


# ─────────────────────────────────────────────
# SPAN EMBEDDING
# ─────────────────────────────────────────────

def get_span_embeddings_batched(model, tokenizer, spans_with_sentences, device):
    """
    spans_with_sentences: list of (sentence, char_start, char_end)
    One forward pass per unique sentence.
    Returns mean-pooled span embeddings (for contrastive loss).
    """
    sentence_to_spans = {}
    for i, (sentence, char_start, char_end) in enumerate(spans_with_sentences):
        if sentence not in sentence_to_spans:
            sentence_to_spans[sentence] = []
        sentence_to_spans[sentence].append((i, char_start, char_end))

    span_embs = [None] * len(spans_with_sentences)

    for sentence, span_list in sentence_to_spans.items():
        encoding = tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN,
            return_offsets_mapping=True
        )
        offset_mapping = encoding["offset_mapping"][0]
        encoding.pop("offset_mapping")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        outputs = model(**encoding, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]

        for original_idx, char_start, char_end in span_list:
            span_token_ids = [
                idx for idx, (start, end) in enumerate(offset_mapping)
                if end > char_start and start < char_end
            ]
            if not span_token_ids:
                span_token_ids = [0]
            span_embs[original_idx] = hidden[span_token_ids].mean(dim=0)

    return span_embs


def get_token_embeddings_batched(model, tokenizer, spans_with_sentences, device):
    """
    Returns individual token embeddings for each span (for orthogonal loss).
    One forward pass per unique sentence.
    Returns list of tensors (n_tokens, 768) per span.
    """
    sentence_to_spans = {}
    for i, (sentence, char_start, char_end) in enumerate(spans_with_sentences):
        if sentence not in sentence_to_spans:
            sentence_to_spans[sentence] = []
        sentence_to_spans[sentence].append((i, char_start, char_end))

    token_embs = [None] * len(spans_with_sentences)

    for sentence, span_list in sentence_to_spans.items():
        encoding = tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN,
            return_offsets_mapping=True
        )
        offset_mapping = encoding["offset_mapping"][0]
        encoding.pop("offset_mapping")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        outputs = model(**encoding, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]

        for original_idx, char_start, char_end in span_list:
            span_token_ids = [
                idx for idx, (start, end) in enumerate(offset_mapping)
                if end > char_start and start < char_end
            ]
            if not span_token_ids:
                span_token_ids = [0]
            token_embs[original_idx] = hidden[span_token_ids]  # (n_tokens, 768)

    return token_embs


# ─────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────

def contrastive_loss(span_emb, anchor_embeddings, concept_id,
                     batch_concept_ids, distances, semantic_types,
                     concept_id_to_idx):
    if concept_id not in concept_id_to_idx:
        return None

    pos_idx = concept_id_to_idx[concept_id]
    pos_anchor = anchor_embeddings(torch.tensor(pos_idx).to(span_emb.device))
    pos_sim = F.cosine_similarity(span_emb.unsqueeze(0), pos_anchor.unsqueeze(0)) / TAU

    denom = torch.exp(pos_sim)
    for neg_concept_id in batch_concept_ids:
        if neg_concept_id == concept_id:
            continue
        if neg_concept_id not in concept_id_to_idx:
            continue
        neg_idx = concept_id_to_idx[neg_concept_id]
        neg_anchor = anchor_embeddings(torch.tensor(neg_idx).to(span_emb.device))
        neg_sim = F.cosine_similarity(span_emb.unsqueeze(0), neg_anchor.unsqueeze(0)) / TAU
        w = get_weight(concept_id, neg_concept_id, distances, semantic_types)
        denom = denom + w * torch.exp(neg_sim)

    return -pos_sim + torch.log(denom)


def orthogonal_loss(span_emb, U):
    projection = U.T @ span_emb
    return (projection ** 2).mean()


# ─────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────

class MLMDataset(Dataset):
    def __init__(self, corpus_dir, tokenizer):
        self.tokenizer = tokenizer
        self.texts = []
        for filename in sorted(os.listdir(corpus_dir)):
            if filename.endswith(".txt"):
                with open(os.path.join(corpus_dir, filename), encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.texts.append(line)
        print(f"  Loaded {len(self.texts)} lines from {corpus_dir}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt"
        )


class MedSpanDataset(Dataset):
    def __init__(self, spans):
        self.spans = spans

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, idx):
        return self.spans[idx]


class NonMedSpanDataset(Dataset):
    def __init__(self, spans):
        self.spans = spans

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, idx):
        return self.spans[idx]


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train():
    print("Loading data...")
    sentences = load_sentences(SENTENCES_PATH)
    medical_spans, _ = load_spans(SPANS_PATH)
    snomed_concepts = load_snomed(SNOMED_TSV)
    distances = load_distances(DISTANCES_FILE)
    semantic_types = {cid: info["semantic_type"] for cid, info in snomed_concepts.items()}

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)

    print("Initializing anchors...")
    spans_concept_ids = set(str(s["concept_id"]) for s in medical_spans if s.get("concept_id"))
    print(f"  Unique concept IDs in spans: {len(spans_concept_ids)}")
    anchor_embeddings, concept_id_to_idx = initialize_anchors(
        snomed_concepts, model, tokenizer, DEVICE, spans_concept_ids
    )
    anchor_embeddings = anchor_embeddings.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    mlm_dataset         = MLMDataset(CORPUS_PATH, tokenizer)
    med_span_dataset    = MedSpanDataset(medical_spans)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=MLM_PROB)

    mlm_loader = DataLoader(mlm_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=lambda x: data_collator(
                               [{k: v.squeeze() for k, v in item.items()} for item in x]
                           ))
    med_span_loader    = DataLoader(med_span_dataset,    batch_size=MED_SPAN_BATCH,
                                   shuffle=True, collate_fn=lambda x: x)

    total_steps = len(mlm_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    # Debug: verify sentence lookup works
    if medical_spans:
        test = medical_spans[0]
        test_sent = sentences.get(test["sentence_id"], "")
        print(f"  Sentence lookup test — id: {test['sentence_id']}, sentence: '{test_sent[:80]}'")

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        med_span_iter    = iter(med_span_loader)
        total_loss = total_mlm = total_cont = 0

        for step, mlm_batch in enumerate(mlm_loader):
            mlm_batch = {k: v.to(DEVICE) for k, v in mlm_batch.items()}

            # MLM loss
            mlm_out = model(**mlm_batch)
            mlm_loss = mlm_out.loss

            # Sample span batches
            try:
                med_batch = next(med_span_iter)
            except StopIteration:
                med_span_iter = iter(med_span_loader)
                med_batch = next(med_span_iter)

            cont_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            n_cont = 0

            # Collect negatives from medical batch
            batch_concept_ids = [str(s["concept_id"]) for s in med_batch if s.get("concept_id")]

            # Medical spans -> contrastive loss
            med_spans_data = []
            med_spans_meta = []
            for span in med_batch:
                sent = sentences.get(span["sentence_id"], "")
                if sent:
                    med_spans_data.append((sent, span["char_start"], span["char_end"]))
                    med_spans_meta.append(span)

            if med_spans_data:
                med_embs = get_span_embeddings_batched(model, tokenizer, med_spans_data, DEVICE)
                for span_emb, span in zip(med_embs, med_spans_meta):
                    if span_emb is None:
                        continue
                    loss = contrastive_loss(
                        span_emb, anchor_embeddings,
                        str(span["concept_id"]), batch_concept_ids,
                        distances, semantic_types, concept_id_to_idx
                    )
                    if loss is not None:
                        cont_loss = cont_loss + loss
                        n_cont += 1

            if n_cont > 0:
                cont_loss = cont_loss / n_cont

            loss = mlm_loss + ALPHA * cont_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_mlm  += mlm_loss.item()
            total_cont += cont_loss.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch+1} Step {step}/{len(mlm_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"MLM: {mlm_loss.item():.4f} | "
                      f"Cont: {cont_loss.item():.4f}")

        avg = total_loss / len(mlm_loader)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg:.4f}")
        model.save_pretrained(f"{SAVE_PATH}_epoch{epoch+1}")
        tokenizer.save_pretrained(f"{SAVE_PATH}_epoch{epoch+1}")
        print(f"  Saved to {SAVE_PATH}_epoch{epoch+1}")

    print("Training complete.")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    train()
