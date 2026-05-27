"""
DOKTERBERT Per-Type Fine-tuning — UNIFIED SCRIPT
Configure which losses to use via the LOSS_MODE setting.

LOSS_MODE options:
    "ce"              — cross-entropy only (baseline)
    "orth"            — CE + orthogonal loss (fixed U)
    "orth_contrastive" — CE + orthogonal + InfoNCE contrastive
    "orth_cov"        — CE + orthogonal + covariance regularization
    "proj_head"       — CE through MLP projection head, orth on raw encoder output
    "dynamic_u"       — CE + orth with U rebuilt every epoch from current entity embeddings (per-type)
    "dynamic_u_hard"  — dynamic_u + classifier constrained to U subspace (hard projection)
    "dynamic_u_soft"  — dynamic_u + classifier weights regularized to stay within U subspace

FINETUNE_MODE options (independent of LOSS_MODE):
    "full"            — standard full fine-tuning (default)
    "lora"            — LoRA adapters only, most encoder weights frozen
    "layerwise_lr"    — layerwise learning rate decay (higher LR for later layers)

Usage:
    python finetune.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModel,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# LoRA import — optional, only needed for lora mode
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def apply_lora(model):
    """Wrap model with LoRA adapters."""
    if not PEFT_AVAILABLE:
        raise ImportError("peft not installed. Run: pip install peft")
    config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
    )
    return get_peft_model(model, config)


def get_layerwise_optimizer(model, base_lr):
    """
    Build optimizer with layerwise LR decay.
    Later layers get higher LR, earlier layers get lower LR.
    Classifier head gets base_lr, each lower layer gets * LAYERWISE_LR_DECAY.
    """
    # Get named layers — works for RoBERTa/BERT architecture
    no_decay = ["bias", "LayerNorm.weight"]

    # Classifier head — full LR
    optimizer_params = [
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n and not any(nd in n for nd in no_decay)],
         "lr": base_lr, "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n and any(nd in n for nd in no_decay)],
         "lr": base_lr, "weight_decay": 0.0},
    ]

    # Encoder layers — decaying LR
    num_layers = 12  # BERT/RoBERTa base
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (LAYERWISE_LR_DECAY ** (num_layers - 1 - layer_idx))
        layer_name = f"encoder.layer.{layer_idx}."
        # Also handle roberta prefix
        layer_names = [layer_name, f"roberta.{layer_name}", f"bert.{layer_name}"]
        optimizer_params.extend([
            {"params": [p for n, p in model.named_parameters()
                        if any(ln in n for ln in layer_names)
                        and not any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in model.named_parameters()
                        if any(ln in n for ln in layer_names)
                        and any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": 0.0},
        ])

    # Embeddings — lowest LR
    emb_lr = base_lr * (LAYERWISE_LR_DECAY ** num_layers)
    optimizer_params.extend([
        {"params": [p for n, p in model.named_parameters()
                    if "embeddings" in n and not any(nd in n for nd in no_decay)],
         "lr": emb_lr, "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if "embeddings" in n and any(nd in n for nd in no_decay)],
         "lr": emb_lr, "weight_decay": 0.0},
    ])

    # Filter out empty groups
    optimizer_params = [g for g in optimizer_params if len(g["params"]) > 0]
    return torch.optim.AdamW(optimizer_params)

# =======================
# CONFIG — EDIT HERE
# =======================
LOSS_MODE     = "ce"      # "ce" | "orth" | "orth_contrastive" | "orth_cov" | "proj_head" | "dynamic_u" | "dynamic_u_hard" | "dynamic_u_soft"
FINETUNE_MODE = "layerwise_lr"    # "full" | "lora" | "layerwise_lr"

MODEL_DIR = r"path/to/medroberta"
TRAIN_JSON = r"./data/multiclin/train.json"
DEV_JSON = r"./data/multiclin/dev.json"

OUTPUT_DIR = r"./finetuned_model" # will be suffixed with loss mode + finetune mode

MAX_LENGTH      = 512
EPOCHS          = 20
LR              = 1e-4
TRAIN_BATCH     = 8
EVAL_BATCH      = 8
WEIGHT_DECAY    = 0.01
SEED            = 42
FP16            = True
DEVICE          = "cuda"

# Class weights
O_WEIGHT      = 0.5
ENTITY_WEIGHT = 2.0

# Orth loss
LAMBDA_ORTH        = 0.1
LAMBDA_ORTH_WARMUP = 1.0
WARMUP_EPOCHS      = 5
N_COMPONENTS       = 80
N_COMPONENTS_HARD  = 15
K_PER_TYPE         = 50

# Contrastive loss
LAMBDA_CONT       = 0.1
TEMPERATURE       = 0.07
MIN_ENTITY_TOKENS = 2

# Covariance loss
LAMBDA_COV        = 0.1

# Confidence penalty — penalizes classifier confidence on O tokens being medical
USE_CONFIDENCE_PENALTY = False   # set True to enable
LAMBDA_CONF            = 0.1    # weight of confidence penalty

# LoRA config
LORA_R          = 16     # rank
LORA_ALPHA      = 32     # scaling
LORA_DROPOUT    = 0.1
LORA_TARGET     = ["query", "value"]  # which modules to apply LoRA to

# Layerwise LR config
LAYERWISE_LR_DECAY = 0.9   # multiply LR by this for each lower layer
                            # layer 12 gets LR, layer 11 gets LR*0.9, etc.

ENTITY_TYPES = ["DISEASE", "PROCEDURE", "SYMPTOM"]

assert LOSS_MODE in ("ce", "orth", "orth_contrastive", "orth_cov", "proj_head",
                     "dynamic_u", "dynamic_u_hard", "dynamic_u_soft"), \
    f"Unknown LOSS_MODE '{LOSS_MODE}'."
assert FINETUNE_MODE in ("full", "lora", "layerwise_lr"), \
    f"Unknown FINETUNE_MODE '{FINETUNE_MODE}'."

# Projection head config
PROJ_HEAD_DIM      = 256
WARMUP_EPOCHS_HARD = 5
LAMBDA_CLF_SOFT    = 0.1

OUTPUT_DIR = f"{OUTPUT_DIR.rstrip('/')}_{LOSS_MODE}_{FINETUNE_MODE}"


# =======================
# BUILD U
# =======================
def build_U(train_records, model, tokenizer, device):
    from collections import defaultdict
    type_token_embs = defaultdict(list)
    model.eval()

    for record in train_records:
        text     = record["text"]
        entities = record.get("entities", [])
        if not entities:
            continue

        encoding = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LENGTH, return_offsets_mapping=True)
        offset_mapping = encoding["offset_mapping"][0]
        encoding.pop("offset_mapping")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]

        for entity in entities:
            label = entity["label"]
            if len(type_token_embs[label]) >= 5000:
                continue
            span_token_ids = [
                idx for idx, (start, end) in enumerate(offset_mapping)
                if end > entity["start"] and start < entity["end"]
            ]
            for token_id in span_token_ids:
                if len(type_token_embs[label]) < 5000:
                    type_token_embs[label].append(hidden[token_id].cpu().numpy())

    all_centroids = []
    for label, embs in type_token_embs.items():
        embs_arr = np.vstack(embs)
        k = min(K_PER_TYPE, len(embs_arr))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(embs_arr)
        all_centroids.append(km.cluster_centers_)
        print(f"  {label}: {len(embs_arr)} token embs, k={k}")

    centroids = np.vstack(all_centroids)
    svd = TruncatedSVD(n_components=min(N_COMPONENTS, centroids.shape[0]-1), random_state=42)
    svd.fit(centroids)
    U = torch.tensor(svd.components_.T, dtype=torch.float32).to(device)
    print(f"  U shape: {U.shape}")
    return U


# =======================
# INFONCE LOSS
# =======================
def infonce_loss(embeddings, labels):
    embeddings = F.normalize(embeddings.float(), dim=-1)  # float32 to avoid FP16 overflow
    sim        = embeddings @ embeddings.T / TEMPERATURE
    N          = embeddings.size(0)
    diag_mask  = torch.eye(N, dtype=torch.bool, device=embeddings.device)
    sim        = sim.masked_fill(diag_mask, float('-inf'))

    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~diag_mask
    if not pos_mask.any():
        return torch.tensor(0.0, device=embeddings.device)

    log_softmax  = F.log_softmax(sim, dim=-1)
    pos_log_prob = (log_softmax * pos_mask.float()).sum(dim=-1)
    n_positives  = pos_mask.float().sum(dim=-1).clamp(min=1)
    loss         = -(pos_log_prob / n_positives)

    has_positive = pos_mask.any(dim=-1)
    if not has_positive.any():
        return torch.tensor(0.0, device=embeddings.device)
    return loss[has_positive].mean()


def covariance_loss(embeddings):
    """
    Penalizes correlation between dimensions of entity token embeddings.
    Forces the representation space to stay high-dimensional and non-degenerate.
    Based on VICReg/Barlow Twins covariance regularization.
    embeddings: (N, D)
    """
    embeddings = embeddings.float()
    N, D = embeddings.shape
    if N < 2:
        return torch.tensor(0.0, device=embeddings.device)
    embeddings = embeddings - embeddings.mean(0)
    cov = (embeddings.T @ embeddings) / (N - 1)  # (D, D)
    diag = torch.eye(D, device=embeddings.device, dtype=torch.bool)
    cov_loss = (cov[~diag] ** 2).sum() / D
    return cov_loss


def confidence_penalty_loss(logits, labels, O_idx):
    """
    Penalizes classifier confidence on O tokens being predicted as entities.
    For each O-labeled token, computes max entity class probability and penalizes it.
    High confidence false fires get penalized most — directly targets precision.

    logits: (batch, seq, num_labels)
    labels: (batch, seq)
    O_idx:  integer index of O class
    """
    flat_logits = logits.view(-1, logits.size(-1)).float()  # (N, num_labels)
    flat_labels = labels.view(-1)

    # Only O-labeled non-padding tokens
    o_mask = (flat_labels == O_idx)
    if not o_mask.any():
        return torch.tensor(0.0, device=logits.device)

    o_logits = flat_logits[o_mask]                          # (n_o, num_labels)
    probs     = F.softmax(o_logits, dim=-1)                 # (n_o, num_labels)
    entity_probs = torch.cat([probs[:, :O_idx],
                               probs[:, O_idx+1:]], dim=-1) # (n_o, num_labels-1)
    # Mean of max entity probability across O tokens
    return entity_probs.max(dim=-1).values.mean()


def classifier_weight_constraint(classifier, U):
    """
    Soft classifier constraint: penalizes classifier weight components outside U subspace.
    Forces decision boundary to align with medical subspace directions.
    classifier: nn.Linear (hidden_size, num_labels)
    U: (hidden_size, N_COMPONENTS)
    """
    W = classifier.weight.float()  # (num_labels, hidden_size)
    U_f = U.float()                # (hidden_size, N_COMPONENTS)
    # Project W onto U subspace
    W_proj = W @ U_f @ U_f.T      # (num_labels, hidden_size) — component in U
    W_orth = W - W_proj            # component outside U
    return (W_orth ** 2).mean()


def build_U_from_current_model(model, train_records, tokenizer, entity_type, device,
                               n_components=None):
    """
    Build per-type U from current model's entity token embeddings.
    Called every epoch to track the evolving medical subspace.
    """
    if n_components is None:
        n_components = N_COMPONENTS_HARD if LOSS_MODE == "dynamic_u_hard" else N_COMPONENTS
    from collections import defaultdict
    type_token_embs = []
    was_training = model.training
    model.eval()

    # Extract correct encoder to call with input_ids
    # Custom models (BertWithUProjection, BertWithProjHead) have .encoder = AutoModel
    # Everything else (AutoModelForTokenClassification, raw AutoModel) accepts input_ids directly
    if isinstance(model, (BertWithUProjection, BertWithProjHead)):
        encoder = model.encoder
    else:
        encoder = model

    for record in train_records:
        if entity_type.lower() not in record.get("id", "").lower():
            continue
        text     = record["text"]
        entities = record.get("entities", [])
        if not entities:
            continue

        encoding = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAX_LENGTH, return_offsets_mapping=True)
        offset_mapping = encoding["offset_mapping"][0]
        encoding.pop("offset_mapping")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = encoder(**encoding, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0].float()

        for entity in entities:
            if entity["label"] != entity_type:
                continue
            span_token_ids = [
                idx for idx, (start, end) in enumerate(offset_mapping)
                if end > entity["start"] and start < entity["end"]
            ]
            for token_id in span_token_ids:
                if len(type_token_embs) < 5000:
                    type_token_embs.append(hidden[token_id].cpu().numpy())

    if was_training:
        model.train()

    if len(type_token_embs) < n_components + 1:
        return None

    embs_arr = np.vstack(type_token_embs)
    k = min(K_PER_TYPE, len(embs_arr))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(embs_arr)
    centroids = km.cluster_centers_

    n_comp = min(n_components, centroids.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(centroids)
    components = svd.components_.T  # (hidden_size, n_comp)

    # Pad to n_components if SVD returned fewer components
    if components.shape[1] < n_components:
        pad = np.zeros((components.shape[0], n_components - components.shape[1]))
        components = np.concatenate([components, pad], axis=1)

    U = torch.tensor(components, dtype=torch.float32).to(device)
    return U


# =======================
# HARD PROJECTION CLASSIFIER MODEL
# =======================
class BertWithUProjection(nn.Module):
    """
    Classifier constrained to U subspace (hard version).
    Encoder output projected onto U before classification.
    O tokens orthogonal to U → near-zero projection → classifier cannot fire.
    """
    def __init__(self, encoder, num_labels, U):
        super().__init__()
        self.encoder    = encoder
        self.num_labels = num_labels
        self.register_buffer('U', U)  # (hidden_size, N_COMPONENTS)
        n_comp = U.shape[1]
        self.classifier = nn.Linear(n_comp, num_labels)

    def update_U(self, U):
        self.U = U.to(next(self.parameters()).device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, output_hidden_states=False, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        raw_hidden = outputs.last_hidden_state.float()           # (batch, seq, 768)
        proj       = raw_hidden @ self.U.float()                 # (batch, seq, N_COMP)
        logits     = self.classifier(proj).to(raw_hidden.dtype)  # (batch, seq, num_labels)

        from transformers.modeling_outputs import TokenClassifierOutput
        return TokenClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=(raw_hidden.to(raw_hidden.dtype),),
            attentions=None,
        )


# =======================
# PROJECTION HEAD MODEL
# =======================
class BertWithProjHead(nn.Module):
    """
    Wraps a BERT encoder with:
      - A small MLP projection head for CE (attenuates CE gradients to encoder)
      - Raw encoder output for orth loss (full gradient strength)
    """
    def __init__(self, encoder, num_labels, hidden_size=768, proj_dim=PROJ_HEAD_DIM):
        super().__init__()
        self.encoder    = encoder
        self.proj_head  = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, hidden_size),
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, output_hidden_states=False, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        raw_hidden  = outputs.last_hidden_state                        # (batch, seq, 768)
        proj_hidden = self.proj_head(raw_hidden.float()).to(raw_hidden.dtype)  # cast for FP16 compat
        logits      = self.classifier(proj_hidden)

        # Return a ModelOutput-compatible object
        from transformers.modeling_outputs import TokenClassifierOutput
        return TokenClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=(raw_hidden,),
            attentions=None,
        )


# =======================
# HELPERS
# =======================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_by_type(records):
    by_type = {"DISEASE": [], "SYMPTOM": [], "PROCEDURE": []}
    for rec in records:
        for entity_type in by_type:
            if entity_type.lower() in rec.get("id", "").lower():
                by_type[entity_type].append(rec)
                break
    return by_type


def normalize_entities(entities):
    out = []
    for e in entities:
        s, t, lab = int(e["start"]), int(e["end"]), str(e["label"])
        if t > s:
            out.append((s, t, lab))
    out.sort(key=lambda x: (-(x[1]-x[0]), x[0], x[1], x[2]))
    return out


def char_span_to_token_labels(offsets, special_mask, entities):
    labels      = [None] * len(offsets)
    token_spans = [None if special_mask[i] == 1 else (a, b)
                   for i, (a, b) in enumerate(offsets)]
    for (start, end, lab) in entities:
        touched = [i for i, span in enumerate(token_spans)
                   if span and not (span[1] <= start or span[0] >= end) and span[1] > span[0]]
        if not touched:
            continue
        touched.sort()
        first = True
        for i in touched:
            if labels[i] is not None:
                continue
            labels[i] = f"B-{lab}" if first else f"I-{lab}"
            first = False
    return labels


def to_dataset(records):
    return Dataset.from_list([
        {"id": r.get("id", ""), "text": r["text"],
         "entities": r.get("entities", []),
         "_char_offset": r.get("_char_offset", 0)}
        for r in records
    ])


# =======================
# UNIFIED TRAINER
# =======================
DYNAMIC_MODES = ("dynamic_u", "dynamic_u_hard", "dynamic_u_soft")

class UnifiedTrainer(Trainer):
    def __init__(self, label2id, U=None, steps_per_epoch=None,
                 train_records=None, tokenizer_ref=None, entity_type=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label2id        = label2id
        self.U               = U
        self.loss_mode       = LOSS_MODE
        self.steps_per_epoch = steps_per_epoch
        self.train_records   = train_records    # needed for dynamic U rebuild
        self.tokenizer_ref   = tokenizer_ref    # needed for dynamic U rebuild
        self.entity_type     = entity_type      # needed for per-type U rebuild
        self._last_rebuild_epoch = -1

    def _maybe_rebuild_U(self):
        """Rebuild U from current model every epoch in dynamic modes.
        For dynamic_u_hard, U is frozen for first WARMUP_EPOCHS_HARD epochs
        to allow classifier to stabilize before subspace starts shifting."""
        if self.loss_mode not in DYNAMIC_MODES:
            return
        if self.steps_per_epoch is None:
            return
        current_epoch = int(self.state.global_step / self.steps_per_epoch)
        if current_epoch == self._last_rebuild_epoch:
            return
        self._last_rebuild_epoch = current_epoch

        # For hard mode: freeze U during warmup so classifier can stabilize
        if self.loss_mode == "dynamic_u_hard" and current_epoch < WARMUP_EPOCHS_HARD:
            return

        new_U = build_U_from_current_model(
            self.model, self.train_records, self.tokenizer_ref,
            self.entity_type, DEVICE
        )
        if new_U is not None:
            self.U = new_U
            if self.loss_mode == "dynamic_u_hard" and hasattr(self.model, 'update_U'):
                self.model.update_U(new_U)

    def _current_lambda_orth(self):
        """Warmup: high lambda for first WARMUP_EPOCHS, then anneal to LAMBDA_ORTH."""
        if self.steps_per_epoch is None or self.loss_mode == "ce":
            return LAMBDA_ORTH
        current_epoch = self.state.global_step / self.steps_per_epoch
        if current_epoch < WARMUP_EPOCHS:
            frac = current_epoch / WARMUP_EPOCHS
            return LAMBDA_ORTH_WARMUP - frac * (LAMBDA_ORTH_WARMUP - LAMBDA_ORTH)
        return LAMBDA_ORTH

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._maybe_rebuild_U()

        labels  = inputs.get("labels")
        need_hidden = self.loss_mode in (
            "orth", "orth_contrastive", "orth_cov", "proj_head",
            "dynamic_u", "dynamic_u_hard", "dynamic_u_soft"
        )
        outputs = model(**inputs, output_hidden_states=need_hidden)
        logits  = outputs.logits

        # --- CE ---
        num_labels = logits.shape[-1]
        weights    = torch.ones(num_labels, device=logits.device, dtype=logits.dtype)
        for label, idx in self.label2id.items():
            weights[idx] = O_WEIGHT if label == "O" else ENTITY_WEIGHT
        ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)(
            logits.view(-1, num_labels), labels.view(-1)
        )

        if self.loss_mode == "ce":
            if USE_CONFIDENCE_PENALTY:
                O_idx = self.label2id["O"]
                conf_loss = confidence_penalty_loss(logits, labels, O_idx)
                loss = ce_loss + LAMBDA_CONF * conf_loss
            else:
                loss = ce_loss
            return (loss, outputs) if return_outputs else loss

        hidden = outputs.hidden_states[-1]  # (batch, seq, 768)
        O_idx  = self.label2id["O"]

        # --- Orth ---
        orth_loss = torch.tensor(0.0, device=logits.device)
        if self.loss_mode in ("orth", "orth_contrastive", "orth_cov", "proj_head",
                              "dynamic_u", "dynamic_u_hard", "dynamic_u_soft"):
            o_mask = (labels == O_idx)
            if o_mask.any() and self.U is not None:
                o_embs = hidden[o_mask].float()
                if o_embs.dim() == 1:
                    o_embs = o_embs.unsqueeze(0)
                proj      = o_embs @ self.U.to(o_embs.device)
                orth_loss = (proj ** 2).mean()

        # --- Contrastive ---
        cont_loss = torch.tensor(0.0, device=logits.device)
        if self.loss_mode == "orth_contrastive":
            flat_hidden = hidden.view(-1, hidden.size(-1))
            flat_labels = labels.view(-1)
            entity_mask = (flat_labels != -100) & (flat_labels != O_idx)
            if entity_mask.sum() >= MIN_ENTITY_TOKENS:
                cont_loss = infonce_loss(flat_hidden[entity_mask], flat_labels[entity_mask])

        # --- Covariance ---
        cov_loss = torch.tensor(0.0, device=logits.device)
        if self.loss_mode == "orth_cov":
            flat_hidden = hidden.view(-1, hidden.size(-1))
            flat_labels = labels.view(-1)
            entity_mask = (flat_labels != -100) & (flat_labels != O_idx)
            if entity_mask.sum() >= 2:
                cov_loss = covariance_loss(flat_hidden[entity_mask])

        # --- Soft classifier constraint ---
        clf_loss = torch.tensor(0.0, device=logits.device)
        if self.loss_mode == "dynamic_u_soft" and self.U is not None:
            # Get classifier — handle both model types
            clf = self.model.classifier if hasattr(self.model, 'classifier') else None
            if clf is not None:
                clf_loss = classifier_weight_constraint(clf, self.U)

        # --- Confidence penalty ---
        conf_loss = torch.tensor(0.0, device=logits.device)
        if USE_CONFIDENCE_PENALTY:
            conf_loss = confidence_penalty_loss(logits, labels, O_idx)

        lambda_orth = self._current_lambda_orth()
        loss = (ce_loss
                + lambda_orth * orth_loss
                + LAMBDA_CONT * cont_loss
                + LAMBDA_COV * cov_loss
                + LAMBDA_CLF_SOFT * clf_loss
                + LAMBDA_CONF * conf_loss)
        return (loss, outputs) if return_outputs else loss


# =======================
# TRAIN ONE ENTITY TYPE
# =======================
def train_one_type(entity_type, train_records, dev_records, tokenizer, U=None):
    print(f"\n{'='*60}")
    print(f"Training: {entity_type}  |  mode: {LOSS_MODE.upper()}")
    print(f"{'='*60}")

    train_filtered = split_by_type(train_records)[entity_type]
    dev_filtered   = split_by_type(dev_records)[entity_type]
    print(f"  Train: {len(train_filtered)} records, Dev: {len(dev_filtered)} records")

    label_list = ["O", f"B-{entity_type}", f"I-{entity_type}"]
    label2id   = {l: i for i, l in enumerate(label_list)}
    id2label   = {i: l for l, i in label2id.items()}

    def tokenize_and_align(ex):
        """
        Split record text on newlines, tokenize each line separately.
        Returns multiple examples if text has multiple lines.
        Each line is tokenized with truncation=512.
        Character offsets are shifted per line to map back to original text.
        """
        ents     = normalize_entities(ex["entities"])
        text     = ex["text"]
        lines    = text.split('\n')
        char_pos = 0
        all_examples = []

        for line in lines:
            if line.strip():
                tok = tokenizer(
                    line, truncation=True, max_length=MAX_LENGTH,
                    return_offsets_mapping=True, return_special_tokens_mask=True
                )
                offsets      = [(s + char_pos, e + char_pos)
                                for s, e in tok["offset_mapping"]]
                special_mask = tok["special_tokens_mask"]
                bio_labels   = char_span_to_token_labels(offsets, special_mask, ents)
                label_ids    = []
                for i, lab in enumerate(bio_labels):
                    if special_mask[i] == 1:
                        label_ids.append(-100)
                    else:
                        label_ids.append(label2id.get(lab, label2id["O"]) if lab is not None else label2id["O"])
                example = {
                    "input_ids":      tok["input_ids"],
                    "attention_mask": tok["attention_mask"],
                    "labels":         label_ids,
                }
                if "token_type_ids" in tok:
                    example["token_type_ids"] = tok["token_type_ids"]
                all_examples.append(example)

            char_pos += len(line) + 1  # +1 for newline

        return all_examples

    def tokenize_and_align_flat(ex):
        """Wrapper that returns first example — used for dataset.map."""
        examples = tokenize_and_align(ex)
        if not examples:
            # Empty record — return dummy
            tok = tokenizer(".", truncation=True, max_length=MAX_LENGTH)
            return {
                "input_ids":      tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "labels":         [-100] * len(tok["input_ids"]),
            }
        # For multi-line records, we flatten by returning all examples
        # Dataset.map with batched=False can only return one item,
        # so we concatenate all lines into one sequence if they fit,
        # otherwise return the first line only
        # TODO: implement proper multi-example expansion
        return examples[0]

    def expand_records(records):
        """
        Pre-expand records into per-line examples before creating dataset.
        Each line of each record becomes a separate example.
        """
        expanded = []
        for record in records:
            ents     = normalize_entities(record["entities"])
            text     = record["text"]
            lines    = text.split('\n')
            char_pos = 0

            for line in lines:
                if line.strip():
                    expanded.append({
                        "text":     line,
                        "entities": record["entities"],
                        "_char_offset": char_pos,
                        "id":       record.get("id", ""),
                    })
                char_pos += len(line) + 1

        return expanded

    def tokenize_and_align_line(ex):
        """Tokenize a single line with its char offset applied."""
        ents        = normalize_entities(ex["entities"])
        char_offset = ex.get("_char_offset", 0)
        tok = tokenizer(
            ex["text"], truncation=True, max_length=MAX_LENGTH,
            return_offsets_mapping=True, return_special_tokens_mask=True
        )
        offsets      = [(s + char_offset, e + char_offset)
                        for s, e in tok["offset_mapping"]]
        special_mask = tok["special_tokens_mask"]
        bio_labels   = char_span_to_token_labels(offsets, special_mask, ents)
        label_ids    = []
        for i, lab in enumerate(bio_labels):
            if special_mask[i] == 1:
                label_ids.append(-100)
            else:
                label_ids.append(label2id.get(lab, label2id["O"]) if lab is not None else label2id["O"])
        tok.pop("offset_mapping")
        tok.pop("special_tokens_mask")
        tok["labels"] = label_ids
        return tok

    # Expand records into per-line examples before creating dataset
    train_expanded = expand_records(train_filtered)
    dev_expanded   = expand_records(dev_filtered)
    print(f"  Train: {len(train_filtered)} records → {len(train_expanded)} line examples")
    print(f"  Dev:   {len(dev_filtered)} records → {len(dev_expanded)} line examples")

    train_ds  = to_dataset(train_expanded)
    dev_ds    = to_dataset(dev_expanded)
    train_tok = train_ds.map(tokenize_and_align_line, remove_columns=train_ds.column_names)
    dev_tok   = dev_ds.map(tokenize_and_align_line,   remove_columns=dev_ds.column_names)

    if LOSS_MODE == "proj_head":
        encoder = AutoModel.from_pretrained(MODEL_DIR)
        model   = BertWithProjHead(encoder, num_labels=len(label_list))
    elif LOSS_MODE == "dynamic_u_hard":
        encoder = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
        print(f"  Building initial per-type U for {entity_type}...")
        initial_U = build_U_from_current_model(
            encoder, train_records, tokenizer, entity_type, DEVICE
        )
        if initial_U is None:
            print(f"  Warning: not enough {entity_type} tokens for U, using random init")
            initial_U = torch.nn.functional.normalize(
                torch.randn(768, N_COMPONENTS), dim=0
            ).to(DEVICE)
        encoder = encoder.cpu()
        encoder_wrap = AutoModel.from_pretrained(MODEL_DIR)
        model = BertWithUProjection(encoder_wrap, num_labels=len(label_list), U=initial_U.cpu())
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_DIR,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    # Apply LoRA if requested
    if FINETUNE_MODE == "lora" and LOSS_MODE not in ("proj_head", "dynamic_u_hard"):
        model = apply_lora(model)
        model.print_trainable_parameters()

    # Build optimizer for layerwise LR
    optimizer = None
    if FINETUNE_MODE == "layerwise_lr" and LOSS_MODE not in ("proj_head", "dynamic_u_hard"):
        optimizer = get_layerwise_optimizer(model, LR)
        print(f"  Layerwise LR: base={LR}, decay={LAYERWISE_LR_DECAY}")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        true_labels, true_preds = [], []
        for pred_seq, lab_seq in zip(preds, labels):
            sl, sp = [], []
            for p, l in zip(pred_seq, lab_seq):
                if l == -100:
                    continue
                sl.append(id2label[int(l)])
                sp.append(id2label[int(p)])
            true_labels.append(sl)
            true_preds.append(sp)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall":    recall_score(true_labels, true_preds),
            "f1":        f1_score(true_labels, true_preds),
        }

    output_dir = os.path.join(OUTPUT_DIR, entity_type.lower())

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=FP16,
        seed=SEED,
        report_to="none",
    )

    steps_per_epoch = max(1, len(train_tok) // TRAIN_BATCH)

    trainer_kwargs = dict(
        label2id=label2id,
        U=U,
        steps_per_epoch=steps_per_epoch,
        train_records=train_records,
        tokenizer_ref=tokenizer,
        entity_type=entity_type,
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Pass custom optimizer for layerwise LR
    if optimizer is not None:
        trainer_kwargs["optimizers"] = (optimizer, None)  # (optimizer, scheduler) — None = default scheduler

    trainer = UnifiedTrainer(**trainer_kwargs)

    trainer.train()

    # ignore hidden_states during prediction for custom models
    ignore_keys = ["hidden_states", "attentions"] if LOSS_MODE in (
        "proj_head", "dynamic_u_hard"
    ) else None

    preds_out = trainer.predict(dev_tok, ignore_keys=ignore_keys)
    pred_ids  = preds_out.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_ids   = np.argmax(pred_ids, axis=-1)
    labels_out = preds_out.label_ids

    true_labels, true_preds = [], []
    for pred_seq, lab_seq in zip(pred_ids, labels_out):
        sl, sp = [], []
        for p, l in zip(pred_seq, lab_seq):
            if l == -100:
                continue
            sl.append(id2label[int(l)])
            sp.append(id2label[int(p)])
        true_labels.append(sl)
        true_preds.append(sp)

    p = precision_score(true_labels, true_preds)
    r = recall_score(true_labels, true_preds)
    f = f1_score(true_labels, true_preds)

    print(f"\n=== {entity_type} DEV REPORT ===")
    print(classification_report(true_labels, true_preds, digits=4))

    false_fires = boundary_errs = total_fp = 0
    for gold_seq, pred_seq in zip(true_labels, true_preds):
        for g, p_ in zip(gold_seq, pred_seq):
            if p_ != "O" and g == "O":
                false_fires += 1; total_fp += 1
            elif p_ != "O" and g != "O" and p_ != g:
                boundary_errs += 1; total_fp += 1
    if total_fp > 0:
        print(f"  False fires:     {false_fires} ({100*false_fires/total_fp:.1f}% of FP)")
        print(f"  Boundary errors: {boundary_errs} ({100*boundary_errs/total_fp:.1f}% of FP)")
    else:
        print("  No false positives.")

    if LOSS_MODE in ("proj_head", "dynamic_u_hard"):
        os.makedirs(output_dir, exist_ok=True)
        trainer.model.encoder.save_pretrained(output_dir)
        extra = {"classifier": trainer.model.classifier.state_dict()}
        if hasattr(trainer.model, 'proj_head'):
            extra["proj_head"] = trainer.model.proj_head.state_dict()
        if hasattr(trainer.model, 'U'):
            extra["U"] = trainer.model.U.cpu()
        torch.save(extra, os.path.join(output_dir, "head.pt"))
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to: {output_dir}")

    return {"precision": p, "recall": r, "f1": f}


# =======================
# MAIN
# =======================
def main():
    print(f"\nLOSS MODE: {LOSS_MODE.upper()}")
    print(f"Output dir: {OUTPUT_DIR}\n")

    train_records = load_json(TRAIN_JSON)
    dev_records   = load_json(DEV_JSON)
    tokenizer     = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    U = None
    if LOSS_MODE in ("orth", "orth_contrastive", "orth_cov", "proj_head"):
        print("Building U from MultiClinNER training tokens...")
        tmp_model = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
        U = build_U(train_records, tmp_model, tokenizer, DEVICE)
        del tmp_model
        torch.cuda.empty_cache()

    results = {}
    for entity_type in ENTITY_TYPES:
        results[entity_type] = train_one_type(entity_type, train_records, dev_records, tokenizer, U)

    print("\n" + "="*60)
    print(f"FINAL SUMMARY  [{LOSS_MODE.upper()}]")
    print(f"  {'Type':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for entity_type, scores in results.items():
        print(f"  {entity_type:<12} {scores['precision']:>10.4f} {scores['recall']:>10.4f} {scores['f1']:>10.4f}")
    macro_p = sum(s['precision'] for s in results.values()) / len(results)
    macro_r = sum(s['recall']    for s in results.values()) / len(results)
    macro_f = sum(s['f1']        for s in results.values()) / len(results)
    print(f"  {'-'*44}")
    print(f"  {'Macro avg':<12} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>10.4f}")


if __name__ == "__main__":
    main()
