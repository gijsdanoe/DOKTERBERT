import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
CANDIDATES_FILE = "./data/candidates.jsonl"
SENTENCES_FILE = "./data/sentences.jsonl"
SNOMED_FILE = "./data/snomed_nl.tsv"
OUTPUT_FILE = "./data/spans.jsonl"
SAPBERT_MODEL = r"path/to/sapbert"
THRESHOLD = 0.85
BATCH_SIZE = 512
MAX_LENGTH = 64


STOPWORDS = {
    "hij", "zij", "het", "de", "een", "aan", "bij", "van",
    "op", "in", "te", "en", "of", "maar", "als", "dat",
    "die", "dit", "wat", "wie", "hoe", "wel", "niet", "ook",
    "nog", "dan", "nu", "al", "zo", "er", "ze", "we", "je",
    "me", "u", "uw", "ik", "mij", "hem", "haar", "hun", "hen",
    "zich", "om", "uit", "na", "af", "toe", "mee", "door",
    "over", "voor", "naar", "zijn", "was", "werd", "heeft",
    "had", "hebben", "worden", "wordt", "kan", "kon", "mag",
    "mocht", "moet", "moest", "wil", "wilde", "zal", "zou",
    "geen", "veel", "meer", "zeer", "heel", "erg", "andere",
    "beide", "alle", "elke", "deze"
}


def load_snomed(snomed_file):
    term_to_concept = {}
    with open(snomed_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            concept_id, term, sem_type = parts[0], parts[1], parts[2]
            term_to_concept[term.lower()] = (concept_id, sem_type)
    logger.info(f"Loaded {len(term_to_concept)} SNOMED terms")
    return term_to_concept


def load_candidates(candidates_file):
    candidates = []
    with open(candidates_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    logger.info(f"Loaded {len(candidates)} candidate spans")
    return candidates


def embed_batch(texts, model, tokenizer, device):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**enc)
        embs = outputs.last_hidden_state[:, 0, :]
        embs = F.normalize(embs, dim=-1)
    return embs.cpu().numpy().astype("float32")


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeddings)
    return index


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    term_to_concept = load_snomed(SNOMED_FILE)
    candidates = load_candidates(CANDIDATES_FILE)

    logger.info(f"Loading SapBERT: {SAPBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
    model = AutoModel.from_pretrained(SAPBERT_MODEL).to(device)
    model.eval()

    # Build SNOMED FAISS index
    logger.info("Embedding SNOMED terms...")
    snomed_terms = list(term_to_concept.keys())
    all_snomed_embs = []
    for i in tqdm(range(0, len(snomed_terms), BATCH_SIZE), desc="SNOMED"):
        batch = snomed_terms[i:i+BATCH_SIZE]
        all_snomed_embs.append(embed_batch(batch, model, tokenizer, device))
    snomed_embs = np.concatenate(all_snomed_embs, axis=0)
    index = build_faiss_index(snomed_embs)
    snomed_concept_ids = [term_to_concept[t][0] for t in snomed_terms]
    snomed_sem_types = [term_to_concept[t][1] for t in snomed_terms]
    logger.info(f"SNOMED index built: {index.ntotal} vectors")

    exact_matched = set()
    linked_count = 0
    nonmedical_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        # Step 1 — Exact match
        logger.info("Running exact match...")
        for i, c in enumerate(tqdm(candidates, desc="Exact match")):
            term = c["text"].lower()
            if term in STOPWORDS:
                continue
            if term in term_to_concept:
                concept_id, sem_type = term_to_concept[term]
                out.write(json.dumps({
                    **c,
                    "concept_id": concept_id,
                    "semantic_type": sem_type,
                    "is_medical": True,
                    "source": "exact_match",
                }, ensure_ascii=False) + "\n")
                exact_matched.add(i)
                linked_count += 1

        logger.info(f"Exact match: {len(exact_matched)} linked")

        # Step 2 — SapBERT in batches
        unmatched_indices = [
            i for i in range(len(candidates))
            if i not in exact_matched
            and candidates[i]["text"].lower() not in STOPWORDS
        ]

        logger.info(f"SapBERT linking {len(unmatched_indices)} unmatched spans...")

        for batch_start in tqdm(range(0, len(unmatched_indices), BATCH_SIZE), desc="SapBERT"):
            batch_idx = unmatched_indices[batch_start:batch_start+BATCH_SIZE]
            batch_texts = [candidates[i]["text"] for i in batch_idx]

            batch_embs = embed_batch(batch_texts, model, tokenizer, device)
            distances, indices = index.search(batch_embs, k=1)

            for j, orig_idx in enumerate(batch_idx):
                c = candidates[orig_idx]
                dist = float(distances[j][0])
                idx = int(indices[j][0])
                if dist >= THRESHOLD:
                    out.write(json.dumps({
                        **c,
                        "concept_id": snomed_concept_ids[idx],
                        "semantic_type": snomed_sem_types[idx],
                        "is_medical": True,
                        "source": "sapbert",
                        "similarity": dist,
                    }, ensure_ascii=False) + "\n")
                    linked_count += 1
                else:
                    out.write(json.dumps({
                        **c,
                        "concept_id": None,
                        "semantic_type": None,
                        "is_medical": False,
                        "source": "non_linked",
                    }, ensure_ascii=False) + "\n")
                    nonmedical_count += 1

        # Stopword spans as non-medical
        for i, c in enumerate(candidates):
            if c["text"].lower() in STOPWORDS:
                out.write(json.dumps({
                    **c,
                    "concept_id": None,
                    "semantic_type": None,
                    "is_medical": False,
                    "source": "stopword",
                }, ensure_ascii=False) + "\n")
                nonmedical_count += 1

    logger.info(
        f"\nDone → {OUTPUT_FILE}\n"
        f"  Exact match linked:  {len(exact_matched)}\n"
        f"  SapBERT linked:      {linked_count - len(exact_matched)}\n"
        f"  Non-medical:         {nonmedical_count}\n"
        f"  Total:               {linked_count + nonmedical_count}"
    )


if __name__ == "__main__":
    main()
