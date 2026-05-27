import os
import re
import json
import gc
import logging
from pathlib import Path
import time

import spacy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
CORPUS_DIR = r"path/to/corpus"
SPANS_FILE = "./data/candidates.jsonl"
SENTENCES_FILE = "./data/sentences.jsonl"
N_PROCESS = 8
BATCH_SIZE = 128
READ_CHUNK_SIZE = 100000
MAX_LINE_LEN = 2000


def iter_line_chunks(txt_files, chunk_size):
    chunk = []
    for fpath in txt_files:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if len(line) <= 10:
                    continue
                if len(line) > MAX_LINE_LEN:
                    line = line[:MAX_LINE_LEN]
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def extract_spans_from_doc(doc, sentence_id_counter):
    spans = []
    sentences = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        sent_char_start = sent.start_char

        if len(sentence_text) < 5:
            continue

        sent_id = sentence_id_counter[0]
        sentence_id_counter[0] += 1
        sentences.append({"id": sent_id, "sentence": sentence_text})

        seen = set()

        def add_span(start_tok, end_tok):
            span = doc[start_tok:end_tok]
            span_text = span.text.strip()
            if len(span_text) < 3:
                return
            char_start = span.start_char - sent_char_start
            char_end = span.end_char - sent_char_start
            key = (char_start, char_end)
            if key in seen:
                return
            seen.add(key)
            spans.append({
                "text": span_text,
                "sentence_id": sent_id,
                "char_start": char_start,
                "char_end": char_end,
                "has_context": True,
            })

        for chunk in sent.noun_chunks:
            add_span(chunk.start, chunk.end)

        for token in sent:
            if token.dep_ == "conj" and token.pos_ in ("NOUN", "PROPN"):
                left = min(
                    (t.i for t in token.children if t.dep_ == "amod"),
                    default=token.i
                )
                add_span(left, token.i + 1)
                add_span(token.i, token.i + 1)

            if token.pos_ == "ADJ" and token.dep_ in ("acomp", "xcomp"):
                add_span(token.i, token.i + 1)

            if token.dep_ == "prep" and token.head.pos_ in ("NOUN", "PROPN"):
                add_span(token.head.i, token.right_edge.i + 1)

            if token.pos_ == "PROPN" and token.dep_ in ("nsubj", "obj", "nmod", "appos", "conj", "ROOT"):
                add_span(token.i, token.i + 1)

    return spans, sentences


def main():
    nlp = spacy.load("nl_core_news_lg", disable=["ner", "lemmatizer"])

    txt_files = sorted(Path(CORPUS_DIR).glob("*.txt"))
    logger.info(f"Found {len(txt_files)} files")

    os.makedirs(os.path.dirname(SPANS_FILE) or ".", exist_ok=True)
    total_spans = 0
    total_sentences = 0
    sentence_id_counter = [0]
    chunk_idx = 0
    start_time = time.time()

    with open(SPANS_FILE, "w", encoding="utf-8") as spans_out, \
            open(SENTENCES_FILE, "w", encoding="utf-8") as sents_out:

        for lines in iter_line_chunks(txt_files, READ_CHUNK_SIZE):
            chunk_start = time.time()

            for doc in nlp.pipe(lines, batch_size=BATCH_SIZE, n_process=N_PROCESS):
                spans, sentences = extract_spans_from_doc(doc, sentence_id_counter)
                for sent in sentences:
                    sents_out.write(json.dumps(sent, ensure_ascii=False) + "\n")
                for span in spans:
                    spans_out.write(json.dumps(span, ensure_ascii=False) + "\n")
                total_spans += len(spans)
                total_sentences += len(sentences)

            chunk_idx += 1
            elapsed = (time.time() - start_time) / 60
            chunk_time = (time.time() - chunk_start) / 60
            rate = total_sentences / elapsed if elapsed > 0 else 0

            logger.info(
                f"Chunk {chunk_idx} | "
                f"{chunk_time:.1f}min/chunk | "
                f"{total_sentences} sents | "
                f"{rate:.0f} sents/min | "
                f"{elapsed:.0f}min elapsed"
            )
            gc.collect()

    logger.info(f"Done. {total_spans} spans → {SPANS_FILE}")
    logger.info(f"      {total_sentences} sentences → {SENTENCES_FILE}")


if __name__ == "__main__":
    main()

