# DOKTERBERT

Dutch clinical language model with SNOMED CT-grounded contrastive pretraining and contextual span anchoring (SMM4H/HeaRD @ ACL 2026).

## Overview

DOKTERBERT is a Dutch clinical language model pretrained with a SNOMED CT-grounded
contrastive objective. Built on MedRoBERTa.nl, it aligns contextual span
representations to SNOMED concept anchors, with contrastive pressure weighted by
graph distance in the SNOMED IS-A hierarchy, organising clinical concept
representations against the ontology rather than treating terms in isolation.

This repository contains the code accompanying the paper *DOKTERBERT at
\#SMM4H–HeaRD 2026: Ontology-Grounded Contextual Representations for Dutch
Clinical NLP*. It targets reproducibility of the method and results, not
production use.

## What you need to provide

The repository contains code only. You supply the following yourself:

- **SNOMED CT, Dutch edition (RF2 release).** Requires a licence; in the
  Netherlands it is available free of charge through Nictiz. No SNOMED-derived
  data is redistributed here. The pipeline was built against release
  `NL1000146_20250831` (Dutch edition, 31 August 2025); other releases will
  shift term sets and graph distances.
- **A Dutch clinical text corpus** for continued pretraining (plain `.txt`
  files).
- **MedRoBERTa.nl and SapBERT** model checkpoints (from HuggingFace).
- **The MultiClinNER-nl data**, available to shared task participants through
  the organisers, for fine-tuning and evaluation.

Pretrained DOKTERBERT weights are released separately on HuggingFace.

## Setup

Paths are configured in a block at the top of each script. Values written as
`path/to/...` are placeholders that you must point at your own files;
relative `./data/...` paths are intermediate files the pipeline creates and
can usually be left as they are.

```bash
pip install -r requirements.txt
python -m spacy download nl_core_news_lg
```

Requires Python 3.10+ and a CUDA-capable GPU for the pretraining, fine-tuning,
and evaluation stages. `requirements.txt` lists `faiss-gpu`; on a CPU-only
machine, install `faiss-cpu` instead.

## Pipeline

The scripts are numbered in execution order.

**SNOMED preparation**

1. `01_build_snomed_termset.py` — reads the RF2 release, walks the IS-A
   hierarchy to assign each concept a semantic type (disorder / finding /
   procedure), and writes `data/snomed_nl.tsv` (concept id, Dutch term,
   semantic type).
2. `02_build_snomed_distances.py` — builds the IS-A graph and precomputes
   shortest-path distances between concepts, writing
   `data/snomed_distances.json`.
3. `03_build_snomed_index.py` — encodes the SNOMED terms and builds the FAISS
   index (`snomed.index`, `snomed_meta.json`) used for similarity linking.

**Corpus processing**

4. `04_extract_spans.py` — extracts candidate medical spans from the corpus
   with spaCy dependency parsing, writing `data/candidates.jsonl` and
   `data/sentences.jsonl`.
5. `05_link_spans.py` — links each candidate span to a SNOMED concept by exact
   match, with a SapBERT similarity fallback, writing `data/spans.jsonl`.

**Pretraining**

6. `06_pretrain_dokterbert.py` — continues pretraining from MedRoBERTa.nl with
   the distance-weighted contrastive objective combined with masked language
   modelling.

**Evaluation**

7. `07_finetune_ner.py` — supervised NER fine-tuning on MultiClinNER-nl.
8. `08_geometry_checks.py` — representational analysis (retrieval, clustering,
   concept discrimination, intra/inter-concept geometry). Run once per model.
9. `09_inference.py` — runs fine-tuned models on test text and writes
   `run.tsv`.

Stages 1–3 and 4–5 are independent of each other and can run in either order;
stage 6 depends on the outputs of 2 and 5; stages 7–9 use the model from 6.

> **Note.** Fine-tuning, inference, and evaluation expect the MultiClinNER-nl
> data as JSON (`data/multiclin/train.json`, `dev.json`, and per-type test
> files). Converting the shared task release into this format is a small
> preprocessing step not included here.

## Key result

On a representational analysis against RobBERT, MedRoBERTa.nl, and
MedRoBERTa.nl-SapBERT, DOKTERBERT substantially outperforms every baseline on
concept-level discrimination (discrimination gap +0.592 vs. +0.170 for the
next-best model). Contextual span anchoring produces ontology-aligned
representations that isolated-term contrastive training does not.

## Citation

```bibtex
@inproceedings{danoe2026dokterbert,
  title     = {DOKTERBERT at \#SMM4H--HeaRD 2026: Ontology-Grounded Contextual Representations for Dutch Clinical NLP},
  author    = {Danoe, Gijs and Berends, Matthijs S. and Voss, Andreas and Hamprecht, Axel},
  booktitle = {Proceedings of the SMM4H/HeaRD Workshop},
  year      = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
