# DOKTERBERT

Dutch clinical language model with SNOMED CT-grounded contrastive pretraining and contextual span anchoring (SMM4H/HeaRD @ ACL 2026).

## Overview

DOKTERBERT is a Dutch clinical language model pretrained with SNOMED CT-grounded
contrastive learning. Built on MedRoBERTa.nl, it uses distance-weighted contextual
span anchoring to organise clinical concept representations against the SNOMED CT
ontology, rather than treating terms in isolation as prior synonym-collapse
approaches do.

This repository contains the code accompanying the paper *DOKTERBERT:
Ontology-Grounded Contextual Representations for Dutch Clinical NLP*
(SMM4H/HeaRD workshop, ACL 2026).

## Repository contents

| File | Description |
|------|-------------|
| `dokterbert_train.py` | Contrastive pretraining with SNOMED-grounded concept anchors |
| `finetune.py` | Supervised NER fine-tuning on MultiClinNER-nl |
| `geometry_checks.py` | Representational analysis (retrieval, clustering, discrimination gap, intra/inter-concept geometry) |
| `build_snomed_index.py` | Builds the FAISS index over SNOMED CT concepts |
| `linking.py` | Entity-to-SNOMED linking pipeline |
| `spacy_extraction.py` | SOAP-structured preprocessing of GP consultation notes |
| `inference.py` | NER inference |

## Installation

```bash
git clone https://github.com/gijsdanoe/dokterbert
cd dokterbert
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU for pretraining and fine-tuning.

## Usage

The pipeline runs in four stages. Paths and hyperparameters are set in the
config block at the top of each script.

1. **Preprocess** the consultation notes into SOAP-structured spans:
```bash
   python spacy_extraction.py
```

2. **Build the SNOMED index** used for concept anchoring and linking:
```bash
   python build_snomed_index.py
```

3. **Pretrain** DOKTERBERT with the contrastive objective:
```bash
   python dokterbert_train.py
```

4. **Fine-tune** for NER and **evaluate** the representations:
```bash
   python finetune.py
   python geometry_checks.py
```

For inference with a trained model:
```bash
python inference.py
```

## Key result

On a representational analysis against RobBERT, MedRoBERTa.nl, and
MedRoBERTa-SapBERT, DOKTERBERT substantially outperforms all baselines on
concept-level discrimination (discrimination gap +0.592 vs. +0.170 for the
next-best model). Contextual span anchoring produces ontology-aligned
representations that isolated-term contrastive training does not.

## Data and model weights

The pretraining corpus consists of Dutch GP consultation notes and is not
publicly available. The MultiClinNER-nl benchmark used for evaluation is
publicly available. Pretrained model weights are released separately on
HuggingFace: https://huggingface.co/gijsdanoe/DOKTERBERT.

## Citation

```bibtex
@inproceedings{danoe2026dokterbert,
  title     = {DOKTERBERT: Ontology-Grounded Contextual Representations for Dutch Clinical NLP},
  author    = {Danoe, Gijs and others},
  booktitle = {Proceedings of SMM4H/HeaRD},
  year      = {2026}
}
```

## License

[Choose a license — MIT is common for research code.]
