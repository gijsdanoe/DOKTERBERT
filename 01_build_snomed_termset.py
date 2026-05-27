import os
import pandas as pd
from collections import defaultdict

# Load active descriptions (Dutch)
desc = pd.read_csv(
    "path/to/snomed-release/Snapshot/Terminology/sct2_Description_Snapshot-nl_NL1000146_20250831.txt",
    sep="\t", dtype=str
)
concepts = pd.read_csv(
    "path/to/snomed-release/Snapshot/Terminology/sct2_Concept_Snapshot_NL1000146_20250831.txt",
    sep="\t", dtype=str
)
rel = pd.read_csv(
    "path/to/snomed-release/Snapshot/Terminology/sct2_Relationship_Snapshot_NL1000146_20250831.txt",
    sep="\t", dtype=str
)

# Active concepts only
active_concepts = set(concepts[concepts["active"] == "1"]["id"])

# Active Dutch descriptions
desc_nl = desc[
    (desc["languageCode"] == "nl") &
    (desc["active"] == "1") &
    (desc["conceptId"].isin(active_concepts))
]

# Active relationships — IS-A only (typeId 116680003)
isa = rel[
    (rel["active"] == "1") &
    (rel["typeId"] == "116680003")
]

# Build parent lookup: concept_id -> set of direct parents
parents = defaultdict(set)
for _, row in isa.iterrows():
    parents[row["sourceId"]].add(row["destinationId"])

# Semantic type roots in the SNOMED IS-A hierarchy
CLINICAL_FINDING_ROOT = "404684003"  # clinical finding (includes disorders and findings)
PROCEDURE_ROOT        = "71388002"   # procedure
DISORDER_ROOT         = "64572001"   # disorder (a subtype of clinical finding)

def get_semantic_type(concept_id: str, visited=None) -> str:
    """Walk up IS-A hierarchy to find semantic type."""
    if visited is None:
        visited = set()
    if concept_id in visited:
        return "other"
    visited.add(concept_id)

    if concept_id == DISORDER_ROOT:
        return "disorder"
    if concept_id == PROCEDURE_ROOT:
        return "procedure"
    if concept_id == CLINICAL_FINDING_ROOT:
        return "finding"

    for parent in parents.get(concept_id, []):
        result = get_semantic_type(parent, visited)
        if result != "other":
            return result
    return "other"

# Build concept -> semantic type mapping
concept_ids = list(active_concepts)
print(f"Mapping semantic types for {len(concept_ids)} concepts...")

concept_to_type = {}
for cid in concept_ids:
    concept_to_type[cid] = get_semantic_type(cid)

# Keep only disorder, finding, procedure
relevant_types = {"disorder", "finding", "procedure"}
relevant_concepts = {
    cid for cid, t in concept_to_type.items() if t in relevant_types
}
print(f"Relevant concepts: {len(relevant_concepts)}")

# Filter descriptions to relevant concepts
desc_filtered = desc_nl[desc_nl["conceptId"].isin(relevant_concepts)]

# Write snomed_nl.tsv: concept_id \t term \t semantic_type
os.makedirs("data", exist_ok=True)
with open("data/snomed_nl.tsv", "w", encoding="utf-8") as f:
    for _, row in desc_filtered.iterrows():
        sem_type = concept_to_type[row["conceptId"]]
        f.write(f"{row['conceptId']}\t{row['term']}\t{sem_type}\n")

print(f"Written {len(desc_filtered)} rows to data/snomed_nl.tsv")

