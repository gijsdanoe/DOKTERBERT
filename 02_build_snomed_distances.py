import os
import json
import math
import pandas as pd
import networkx as nx
from collections import defaultdict

# --- Config ---
RELATIONSHIPS_FILE = "path/to/snomed-release/Snapshot/Terminology/sct2_Relationship_Snapshot_NL1000146_20250831.txt"
CONCEPTS_FILE = "path/to/snomed-release/Snapshot/Terminology/sct2_Concept_Snapshot_NL1000146_20250831.txt"
SNOMED_TSV = "data/snomed_nl.tsv"
SIGMA = 15
CUTOFF = int(4 * SIGMA)  # 60

# Load active concepts
concepts = pd.read_csv(CONCEPTS_FILE, sep="\t", dtype=str)
active_concepts = set(concepts[concepts["active"] == "1"]["id"])

# Load active IS-A relationships
rel = pd.read_csv(RELATIONSHIPS_FILE, sep="\t", dtype=str)
isa = rel[(rel["active"] == "1") & (rel["typeId"] == "116680003")]

# Build NetworkX graph
G = nx.DiGraph()
for _, row in isa.iterrows():
    if row["sourceId"] in active_concepts and row["destinationId"] in active_concepts:
        G.add_edge(row["sourceId"], row["destinationId"])  # child -> parent

# Load relevant concepts from snomed_nl.tsv
relevant_concepts = set()
semantic_types = {}
with open(SNOMED_TSV, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            relevant_concepts.add(parts[0])
            semantic_types[parts[0]] = parts[2]

print(f"Relevant concepts: {len(relevant_concepts)}")
print(f"Nodes in graph: {G.number_of_nodes()}")

# Precompute distances for relevant concepts only
print(f"Precomputing distances (cutoff={CUTOFF})...")
distances = {}
for i, concept_id in enumerate(relevant_concepts):
    if concept_id in G:
        distances[concept_id] = nx.single_source_shortest_path_length(
            G, concept_id, cutoff=CUTOFF
        )
    else:
        distances[concept_id] = {}
    if i % 10000 == 0:
        print(f"  {i}/{len(relevant_concepts)}...")

# Weight function
def get_weight(c1, c2, beta_1=2.0):
    cross_type = semantic_types.get(c1) != semantic_types.get(c2)
    d = distances.get(c1, {}).get(c2, None)
    if d is not None:
        w = math.exp(-d / SIGMA)
    else:
        w = 1.0  # beyond cutoff
    if cross_type:
        w *= beta_1
    return w

# Save distances
os.makedirs("data", exist_ok=True)
with open("data/snomed_distances.json", "w") as f:
    json.dump({k: dict(v) for k, v in distances.items()}, f)
print("Saved to data/snomed_distances.json")