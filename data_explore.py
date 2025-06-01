import msgpack
import os
import numpy as np
from utils.data_prep import create_kb_filter_multi

test_kb_name = "Very-Small-Typ"

def load_kb_dicts(kb_name, kb_multi_dir="DatasetsMulti"):
    kb_directory = os.path.join(kb_multi_dir, kb_name)

    path_to_e2id = os.path.join(kb_directory, "Ent2ID.dict")
    path_to_r2id = os.path.join(kb_directory, "Rel2ID.dict")
    path_to_c2id = os.path.join(kb_directory, "Con2ID.dict")

    with open(path_to_e2id, 'rb') as f:
        e2id_dict = msgpack.unpack(f, raw=False)
        # print(json.dumps(e2id_dict, indent=2, sort_keys=True))
    with open(path_to_r2id, 'rb') as f:
        r2id_dict = msgpack.unpack(f, raw=False)
        # print(json.dumps(r2id_dict, indent=2, sort_keys=True))
    with open(path_to_c2id, 'rb') as f:
        c2id_dict = msgpack.unpack(f, raw=False)
        # print(json.dumps(c2id_dict, indent=2, sort_keys=True))

    return e2id_dict, r2id_dict, c2id_dict

entity_id_dict, relation_id_dict, concepts_id_dict = load_kb_dicts(test_kb_name)

print("Number of entities:", len(entity_id_dict))
print("Number of relations:", len(relation_id_dict))
print("Number of concepts:", len(concepts_id_dict))

print("\nSample entities and their IDs:")
for entity, idx in list(entity_id_dict.items())[:25]:
    print(f"{entity}: {idx}")

print("\nRelations and their IDs:")
for relation, idx in relation_id_dict.items():
    print(f"{relation}: {idx}")

print("\nConcepts and their IDs:")
for concept, idx in concepts_id_dict.items():
    print(f"{concept}: {idx}")

def load_binary_kb(kb_name, split="train", kb_multi_dir="DatasetsMulti"):
    kb_directory = os.path.join(kb_multi_dir, kb_name)
    kb_file = os.path.join(kb_directory, f"{split}.kb")
    with open(kb_file, "rb") as f:
        kb_data = msgpack.unpack(f, raw=False)
    return kb_data

kb_data = load_binary_kb(test_kb_name, split="train")

print(f"\nNumber of facts in train set: {len(kb_data)}")

print("\nSample facts (ID representation)")
print("[entity, relation, concept, val]:\n")
for fact in kb_data[:5]:
    print(fact)

id_entity_dict = {idx: entity for entity, idx in entity_id_dict.items()}
id_relation_dict = {idx: relation for relation, idx in relation_id_dict.items()}
id_concept_dict = {idx: concept for concept, idx in concepts_id_dict.items()}

print("\nSample facts (mapped back to entities and relations):")
for fact in kb_data[:5]:
    entity_ids = fact[0]
    relation_id = fact[1]
    concept_id = fact[2]
    typicality_score = fact[-1] # or just fact[3] for now

    entities = id_entity_dict[entity_ids]
    relation = id_relation_dict[relation_id]
    concept = id_concept_dict[concept_id]

    print(f"Entity: {entities}")
    print(f"Relation: {relation}")
    print(f"Concept: {concept}\n")
    print(f"Typicality Score: {typicality_score}")

def load_batches(kb_name, split="train", kb_multi_dir="DatasetsMulti"):
    kb_directory = os.path.join(kb_multi_dir, kb_name)
    batch_file = os.path.join(kb_directory, f"{split}.kbb.npy")
    batches = np.load(batch_file, allow_pickle=True)
    return batches

batches = load_batches(test_kb_name, split="train")

print(f"Number of batches in train set: {len(batches)}")

print(f"Number of facts in first batch: {len(batches[0])}")

print("\nSample facts from the first batch:")
for fact in batches[0][:5]:
    print(fact)

def load_kb_metadata_multi():
    metadata_file_path = os.path.join("MetadataMulti.mpk")
    with open(metadata_file_path, 'rb') as f:
        metadata_dict = msgpack.unpack(f, raw=False)
    return metadata_dict

metadata_dict = load_kb_metadata_multi()

kb_metadata = metadata_dict.get(test_kb_name)
print(f"\nMetadata for {test_kb_name}:")
print(kb_metadata)
print("[#entities, #relations, #concepts, max arity]")

max_entity_id = max(entity_id_dict.values())
max_relation_id = max(relation_id_dict.values())

for fact in kb_data:
    entity_id = fact[0]
    relation_id = fact[1]
    concept_id = fact[2]

    if entity_id > max_entity_id:
        print(f"Invalid entity ID: {entity_id}")
    if relation_id > max_relation_id:
        print(f"Invalid relation ID: {relation_id}")
    if concept_id > max_entity_id:
        print(f"Invalid concept ID: {concept_id}")

kb_filter = create_kb_filter_multi(test_kb_name)

sample_fact = kb_data[0]
entity_id = sample_fact[0]
relation_id = sample_fact[1]
concept_id = sample_fact[2]

triple = tuple([entity_id] + [relation_id] + [concept_id])

print("\nTest triple (without typicality): ", triple)
if not kb_filter[triple]:
    print("Triple exists in the KB.")
else:
    print("Triple does not exist in the KB.")
