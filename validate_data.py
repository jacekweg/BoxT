import argparse
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Check generated data against the original concept hierarchy.")
    parser.add_argument("--concept_relations", type=str, required=True,
                        help="Path to the original 'concept_relations.json' hierarchy.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to one of the generated data files, e.g. 'train.txt'.")
    parser.add_argument("--output_file", type=str, default="validated_concepts.json",
                        help="Where to save the re-derived concept relations with computed typicalities.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.concept_relations, "r") as f:
        original_rels = json.load(f)

    parent_map = {}
    for concept, info in original_rels.items():
        parent_map[concept] = info.get("parent", None)

    concept_to_entities = defaultdict(set)

    with open(args.data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                continue

            entity = parts[1]
            relation = parts[2]
            concept = parts[3]

            if relation != "is_a":
                continue

            concept_to_entities[concept].add(entity)

    new_relations = {}
    for concept, info in original_rels.items():
        parent = info.get("parent", None)

        new_info = dict(info)  # shallow copy of original
        if parent:
            child_entities = concept_to_entities.get(concept, set())
            parent_entities = concept_to_entities.get(parent, set())
            num_child = len(child_entities)
            if num_child == 0:
                new_info["typicality"] = 0.0
            else:
                both_count = len(child_entities.intersection(parent_entities))
                ratio = both_count / float(num_child)
                new_info["typicality"] = round(ratio, 3)
        else:
            pass

        new_relations[concept] = new_info

    with open(args.output_file, "w") as f:
        json.dump(new_relations, f, indent=2)


if __name__ == "__main__":
    main()
