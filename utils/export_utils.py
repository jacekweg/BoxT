import numpy as np


def export_concept_relations_from_boxes(concept_bases_np, concept_deltas_np, concept_id_to_name, distance_weight=0.0):
    """
    Exports data about concept relations from box embeddings.
    """
    concept_relations_export = {}
    if concept_bases_np is None or concept_deltas_np is None or concept_id_to_name is None:
        print("Error: Missing data for export_concept_relations_from_boxes.")
        return concept_relations_export

    num_concepts = concept_bases_np.shape[0]

    def get_name(idx):
        name = concept_id_to_name.get(int(idx))
        if name is None:
            name = concept_id_to_name.get(str(idx))
        return name

    concept_deltas_abs_np = np.abs(concept_deltas_np)
    lows = (concept_bases_np - concept_deltas_abs_np / 2).astype(np.float64)
    highs = (concept_bases_np + concept_deltas_abs_np / 2).astype(np.float64)

    def area(low, high):
        lengths = high - low
        if np.any(lengths <= 0):
            return 0.0
        return np.prod(lengths)

    def overlap_area(low1, high1, low2, high2):
        overlap_low = np.maximum(low1, low2)
        overlap_high = np.minimum(high1, high2)

        for d in range(low1.shape[0]):
            if overlap_low[d] >= overlap_high[d]:
                return 0.0

        return np.prod(overlap_high - overlap_low)

    def point_to_box_distance(point, box_low, box_high):
        clamped_point = np.maximum(np.minimum(point, box_high), box_low)
        dist = np.linalg.norm(point - clamped_point)
        if np.isnan(dist):
            return np.inf
        return dist

    concept_areas = [area(lows[i], highs[i]) for i in range(num_concepts)]
    candidate_parents = [[] for _ in range(num_concepts)]

    overlaps_found = 0
    for i in range(num_concepts):
        for j in range(num_concepts):
            if i == j: continue

            ov_area = overlap_area(lows[i], highs[i], lows[j], highs[j])
            
            if ov_area <= 0: 
                continue
            
            overlaps_found += 1
            area1 = concept_areas[i]
            area2 = concept_areas[j]

            if area1 <= 0 or area2 <= 0: 
                continue

            if area1 >= area2:
                parent_idx, child_idx = i, j
                overlap_ratio_val = ov_area / area2
                is_fully_contained = np.all(lows[child_idx] >= lows[parent_idx]) and np.all(highs[child_idx] <= highs[parent_idx])
            else:
                parent_idx, child_idx = j, i
                overlap_ratio_val = ov_area / area1
                is_fully_contained = np.all(lows[child_idx] >= lows[parent_idx]) and np.all(highs[child_idx] <= highs[parent_idx])

            score = overlap_ratio_val
            if distance_weight > 0:
                child_center = concept_bases_np[child_idx]
                dist = point_to_box_distance(child_center, lows[parent_idx], highs[parent_idx])
                max_possible_dist = np.sqrt(np.sum(np.square(concept_deltas_abs_np[parent_idx])))
                dist_factor = max(0.0, min(1.0, 1.0 - dist / max_possible_dist)) if not np.isinf(dist) else 0.0
                score = (1.0 - distance_weight) * overlap_ratio_val + distance_weight * dist_factor

            if is_fully_contained:
                score = 2.0
            
            if not np.isnan(score) and score > 0:
                candidate_parents[child_idx].append((parent_idx, score))


    for i in range(num_concepts):
        concept_name = get_name(i)
        if concept_name is None:
            continue

        parents = candidate_parents[i]
        if not parents:
            concept_relations_export[concept_name] = {"parents": []}
        else:
            filtered_parents = [(p_idx, min(score, 1.0)) for p_idx, score in parents]
            filtered_parents.sort(key=lambda x: x[1], reverse=True)
            
            parent_entries = []
            for p_idx, score in filtered_parents:
                p_name = get_name(p_idx)
                if p_name:
                    parent_entries.append({
                        "parent": p_name,
                        "inclusion_degree": float(round(score, 3))
                    })
                else:
                    print(f"Warning: No name for parent concept index {p_idx} of child {concept_name}. Skipping.")
            concept_relations_export[concept_name] = {"parents": parent_entries}
    
    return concept_relations_export


def export_concept_relations_from_entities(entity_points_np, concept_bases_np, concept_deltas_np, concept_id_to_name):
    """
    Export concepts relationships from entities. (Not used in the end)
    """
    if entity_points_np is None or concept_bases_np is None or concept_deltas_np is None or concept_id_to_name is None:
        print("Error: Missing data for export_concept_relations_from_learned_entity_sets.")
        return {}

    num_entities = entity_points_np.shape[0]
    num_concepts = concept_bases_np.shape[0]

    def get_concept_name(idx):
        name = concept_id_to_name.get(int(idx))
        if name is None: name = concept_id_to_name.get(str(idx))
        return name

    concept_deltas_abs_np = np.abs(concept_deltas_np)
    concept_lows = (concept_bases_np - concept_deltas_abs_np).astype(np.float64)
    concept_highs = (concept_bases_np + concept_deltas_abs_np).astype(np.float64)

    # 1. Determine "learned" entity membership
    concepts_to_learned_entities = {i: set() for i in range(num_concepts)}
    for entity_idx in range(num_entities):
        entity_point = entity_points_np[entity_idx]
        for concept_idx in range(num_concepts):
            low = concept_lows[concept_idx]
            high = concept_highs[concept_idx]
            if np.all(entity_point >= low) and np.all(entity_point <= high):
                concepts_to_learned_entities[concept_idx].add(entity_idx)

    # 2. Calculate hierarchy based on these learned entity sets
    concept_relations_export = {}
    concept_ids_list = list(concepts_to_learned_entities.keys())  # These are integer IDs

    for concept_id in concept_ids_list:  # child concept_id
        concept_name = get_concept_name(concept_id)
        if concept_name is None:
            continue

        child_entities = concepts_to_learned_entities.get(concept_id, set())
        if not child_entities:
            concept_relations_export[concept_name] = {"parents": []}
            continue

        candidate_parents = []

        for potential_parent_id in concept_ids_list:
            if concept_id == potential_parent_id:
                continue

            potential_parent_name = get_concept_name(potential_parent_id)
            if potential_parent_name is None:
                continue

            parent_entities = concepts_to_learned_entities.get(potential_parent_id, set())

            if len(parent_entities) < len(child_entities):
                continue

            shared_entities = child_entities & parent_entities

            inclusion_degree = len(shared_entities) / len(child_entities) if len(child_entities) > 0 else 0.0

            if inclusion_degree > 0:
                candidate_parents.append((potential_parent_name, inclusion_degree, len(parent_entities)))

        candidate_parents.sort(key=lambda x: (x[1], x[2]), reverse=True)

        parent_entries = [{
            "parent": p_name,
            "inclusion_degree": float(round(degree, 3))
        } for p_name, degree, _ in candidate_parents]

        concept_relations_export[concept_name] = {"parents": parent_entries}


    return concept_relations_export
