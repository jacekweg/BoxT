import argparse
import os
import random
import json
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for concept hierarchies with kernel/random modes and custom dataset splits."
    )
    parser.add_argument("--dataset_name", type=str, default="Kernel-Data",
                        help="Name of the dataset directory to save files (default: 'Kernel-Data')")
    parser.add_argument("--max_examples", type=int, default=500000,
                        help="Maximum number of examples to generate (default: 500000)")
    parser.add_argument("--mode", type=str, default="kernel", choices=["kernel", "random"],
                        help="Mode of generation: 'kernel' uses data from files, 'random' generates random data.")
    parser.add_argument("--files_dir", type=str, default="DataGen/",
                        help="Directory with data files if mode is 'kernel' (default: 'DataGen/')")
    parser.add_argument("--splits", type=str, default="0.8,0.1,0.1",
                        help="Comma-separated train,valid,test proportions (default: '0.8,0.1,0.1'). "
                             "For example '1.0,0.0,0.0' for 100% training only.")
    parser.add_argument("--num_concepts", type=int, default=10,
                        help="Number of concepts to generate in random mode (default: 10)")
    parser.add_argument("--error_rate", type=float, default=0.0,
                        help="Error rate for assigning incorrect concept (default: 0.0, i.e. all assignments are correct)")
    return parser.parse_args()


###################################
# Shared inclusion_degree
###################################

def assign_inclusion_degree(depth):
    """
    If depth=0 => random [0.0..1.0].
    If depth>0 => inclusion_degree decreases with depth => e.g. 0.8 - 0.2 * depth +/- 0.05
    """
    if depth == 0:
        return round(random.uniform(0.0, 1.0), 3)
    else:
        base_t = max(1.0 - 0.2 * depth, 0.0)
        return round(random.uniform(base_t - 0.05, base_t + 0.05), 3)


def calculate_concept_levels(concept_relations):
    """
    Calculates the depth of each concept from a root and the maximum depth.
    A root concept (parent is None) has depth 0.
    Returns a tuple: (dict mapping concept_name to depth, max_depth).
    """
    levels = {}

    max_iters = len(concept_relations) + 5 
    processed_in_iteration = True

    all_concept_names = set(concept_relations.keys())
    for c_data in concept_relations.values():
        if c_data.get("parent"):
            all_concept_names.add(c_data["parent"])

    for concept_name in all_concept_names:
        levels[concept_name] = -1

    for iter_count in range(max_iters):
        if not processed_in_iteration and all(level != -1 for level in levels.values()):
            break
        
        processed_in_iteration = False
        for concept_name in list(all_concept_names):
            if levels[concept_name] != -1:
                continue

            parent_name = concept_relations.get(concept_name, {}).get("parent")

            if parent_name is None:
                if levels[concept_name] == -1:
                    levels[concept_name] = 0
                    processed_in_iteration = True
            elif parent_name in levels and levels[parent_name] != -1:
                if levels[concept_name] == -1:
                    levels[concept_name] = levels[parent_name] + 1
                    processed_in_iteration = True
            elif parent_name not in concept_relations and parent_name not in levels:
                levels[parent_name] = 0
                if levels[concept_name] == -1:
                     levels[concept_name] = levels[parent_name] + 1
                     processed_in_iteration = True

    for _ in range(len(all_concept_names)):
        made_change_in_pass = False
        for concept_name in all_concept_names:
            if levels.get(concept_name, -1) == -1:
                parent_name = None
                if concept_name in concept_relations:
                    parent_name = concept_relations[concept_name].get("parent")
                
                if parent_name is None:
                    if levels.get(concept_name, -1) == -1:
                        levels[concept_name] = 0
                        made_change_in_pass = True
                elif levels.get(parent_name, -1) != -1:
                     if levels.get(concept_name, -1) == -1:
                        levels[concept_name] = levels[parent_name] + 1
                        made_change_in_pass = True
        if not made_change_in_pass:
            break


    for concept_name in list(levels.keys()):
        if levels[concept_name] == -1:
            levels[concept_name] = 0 

    max_depth_val = 0
    if levels:
        max_depth_val = max(levels.values()) if any(lvl != -1 for lvl in levels.values()) else 0
        if max_depth_val == -1: max_depth_val = 0

    return levels, max_depth_val


def _get_direct_children_map(concept_relations):
    """
    Builds a map from parent concept name to a list of its direct children's names.
    """
    children_map = {p: [] for p in concept_relations.keys()}
    all_parents_mentioned = set()
    for concept_data in concept_relations.values():
        if concept_data.get("parent"):
            all_parents_mentioned.add(concept_data.get("parent"))
    for p_name in all_parents_mentioned:
        if p_name not in children_map:
            children_map[p_name] = []

    for concept_name, data in concept_relations.items():
        parent_name = data.get("parent")
        if parent_name is not None and parent_name in children_map:
            children_map[parent_name].append(concept_name)
    return children_map


###################################
# Kernel Mode
###################################

# Define a constant for the adjustment strength
ADJUSTMENT_STRENGTH_FACTOR = 1.0

def load_kernel_data(files_dir):
    with open(os.path.join(files_dir, "adjectives.txt"), "r") as f:
        adjectives = [line.strip() for line in f if line.strip()]

    with open(os.path.join(files_dir, "nouns.txt"), "r") as f:
        nouns = [line.strip() for line in f if line.strip()]

    with open(os.path.join(files_dir, "noun_concept_mapping.json"), "r") as f:
        noun_concept_mapping = json.load(f)

    with open(os.path.join(files_dir, "concept_relations.json"), "r") as f:
        concept_relations = json.load(f)

    return adjectives, nouns, noun_concept_mapping, concept_relations


def get_full_concepts_kernel(base_concepts, concept_relations, max_depth=3):
    depth_map = {}

    for concept in base_concepts:
        d = 0
        curr = concept

        if curr not in depth_map or d < depth_map[curr]:
            depth_map[curr] = d

        while d < max_depth:
            parent = concept_relations.get(curr, {}).get("parent")
            if not parent:
                break

            child_inclusion_degree = concept_relations[curr].get("inclusion_degree", 1.0)
            rand = random.random()
            if rand > child_inclusion_degree:
                break

            d += 1
            if parent not in depth_map or d < depth_map[parent]:
                depth_map[parent] = d

            curr = parent

    return set(depth_map.items())

def name_kernel_entity(adjs, noun):
    """Entity name in kernel mode => "<adj1>_<adj2>_<noun>"."""
    return "_".join(adjs + [noun])


def generate_kernel_data(args):
    adjectives, nouns, noun_concept_mapping, concept_relations = load_kernel_data(args.files_dir)
    all_possible_concepts = list(concept_relations.keys())

    concept_levels, max_depth = calculate_concept_levels(concept_relations)
    direct_children_map = _get_direct_children_map(concept_relations)

    concept_to_nouns_map = {}
    for noun, concept in noun_concept_mapping.items():
        if concept not in concept_to_nouns_map:
            concept_to_nouns_map[concept] = []
        concept_to_nouns_map[concept].append(noun)

    missing_noun_concepts = []
    for concept_name in concept_relations.keys():
        if concept_name not in concept_to_nouns_map or not concept_to_nouns_map[concept_name]:
            missing_noun_concepts.append(concept_name)
    
    if missing_noun_concepts:
        error_message = "Error: The following concepts from concept_relations.json do not have any nouns assigned in noun_concept_mapping.json:\n"
        for mc in missing_noun_concepts:
            error_message += f"- {mc}\n"
        error_message += "Please ensure all concepts have at least one noun mapping and try again."
        print(error_message)
        return

    weighted_candidate_concepts = []
    for concept_name in concept_relations.keys():
        if concept_name in concept_to_nouns_map and concept_to_nouns_map[concept_name]:
            level = concept_levels.get(concept_name, max_depth)
            generality_weight = (max_depth - level) + 1

            avg_child_loss_factor = 0.0
            children = direct_children_map.get(concept_name, [])
            if children:
                total_loss = 0.0
                num_valid_children_for_loss = 0
                for child_name in children:
                    child_data = concept_relations.get(child_name)
                    if child_data:
                        id_child = child_data.get("inclusion_degree") 
                        if id_child is not None:
                            total_loss += (1.0 - id_child)
                            num_valid_children_for_loss += 1
                if num_valid_children_for_loss > 0:
                    avg_child_loss_factor = total_loss / num_valid_children_for_loss
            
            final_weight = generality_weight * (1.0 + ADJUSTMENT_STRENGTH_FACTOR * avg_child_loss_factor)
            weighted_candidate_concepts.append((concept_name, final_weight))
    
    if not weighted_candidate_concepts:
        print("Error: No concepts in concept_relations.json have corresponding nouns in noun_concept_mapping.json. Cannot generate kernel data.")
        return

    candidate_concepts_for_choice = [wc[0] for wc in weighted_candidate_concepts]
    candidate_concept_weights = [wc[1] for wc in weighted_candidate_concepts]

    synthetic_data = []
    used_entities = set()
    used_entity_concept_pairs = set()
    total_entities = 0

    print("Generating synthetic data (kernel mode)...")

    while len(synthetic_data) < args.max_examples:
        # Step 1: Choose a base concept using weighted random choice from candidates
        chosen_base_concept = random.choices(candidate_concepts_for_choice, weights=candidate_concept_weights, k=1)[0]

        # Step 2: Choose a noun that maps to this chosen_base_concept
        available_nouns = concept_to_nouns_map[chosen_base_concept]
        noun = random.choice(available_nouns)
        
        base_concepts = [chosen_base_concept]

        # pick 1..3 adjectives
        num_adjs = random.randint(1, 3)
        chosen_adjs = random.sample(adjectives, num_adjs)

        entity_name = name_kernel_entity(chosen_adjs, noun)

        facts_to_generate_for_entity = [(chosen_base_concept, 0)]
        current_concept = chosen_base_concept
        depth = 1
        while True:
            parent = concept_relations[current_concept].get("parent")
            inclusion_degree = concept_relations[current_concept].get("inclusion_degree")
            if parent is None or inclusion_degree is None:
                break
            if random.random() < inclusion_degree:
                facts_to_generate_for_entity.append((parent, depth))
                current_concept = parent
                depth += 1
            else:
                break

        for (concept_name_selected, depth_from_base) in facts_to_generate_for_entity:
            actual_concept_to_link = concept_name_selected
            if random.random() < args.error_rate:
                incorrect_candidates = [c for c in all_possible_concepts if c != concept_name_selected]
                if incorrect_candidates:
                    actual_concept_to_link = random.choice(incorrect_candidates)

            pair = (entity_name, actual_concept_to_link)
            if pair in used_entity_concept_pairs:
                continue
            used_entity_concept_pairs.add(pair)

            tscore = assign_inclusion_degree(depth_from_base)
            line = f"{tscore}\t{entity_name}\tis_a\t{actual_concept_to_link}"
            synthetic_data.append(line)

            if len(synthetic_data) >= args.max_examples:
                break

        if len(synthetic_data) >= args.max_examples:
            break

    random.shuffle(synthetic_data)

    finalize_and_save(args, synthetic_data, total_entities)


###################################
# Random Mode
###################################

def generate_random_concepts(num_concepts=10):
    c_rels = {}
    for i in range(num_concepts):
        cname = f"c_{i}"
        parent = None
        if i > 0 and random.random() < 0.5:
            pidx = random.randint(0, i - 1)
            parent = f"c_{pidx}"
        typ = None
        if parent:
            typ = round(random.uniform(0.1, 1.0), 2)
        c_rels[cname] = {
            "parent": parent,
            "inclusion_degree": typ
        }
    return c_rels


def get_full_concepts_random(base_concepts, concept_relations, max_depth=3):
    results = set()
    for c in base_concepts:
        d = 0
        cur = c
        while cur and d < max_depth:
            results.add((cur, d))
            par = concept_relations[cur]["parent"]
            cur = par
            d += 1
    return results


def name_random_entity(all_concepts_in_hierarchy, entity_id):
    """
    Random mode => c_<sorted_concept_suffices_of_full_hierarchy>_e<entity_id>
    The "noun" for mapping will be e<entity_id>.
    """
    def parse_suffix(x):
        parts = x.split('_', 1)
        return parts[1] if len(parts) == 2 else x # Get the numeric part of c_ID

    unique_concept_suffixes = sorted(list(set(parse_suffix(c) for c in all_concepts_in_hierarchy)))
    concept_part = "_" + "_".join(unique_concept_suffixes) if unique_concept_suffixes else ""
    
    return f"c{concept_part}_e{entity_id}"


def assign_inclusion_degree_random(cname, depth, concept_relations):
    base_typ = concept_relations[cname]["inclusion_degree"]
    if base_typ is None:
        return round(random.uniform(0.1, 0.3), 3)
    shift = depth * 0.1
    final_t = base_typ - shift
    final_t = max(0.1, min(1.0, final_t))
    return round(random.uniform(final_t - 0.05, final_t + 0.05), 3)


def generate_random_data(args):
    print("Generating random data (random mode).")
    concept_relations = generate_random_concepts(args.num_concepts)
    all_possible_concepts = list(concept_relations.keys())

    concept_levels, max_depth = calculate_concept_levels(concept_relations)
    direct_children_map = _get_direct_children_map(concept_relations)

    weighted_concepts = []
    for concept_name in concept_relations.keys():
        level = concept_levels.get(concept_name, max_depth)
        generality_weight = (max_depth - level) + 1

        avg_child_loss_factor = 0.0
        children = direct_children_map.get(concept_name, [])
        if children:
            total_loss = 0.0
            num_valid_children_for_loss = 0
            for child_name in children:
                child_data = concept_relations.get(child_name)
                if child_data:
                    id_child = child_data.get("inclusion_degree")
                    if id_child is not None:
                        total_loss += (1.0 - id_child)
                        num_valid_children_for_loss += 1
            if num_valid_children_for_loss > 0:
                avg_child_loss_factor = total_loss / num_valid_children_for_loss

        final_weight = generality_weight * (1.0 + ADJUSTMENT_STRENGTH_FACTOR * avg_child_loss_factor)
        weighted_concepts.append((concept_name, final_weight))

    concepts_for_choice = [wc[0] for wc in weighted_concepts]
    concept_weights = [wc[1] for wc in weighted_concepts]
    if not concepts_for_choice:
        print("Error: No concepts available for random generation. Exiting.")
        return

    synthetic_data = []
    used_entities = set()
    used_pairs = set()
    total_entities = 0
    entity_id = 0
    noun_concept_map_for_random_mode = {}

    while len(synthetic_data) < args.max_examples:
        if not concepts_for_choice:
            print("Warning: No concepts available for weighted choice. Falling back to random.choice or skipping.")
            if not list(concept_relations.keys()):
                break
            chosen_base_concept_name = random.choice(list(concept_relations.keys()))
        else:
            chosen_base_concept_name = random.choices(concepts_for_choice, weights=concept_weights, k=1)[0]
        
        base_concepts_for_hierarchy_walk = [chosen_base_concept_name]
        
        cdepths = get_full_concepts_random(base_concepts_for_hierarchy_walk, concept_relations)
        if not cdepths:
            continue

        entity_id += 1
        entity_noun_part = f"e{entity_id}"

        all_concept_names_in_hierarchy = [cd[0] for cd in cdepths]
        entity_name = name_random_entity(all_concept_names_in_hierarchy, entity_id)

        if entity_name in used_entities:
            entity_id -=1 # Roll back ID if entity name collision
            continue
        used_entities.add(entity_name)
        total_entities += 1

        noun_concept_map_for_random_mode[entity_noun_part] = chosen_base_concept_name

        facts_to_generate_for_entity = [(chosen_base_concept_name, 0)]
        current_concept = chosen_base_concept_name
        depth = 1
        while True:
            parent = concept_relations[current_concept].get("parent")
            inclusion_degree = concept_relations[current_concept].get("inclusion_degree")
            if parent is None or inclusion_degree is None:
                break
            if random.random() < inclusion_degree:
                facts_to_generate_for_entity.append((parent, depth))
                current_concept = parent
                depth += 1
            else:
                break

        unique_facts_to_generate = set(facts_to_generate_for_entity)

        for (concept_name_selected, depth_from_base) in unique_facts_to_generate:
            actual_concept_to_link = concept_name_selected
            if random.random() < args.error_rate:
                incorrect_candidates = [cc for cc in all_possible_concepts if cc != concept_name_selected]
                if incorrect_candidates:
                    actual_concept_to_link = random.choice(incorrect_candidates)

            pair = (entity_name, actual_concept_to_link)
            if pair in used_pairs:
                continue
            used_pairs.add(pair)

            tscore = assign_inclusion_degree_random(concept_name_selected, depth_from_base, concept_relations)
            line = f"{tscore}\t{entity_name}\tis_a\t{actual_concept_to_link}"
            synthetic_data.append(line)

            if len(synthetic_data) >= args.max_examples:
                break

        if len(synthetic_data) >= args.max_examples:
            break

        if len(synthetic_data) % 100000 == 0 and len(synthetic_data) > 0:
            print(f"Generated {len(synthetic_data)} examples...")

    random.shuffle(synthetic_data)

    dataset_dir = os.path.join("Datasets", args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "concept_relations.json"), "w") as f:
        json.dump(concept_relations, f, indent=2)

    with open(os.path.join(dataset_dir, "noun_concept_mapping.json"), "w") as f:
        json.dump(noun_concept_map_for_random_mode, f, indent=2)

    finalize_and_save(args, synthetic_data, total_entities)


###################################
# Finalize
###################################

def finalize_and_save(args, synthetic_data, total_entities):

    train_prop, valid_prop, test_prop = parse_splits(args.splits)
    total_examples = len(synthetic_data)

    train_end = int(train_prop * total_examples)
    valid_end = train_end + int(valid_prop * total_examples)
    test_end = total_examples

    train_data = synthetic_data[:train_end]
    valid_data = synthetic_data[train_end:valid_end]
    test_data = synthetic_data[valid_end:test_end]

    dataset_dir = os.path.join("Datasets", args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    def save_data(data_list, filename):
        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, 'w') as f:
            for data in data_list:
                f.write(data + '\n')

    if train_data:
        save_data(train_data, 'train.txt')
    else:
        with open(os.path.join(dataset_dir, 'train.txt'), 'w') as f:
            pass

    if valid_data:
        save_data(valid_data, 'valid.txt')
    else:
        with open(os.path.join(dataset_dir, 'valid.txt'), 'w') as f:
            pass

    if test_data:
        save_data(test_data, 'test.txt')
    else:
        with open(os.path.join(dataset_dir, 'test.txt'), 'w') as f:
            pass

    if args.mode == "kernel":
        src_concept_relations = os.path.join(args.files_dir, "concept_relations.json")
        dest_concept_relations = os.path.join(dataset_dir, "concept_relations.json")
        if os.path.exists(src_concept_relations):
            shutil.copyfile(src_concept_relations, dest_concept_relations)
        else:
            print(f"Warning: concept_relations.json not found in {args.files_dir} for kernel mode.")

        src_noun_mapping = os.path.join(args.files_dir, "noun_concept_mapping.json")
        dest_noun_mapping = os.path.join(dataset_dir, "noun_concept_mapping.json")
        if os.path.exists(src_noun_mapping):
            shutil.copyfile(src_noun_mapping, dest_noun_mapping)
        else:
            print(f"Warning: noun_concept_mapping.json not found in {args.files_dir} for kernel mode.")

    common_args = ["dataset_name", "max_examples", "mode", "splits", "error_rate"]
    kernel_args = ["files_dir"]
    random_args = ["num_concepts"]

    metadata_path = os.path.join(dataset_dir, "metadata.txt")
    with open(metadata_path, "w") as meta_file:
        if args.mode == "kernel":
            include_args = common_args + kernel_args
        else:
            include_args = common_args + random_args

        for arg in include_args:
            value = getattr(args, arg, None)
            meta_file.write(f"{arg}: {value}\n")

    print(f"Data saved in {dataset_dir}")
    print(f"Total examples: {total_examples}")
    print(f"Total unique entities: {total_entities}")
    print(f"Train/Valid/Test: {len(train_data)}/{len(valid_data)}/{len(test_data)}")


def parse_splits(splits_str):
    """
    Parse the user-defined splits, e.g. "0.8,0.1,0.1".
    Return (train_prop, valid_prop, test_prop).
    """
    parts = splits_str.split(',')
    props = [float(x) for x in parts]
    if len(props) == 1:
        train_prop = props[0]
        valid_prop = 0.0
        test_prop = 1.0 - train_prop
        if test_prop < 0:
            test_prop = 0.0
        return (train_prop, valid_prop, test_prop)
    elif len(props) == 2:
        train_prop = props[0]
        valid_prop = props[1]
        sum_two = train_prop + valid_prop
        test_prop = max(0.0, 1.0 - sum_two)
        return (train_prop, valid_prop, test_prop)
    elif len(props) == 3:
        t, v, te = props
        sum_ = t + v + te
        if sum_ < 1.0:
            te += (1.0 - sum_)
        elif sum_ > 1.0:
            t /= sum_
            v /= sum_
            te /= sum_
        return (t, v, te)
    else:
        raise ValueError("Splits must be 1, 2, or 3 values, e.g. '0.8,0.2' or '0.8,0.1,0.1' or '1.0' etc.")


def main():
    args = parse_args()
    if args.mode == "kernel":
        generate_kernel_data(args)
    else:
        generate_random_data(args)


if __name__ == "__main__":
    main()
