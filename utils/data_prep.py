import msgpack
import msgpack_numpy as m
import numpy as np
import os
import torch
import traceback
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cnst
m.patch()

TYPICALITY_PADDING = 1

def read_metadata_value(metadata_path, key):
    try:
        with open(metadata_path, "r") as f:
            for line in f:
                if line.strip().startswith(f"{key}:"):
                    return line.strip().split(":", 1)[1].strip()
    except FileNotFoundError:
        print(f"File not found: {metadata_path}")
    except Exception as e:
        print(f"Error reading metadata: {e}")

    return None


class MissingDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()

def build_entity_to_concepts_mapping(kb_name, file_to_read="train.txt"):
    file_path = os.path.join(cnst.DEFAULT_KB_DIR, kb_name, file_to_read)
    entity_to_concepts = {}

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            entity = parts[1]
            concept = parts[3]

            if entity not in entity_to_concepts:
                entity_to_concepts[entity] = set()
            entity_to_concepts[entity].add(concept)

    return entity_to_concepts


def build_concept_to_entities_mapping(kb_name, file_to_read="train.txt"):
    file_path = os.path.join(cnst.DEFAULT_KB_DIR, kb_name, file_to_read)
    concept_to_entities = {}

    tr_np_arr = load_kb_file(cnst.DEFAULT_KB_MULTI_DIR + str(kb_name) + "/train" + cnst.KB_FORMAT)
    print("tr_np_arr:", tr_np_arr)

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            entity = parts[1]
            concept = parts[3]

            if concept not in concept_to_entities:
                concept_to_entities[concept] = set()
            concept_to_entities[concept].add(entity)

    return concept_to_entities

def build_concept_to_entities_mapping_from_ids(kb_name, subset="train"):
    tr_np_arr = load_kb_file(cnst.DEFAULT_KB_MULTI_DIR + str(kb_name) + "/" + subset + cnst.KB_FORMAT)

    concept_to_entities = {}

    for row in tr_np_arr:
        entity_id = int(row[0])
        concept_id = int(row[2])

        if concept_id not in concept_to_entities:
            concept_to_entities[concept_id] = set()
        concept_to_entities[concept_id].add(entity_id)

    return concept_to_entities

def prepare_eval_dataset(ds_np_arr, nb_ent, max_ar):
    nb_atoms = ds_np_arr.shape[0]
    ds_np_arr_extended = np.tile(ds_np_arr, (nb_ent * max_ar, 1))
    for i in range(max_ar):
        pad_idx = ds_np_arr_extended[nb_ent * nb_atoms * i: nb_ent * nb_atoms * (i + 1), 1 + i] == nb_ent
        replacement = np.repeat(np.arange(nb_ent), nb_atoms)
        replacement[pad_idx] = nb_ent
        ds_np_arr_extended[nb_ent * nb_atoms * i: nb_ent * nb_atoms * (i + 1), 1 + i] = replacement
    final_ds = ds_np_arr_extended
    return final_ds


def parse_kb_fact(fact):
    components = fact.strip().split("\t")
    return components


def get_value(components, padding):
    if padding > 0:
        return int(float(components[0]) * 1000)
    if len(components) >= 4:
        val = int(components[3])
    else:
        val = 1
    return val


def parse_kb_fact_htr(fact, padding=0):
    components = fact.strip().split("\t")
    entity, concept, relation = components[padding + 0], components[padding + 1], components[padding + 2]
    val = get_value(components, padding)
    return [entity, relation, concept, val]


def parse_kb_fact_hrt(fact, padding=0):
    components = fact.strip().split("\t")
    entity, relation, concept = components[padding + 0], components[padding + 1], components[padding + 2]
    val = get_value(components, padding)
    return [entity, relation, concept, val]


def compute_kb_id_mapping_adapter(kb_directory=cnst.DEFAULT_KB_DIR, kb_multi_directory=cnst.DEFAULT_KB_MULTI_DIR,
                                  file_to_read="train.txt", use_eval_data=False):
    path_to_e2id = os.path.join(kb_multi_directory, cnst.ENT_2_ID_DICT_NAME)
    path_to_c2id = os.path.join(kb_multi_directory, cnst.CON_2_ID_DICT_NAME)
    path_to_r2id = os.path.join(kb_multi_directory, cnst.REL_2_ID_DICT_NAME)
    kb_name = os.path.basename(os.path.normpath(kb_directory))

    if not os.path.exists(kb_multi_directory):
        os.makedirs(kb_multi_directory)

    if use_eval_data:
        files = ["train.txt", "test.txt", "valid.txt"]
    else:
        files = [file_to_read]

    entities = []
    relations = []
    concepts = []

    for file in files:
        path_to_file = os.path.join(kb_directory, file)
        with open(path_to_file, "r", encoding="utf8") as kb:
            if kb_name in cnst.HTR_KBs:
                parsing_function = parse_kb_fact_htr
            else:
                parsing_function = parse_kb_fact_hrt
            for fact in kb:
                components = parsing_function(fact, TYPICALITY_PADDING)
                entities.append(components[0])
                relations.append(components[1])
                concepts.append(components[2])

    entities_distinct = sorted(list(set(entities)))
    relations_distinct = sorted(list(set(relations)))
    concepts_distinct = sorted(list(set(concepts)))

    ent_id_dict = {entity: index for index, entity in enumerate(entities_distinct)}
    rel_id_dict = {relation: index for index, relation in enumerate(relations_distinct)}
    con_id_dict = {concept: index for index, concept in enumerate(concepts_distinct)}

    with open(path_to_e2id, 'wb') as f:
        msgpack.pack(ent_id_dict, f)
    with open(path_to_c2id, 'wb') as f:
        msgpack.pack(con_id_dict, f)
    with open(path_to_r2id, 'wb') as f:
        msgpack.pack(rel_id_dict, f)


def compute_kb_id_mapping(kb_directory=cnst.DEFAULT_KB_MULTI_DIR, file_to_read="train.txt",
                          use_eval_data=False):
    path_to_e2id = os.path.join(kb_directory, cnst.ENT_2_ID_DICT_NAME)
    path_to_c2id = os.path.join(kb_directory, cnst.CON_2_ID_DICT_NAME)
    path_to_r2id = os.path.join(kb_directory, cnst.REL_2_ID_DICT_NAME)

    if use_eval_data:
        files = ["train.txt", "test.txt", "valid.txt"]
    else:
        files = [file_to_read]

    entities = []
    concepts = []
    relations = []

    for file in files:
        path_to_file = os.path.join(kb_directory, file)
        with open(path_to_file, "r") as kb:
            for fact in kb:
                components = parse_kb_fact(fact)
                entities.append(components[0])
                relations.append(components[1])
                concepts.append(components[2])

    entities_distinct = list(set(entities))
    relations_distinct = list(set(relations))
    concepts_distinct = list(set(concepts))

    ent_id_dict = {entity: index for index, entity in enumerate(entities_distinct)}
    con_id_dict = {concept: index for index, concept in enumerate(concepts_distinct)}
    rel_id_dict = {relation: index for index, relation in enumerate(relations_distinct)}

    with open(path_to_e2id, 'wb') as f:
        msgpack.pack(ent_id_dict, f)
    with open(path_to_c2id, 'wb') as f:
        msgpack.pack(con_id_dict, f)
    with open(path_to_r2id, 'wb') as f:
        msgpack.pack(rel_id_dict, f)


def load_kb_file(kb_file_path):
    with open(kb_file_path, "rb") as f:
        kb_file = msgpack.unpack(f, raw=False)
    np_kb_file = np.array(kb_file, dtype=np.int32)
    return np_kb_file


def load_kb_metadata_multi(kb_name):
    with open(cnst.KB_META_MULTI_FILE_NAME, 'rb') as f:
        metadata_dict = msgpack.unpack(f, raw=False)
        try:
            return metadata_dict[kb_name]
        except KeyError:
            print("No KB named " + str(kb_name) + " in the default KB folder during metadata extraction")
            return


def adapt_kbs_binary(kb_directory=cnst.DEFAULT_KB_DIR, kb_multi_directory=cnst.DEFAULT_KB_MULTI_DIR,
                     tr_batch_size=1024, tst_batch_size=1024, verbose=False, use_eval_data=False):
    knowledge_bases = [f for f in os.listdir(kb_directory) if not f.startswith('.')]
    for kb in knowledge_bases:
        if verbose:
            print("Processing KB: " + str(kb))
        individual_kb_path = os.path.join(kb_directory, kb)
        destination_kb_path = os.path.join(kb_multi_directory, kb)
        compute_kb_id_mapping_adapter(individual_kb_path, kb_multi_directory=destination_kb_path,
                                      use_eval_data=use_eval_data)
        kb_files = ["train.txt", "test.txt", "valid.txt"]
        for index, kb_file in enumerate(kb_files):
            try:
                convert_kb_to_id_rep_multi_adapter(individual_kb_path, file_to_read=kb_file)
                kb_file_no_ext = os.path.splitext(kb_file)[0]
                if index > 0:
                    convert_id_representation_to_batches(destination_kb_path,
                                                         file_to_convert=kb_file_no_ext + cnst.KB_FORMAT,
                                                         batch_size=tst_batch_size)
                else:
                    convert_id_representation_to_batches(destination_kb_path,
                                                         file_to_convert=kb_file_no_ext + cnst.KB_FORMAT,
                                                         batch_size=tr_batch_size)

            except KeyError:
                print("Error Converting file " + str(kb_file) + ": Contains Entities / Relations Outside Training Set")
                print(traceback.format_exc())

    compute_kb_metadata(kb_directory)


def convert_id_representation_to_batches(kb_directory, batch_size=15000,
                                         file_to_convert="train" + cnst.KB_FORMAT,
                                         random_seed=cnst.DEFAULT_RANDOM_SEED):
    path_to_kb_file = os.path.join(kb_directory, file_to_convert)
    path_to_kb_batch_file = os.path.join(kb_directory, os.path.splitext(file_to_convert)[0] + cnst.KBB_FORMAT)
    with open(path_to_kb_file, "rb") as f:
        kb_file = msgpack.unpack(f, raw=False)
    np_kb_file = np.array(kb_file, dtype=np.int32)
    if np_kb_file.shape[0] == 0:
        print(f"Skipping batch conversion for empty file: {file_to_convert}")
        return []
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(np_kb_file)
    number_of_splits = int(np.ceil(np_kb_file.shape[0] / batch_size))
    batches = np.array_split(np_kb_file, number_of_splits)
    separated_batches = []
    for batch in batches:
        separated_batches.append(batch)
    separated_batches = np.array(separated_batches, dtype=object)
    np.save(path_to_kb_batch_file, separated_batches, allow_pickle=True)
    return separated_batches


def compute_kb_metadata(kb_directory=cnst.DEFAULT_KB_MULTI_DIR):
    knowledge_bases = [f for f in os.listdir(kb_directory) if not f.startswith('.')]
    metadata_dict = {}
    for kb in knowledge_bases:
        e2id_dict, r2id_dict, c2id_dict = load_kb_dicts(kb)
        metadata_dict[kb] = [len(e2id_dict), len(r2id_dict), len(c2id_dict), 2]
    with open(cnst.KB_META_MULTI_FILE_NAME, "wb") as f:
        msgpack.pack(metadata_dict, f)


def compute_all_kb_id_mappings(kb_directory=cnst.DEFAULT_KB_DIR):
    knowledge_bases = [f for f in os.listdir(kb_directory) if not f.startswith('.')]
    for kb in knowledge_bases:
        compute_kb_id_mapping(os.path.join(kb_directory, kb))


def load_kb_dicts(kb_name):
    kb_directory = os.path.join(cnst.DEFAULT_KB_MULTI_DIR, kb_name)

    path_to_e2id = os.path.join(kb_directory, cnst.ENT_2_ID_DICT_NAME)
    path_to_c2id = os.path.join(kb_directory, cnst.CON_2_ID_DICT_NAME)
    path_to_r2id = os.path.join(kb_directory, cnst.REL_2_ID_DICT_NAME)

    with open(path_to_e2id, 'rb') as f:
        e2id_dict = msgpack.unpack(f, raw=False)
    with open(path_to_r2id, 'rb') as f:
        r2id_dict = msgpack.unpack(f, raw=False)
    with open(path_to_c2id, 'rb') as f:
        c2id_dict = msgpack.unpack(f, raw=False)

    return e2id_dict, r2id_dict, c2id_dict


def create_kb_filter_torch(kb_name):
    path_to_kb = os.path.join(cnst.DEFAULT_KB_MULTI_DIR, kb_name)
    path_to_kb_train = os.path.join(path_to_kb, "train" + cnst.KB_FORMAT)
    path_to_kb_valid = os.path.join(path_to_kb, "valid" + cnst.KB_FORMAT)
    path_to_kb_test = os.path.join(path_to_kb, "test" + cnst.KB_FORMAT)
    with open(path_to_kb_train, "rb") as f:
        train_facts = np.array(msgpack.unpack(f, raw=False))[:, :-1]
    with open(path_to_kb_valid, "rb") as f:
        valid_facts = np.array(msgpack.unpack(f, raw=False))[:, :-1]
    with open(path_to_kb_test, "rb") as f:
        test_facts = np.array(msgpack.unpack(f, raw=False))[:, :-1]
    all_facts = np.concatenate([train_facts, valid_facts, test_facts], axis=0)
    # Convert to PyTorch tensor
    all_facts_tensor = torch.tensor(all_facts, dtype=torch.long)
    return all_facts_tensor


def create_kb_filter_multi(kb_name):
    path_to_kb = os.path.join(cnst.DEFAULT_KB_MULTI_DIR, kb_name)
    kb_triple_existence_dict = MissingDict(lambda: True)
    path_to_kb_train = os.path.join(path_to_kb, "train" + cnst.KB_FORMAT)
    path_to_kb_valid = os.path.join(path_to_kb, "valid" + cnst.KB_FORMAT)
    path_to_kb_test = os.path.join(path_to_kb, "test" + cnst.KB_FORMAT)

    with open(path_to_kb_train, "rb") as f:
        train_triples = msgpack.unpack(f, raw=False)
    for triple in train_triples:
        kb_triple_existence_dict[(*triple[:-1],)] = False

    with open(path_to_kb_valid, "rb") as f:
        valid_triples = msgpack.unpack(f, raw=False)
    for triple in valid_triples:
        kb_triple_existence_dict[(*triple[:-1],)] = False

    with open(path_to_kb_test, "rb") as f:
        test_triples = msgpack.unpack(f, raw=False)
    for triple in test_triples:
        kb_triple_existence_dict[(*triple[:-1],)] = False

    return kb_triple_existence_dict


def convert_kb_to_id_rep_multi_adapter(kb_directory, kb_multi_directory=cnst.DEFAULT_KB_MULTI_DIR,
                                       file_to_read="train.txt"):
    path_to_file = os.path.join(kb_directory, file_to_read)
    kb_name = os.path.basename(os.path.normpath(kb_directory))
    if not os.path.exists(kb_multi_directory):
        os.mkdir(kb_multi_directory)

    path_to_kb_idrep = os.path.join(os.path.join(kb_multi_directory, kb_name),
                                    os.path.splitext(file_to_read)[0] + cnst.KB_FORMAT)

    e2id_dict, r2id_dict, c2id_dict = load_kb_dicts(kb_name)

    with open(path_to_file, "r", encoding="utf8") as kb:
        facts = []
        for fact in kb:
            if kb_name in cnst.HTR_KBs:
                parsing_function = parse_kb_fact_htr
            else:
                parsing_function = parse_kb_fact_hrt
            fact_cmpnts = parsing_function(fact, TYPICALITY_PADDING)
            fact_comp_ids = [e2id_dict[cmpnt] if idx == 0
                             else r2id_dict[cmpnt] if idx == 1
                             else c2id_dict[cmpnt] if idx == 2
                             else cmpnt
                             for idx, cmpnt in enumerate(fact_cmpnts)]
            facts.append(fact_comp_ids)
    with open(path_to_kb_idrep, 'wb') as f:
        msgpack.pack(facts, f)


def convert_kb_to_id_rep_multi(kb_directory, file_to_read="train.txt"):
    path_to_file = os.path.join(kb_directory, file_to_read)
    path_to_kb_idrep = os.path.join(kb_directory, os.path.splitext(file_to_read)[0] + cnst.KB_FORMAT)
    kb_name = os.path.basename(os.path.normpath(kb_directory))

    e2id_dict, r2id_dict, c2id_dict = load_kb_dicts(kb_name)

    with open(path_to_file, "r") as kb:
        facts = []
        for fact in kb:
            fact_cmpnts = parse_kb_fact(fact)
            fact_comp_ids = [e2id_dict[cmpnt] if idx == 0
                             else r2id_dict[cmpnt] if idx == 1
                             else c2id_dict[cmpnt] if idx == 2
                             else cmpnt
                             for idx, cmpnt in enumerate(fact_cmpnts)]
            facts.append(fact_comp_ids)
    with open(path_to_kb_idrep, 'wb') as f:
        msgpack.pack(facts, f)


def compute_statistics(kb_name, file_to_read="train" + cnst.KB_FORMAT):
    nb_concepts = load_kb_metadata_multi(kb_name)[2]
    kb_directory = os.path.join(cnst.DEFAULT_KB_MULTI_DIR, kb_name)
    path_to_file = os.path.join(kb_directory, file_to_read)
    with open(path_to_file, "rb") as f:
        train_triples = msgpack.unpack(f, raw=False)
    stats = np.array([0] * nb_concepts, dtype=np.float32)
    for fact in train_triples:
        stats[fact[0]] += 1
    normalised_stats = np.expand_dims(stats / sum(stats) * nb_concepts, axis=-1)
    return normalised_stats


if __name__ == "__main__":
    adapt_kbs_binary(verbose=True, use_eval_data=True, tr_batch_size=15000, tst_batch_size=15000)
