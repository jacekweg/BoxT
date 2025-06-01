from box_t import BoxT
from report_generator import generate_all_reports, compare_concept_relations
from utils.parser_utils import setup_parser
from utils.data_prep import read_metadata_value
from utils import export_utils
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import time
import json
import shutil
import cnst


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_commandline():
    args, model_options = setup_parser()

    target_kb = args.targetKB
    feedback_period = args.printFreq
    save_period = args.savePeriod
    epoch_checkpoint = args.validCkpt
    num_epochs = args.epochs
    reset_weights = args.resetWeights
    loss_file_name = args.lossFName
    log_to_file = args.logToFile
    log_file_name = args.logFName
    sep_valid = args.separateValid
    seed = args.seed
    generate_report = args.report
    export_func = args.export

    # set_seed(seed)

    # Initialize the model
    start_time = time.time()

    model = BoxT(target_kb, model_options)
    model.train_with_valid(epoch_ckpt=epoch_checkpoint, num_epochs=num_epochs,
                           reset_weights=reset_weights, log_to_file=log_to_file,
                           log_file_name=log_file_name, save_period=save_period, separate_valid_model=sep_valid)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time", total_time, "s")

    # Run final validation on test dataset
    if generate_report and model.generate_report:
        print("Running final validation on test dataset...")
        with torch.no_grad():
            test_metrics = model.validate(dataset=cnst.TEST, log_details=False,
                                          max_facts_to_process=4,
                                          max_facts_per_position=4,
                                          batch_facts_for_corruption=32,
                                          corruption_batch_size=2000
                                          )
        
        # Log test metrics to summary writer with a special prefix
        model.summary_writer.add_scalar('TEST_Adjusted Mean Rank Index', test_metrics.adjusted_mean_rank_index, num_epochs)
        model.summary_writer.add_scalar('TEST_Rank Biased Precision', test_metrics.rank_biased_precision, num_epochs)
        model.summary_writer.add_scalar('TEST_Mean First Relevant', test_metrics.mean_first_rank, num_epochs)
        model.summary_writer.add_scalar('TEST_Hits@1', test_metrics.hits_at_values[0], num_epochs)
        model.summary_writer.add_scalar('TEST_Hits@3', test_metrics.hits_at_values[1], num_epochs)
        model.summary_writer.add_scalar('TEST_Hits@5', test_metrics.hits_at_values[2], num_epochs)
        model.summary_writer.add_scalar('TEST_Hits@10', test_metrics.hits_at_values[3], num_epochs)

    export_data = True
    if export_data:
        model.save_data_for_export(os.path.join("exported_data", f"{target_kb}_date_{model.now_str}"))

    distance_weight = None
    if generate_report:
        dir_name = f"report_{target_kb}_date_{model.now_str}"
        results_dir = os.path.join("reports", dir_name)
        json_dir = os.path.join(results_dir, "json")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        if export_func == "entity":
            relations = export_utils.export_concept_relations_from_entities(
                model.entity_points.weight.detach().cpu().numpy(),
                model.concept_bases.weight.detach().cpu().numpy(),
                F.softplus(model.concept_deltas_raw.weight).detach().cpu().numpy(),
                model.concept_id_to_name,
            )
        else:
            distance_weight = 0.5
            relations = export_utils.export_concept_relations_from_boxes(
                concept_bases_np=model.concept_bases.weight.detach().cpu().numpy(),
                concept_deltas_np=F.softplus(model.concept_deltas_raw.weight).detach().cpu().numpy(),
                concept_id_to_name=model.concept_id_to_name,
                distance_weight=distance_weight,
            )

        original_concepts_file = os.path.join("Datasets", target_kb, "concept_relations.json")
        trained_relations_file = os.path.join(json_dir, "trained_concept_relations.json")
        comparison_report_file = os.path.join(json_dir, "comparison_report.json")
        copied_concepts_file = os.path.join(json_dir, "concept_relations.json")

        shutil.copyfile(original_concepts_file, copied_concepts_file)

        with open(trained_relations_file, 'w') as f:
            json.dump(relations, f, indent=2)

        compare_concept_relations(trained_relations_file, copied_concepts_file, comparison_report_file)

        error_rate = read_metadata_value(os.path.join("Datasets", target_kb, "metadata.txt"), "error_rate")
        total_examples = read_metadata_value(os.path.join("Datasets", target_kb, "metadata.txt"), "max_examples")

        metadata = {
            "Date": model.now_str,
            "Dataset name": target_kb,
            "Dataset Size": f"Concepts: {model.nb_concepts}, Entities: {model.nb_entities}, Examples total: {total_examples}",
            "Error Rate": f"{round(float(error_rate), 2)}" if error_rate is not None else "N/A",
            "Typicality Threshold": model.typicality_threshold,
            "Embedding Dimension": model.embedding_dim,
            "Export Mode Used": export_func,
            **({"Distance Weight": distance_weight} if distance_weight is not None else {})
        }

        generate_all_reports(comparison_report_file, relations, results_dir, model.summary_dir, metadata)

if __name__ == "__main__":
    train_commandline()
