import argparse
import os
import numpy as np
import json
import traceback
from datetime import datetime

from utils import export_utils
from report_generator import compare_concept_relations, generate_all_reports
from utils.data_prep import read_metadata_value

def main():
    parser = argparse.ArgumentParser(description="Export relations from saved BoxE data and generate reports.")
    parser.add_argument("-savedDataDir", type=str, required=True, metavar='<path>',
                        help="Path to directory with saved export data (concept_id_to_name.json, export_info.json, etc.).")
    parser.add_argument("-exportMode", choices=["geom", "entity"], required=True,
                        help="Which export function to run ('boxes' or 'entities').")
    parser.add_argument("-typThreshold", type=float, default=0.8, metavar='<float>',
                        help="Typicality threshold.")
    parser.add_argument("-distanceWeight", type=float, default=0.5, metavar='<float>',
                        help="Distance weight for 'boxes' export mode. Required if -exportMode=boxes.")
    parser.add_argument("-exportTemp", type=float, default=1.0, metavar='<float>',
                        help="Temperature for export function.")

    args = parser.parse_args()

    if args.exportMode == "boxes" and args.distanceWeight is None:
         parser.error("-distanceWeight is required when -exportMode=boxes")

    export_data_dir = args.savedDataDir

    # 1. Load Common Data & Export Info
    try:
        id_to_name_path = os.path.join(export_data_dir, "concept_id_to_name.json")
        info_path = os.path.join(export_data_dir, "export_info.json")

        if not os.path.isdir(export_data_dir):
             print(f"Saved data directory not found: {export_data_dir}")
             return
        if not all(os.path.exists(p) for p in [id_to_name_path, info_path]):
            missing_info = [p for p in [id_to_name_path, info_path] if not os.path.exists(p)]
            print(f"Not all required base info files found in {export_data_dir}. Missing: {', '.join(missing_info)}")
            return

        with open(id_to_name_path, 'r') as f:
            loaded_concept_id_to_name = json.load(f)
            concept_id_to_name = {int(k) if k.isdigit() else k: v for k,v in loaded_concept_id_to_name.items()}
        with open(info_path, 'r') as f:
            export_info = json.load(f)
        
        kb_name_from_info = export_info.get("kb_name")
        summary_dir_from_info = export_info.get("summary_dir")
        emb_dim_from_info = export_info.get("embedding_dim")
        nb_entities_from_info = export_info.get("nb_entities")
        nb_concepts_from_info = export_info.get("nb_concepts")

    except Exception as e:
        print(f"Error loading common data: {e}")
        traceback.print_exc()
        return

    # 2. Perform Export
    try:
        if args.exportMode == "geom":
            bases_path = os.path.join(export_data_dir, "concept_bases.npy")
            deltas_path = os.path.join(export_data_dir, "concept_deltas.npy")
            required_files_boxes = [bases_path, deltas_path]
            concept_bases_np = np.load(bases_path)
            concept_deltas_np = np.load(deltas_path)
            exported_relations = export_utils.export_concept_relations_from_boxes(
                concept_bases_np, 
                concept_deltas_np, 
                concept_id_to_name, 
                args.distanceWeight,
            )
        elif args.exportMode == "entity":
            bases_path = os.path.join(export_data_dir, "concept_bases.npy")
            deltas_path = os.path.join(export_data_dir, "concept_deltas.npy")
            entities_path = os.path.join(export_data_dir, "entity_points.npy")
            required_files_learned = [bases_path, deltas_path, entities_path]
            concept_bases_np = np.load(bases_path)
            concept_deltas_np = np.load(deltas_path)
            entity_points_np = np.load(entities_path)

            exported_relations = export_utils.export_concept_relations_from_entities(
                entity_points_np,
                concept_bases_np, 
                concept_deltas_np, 
                concept_id_to_name,
            )

        else:
            print(f"Unknown exportMode: {args.exportMode}")
            return
        
        if exported_relations is None:
            print("Warning: Export step did not produce relations. Reporting might be incomplete.")


    except Exception as e:
        print(f"Error during export from saved data: {e}")
        traceback.print_exc()
        return

    # 3. Prepare for Report Generation
    try:
        report_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_output_dir_name = f"report_export_{kb_name_from_info}_{report_timestamp}"
        report_output_dir = os.path.join("reports", report_output_dir_name)
        report_output_dir_json = os.path.join(report_output_dir, "json")
        
        os.makedirs(report_output_dir, exist_ok=True)
        os.makedirs(report_output_dir_json, exist_ok=True)

        exported_relations_for_report_file = os.path.join(report_output_dir_json, "exported_relations_for_report.json")
        with open(exported_relations_for_report_file, 'w') as f:
            json.dump(exported_relations, f, indent=2)

        comparison_report_file = os.path.join(report_output_dir_json, "comparison_report.json")
        original_file = os.path.join("Datasets", kb_name_from_info, "concept_relations.json")
        try:
            compare_concept_relations(exported_relations_for_report_file, original_file, comparison_report_file)
        except Exception as e:
            print(f"Error during compare_concept_relations: {e}.")
            return

        comparison_data = None
        try:
            with open(comparison_report_file, 'r') as f:
                comparison_data = json.load(f)
        except Exception as e:
            print(f"Error loading generated comparison report {comparison_report_file}: {e}")
            return

        error_rate = None
        total_examples = None
        original_dataset_metadata_path = os.path.join("Datasets", kb_name_from_info, "metadata.txt")
        if os.path.exists(original_dataset_metadata_path):
            error_rate = read_metadata_value(original_dataset_metadata_path, "error_rate")
            total_examples = read_metadata_value(original_dataset_metadata_path, "max_examples")
        else:
            print(f"Original dataset metadata not found at {original_dataset_metadata_path}.")
        
        report_metadata = {
            "Date": report_timestamp,
            "Dataset name": kb_name_from_info,
            "Dataset Size": f"Concepts: {nb_concepts_from_info}, Entities: {nb_entities_from_info}, Examples total: {total_examples}",
            "Error Rate": f"{round(float(error_rate), 2)}" if error_rate is not None else "N/A",
            "Typicality Threshold": args.typThreshold,
            "Embedding Dimension": emb_dim_from_info,
            "Export Mode Used": args.exportMode,
            **({"Distance Weight": args.distanceWeight} if args.exportMode == "geom" and args.distanceWeight is not None else {})
        }

        generate_all_reports(
            comparison_report_file,
            exported_relations,
            report_output_dir,
            summary_dir_from_info,
            report_metadata
        )
        print(f"Reports available in {report_output_dir}")

    except Exception as e:
        print(f"Error during report generation step: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 