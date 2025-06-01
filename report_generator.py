import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import base64
from pathlib import Path

from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def compare_concept_relations(trained_file, original_file, output_file):
    with open(trained_file, "r") as f:
        trained_relations = json.load(f)

    with open(original_file, "r") as f:
        original_relations = json.load(f)

    comparison_report = {
        "missing_in_trained": {},
        "new_in_trained": {},
        "discrepancies": {}
    }

    all_concepts = set(trained_relations.keys()).union(set(original_relations.keys()))

    for concept in all_concepts:

        trained_parents = trained_relations.get(concept, {"parents": []})["parents"]
        trained_parent_dict = {p["parent"]: p["inclusion_degree"] for p in trained_parents}

        original_parent = original_relations.get(concept, {}).get("parent", None)
        original_inclusion_degree = original_relations.get(concept, {}).get("inclusion_degree", None)

        if trained_parents:
            new_in_trained = []
            new_parents_data = {}

            for parent, inclusion_degree in trained_parent_dict.items():
                if parent != original_parent:
                    new_in_trained.append(parent)
                    if parent not in new_parents_data or inclusion_degree > new_parents_data[parent]:
                        new_parents_data[parent] = inclusion_degree

            if new_in_trained:
                comparison_report["new_in_trained"][concept] = {
                    "parents": new_in_trained,
                    "inclusion_degrees": new_parents_data
                }

        if original_parent in trained_parent_dict:
            trained_inclusion_degree = trained_parent_dict[original_parent]
            if original_inclusion_degree is not None:
                if abs(trained_inclusion_degree - original_inclusion_degree) > 0.01:
                    comparison_report["discrepancies"][concept] = {
                        "parent": original_parent,
                        "trained_inclusion_degree": round(trained_inclusion_degree, 3),
                        "original_inclusion_degree": round(original_inclusion_degree, 3)
                    }
        else:
            if original_parent:
                comparison_report["missing_in_trained"][concept] = {
                    "parents": [original_parent],
                    "inclusion_degree": original_inclusion_degree
                }

    with open(output_file, "w") as f:
        json.dump(comparison_report, f, indent=4)


def load_data(file_path):
    def sort_recursively(item):
        if isinstance(item, dict):
            return {k: sort_recursively(v) for k, v in sorted(item.items())}
        elif isinstance(item, list):
            return [sort_recursively(elem) for elem in item]
        else:
            return item

    with open(file_path, 'r') as f:
        data = json.load(f)
    return sort_recursively(data)


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def scatter_plot(discrepancies, output_path):
    trained, original, labels = [], [], []
    for concept, details in discrepancies.items():
        parent = details.get("parent", "unknown")
        trained.append(details['trained_inclusion_degree'])
        original.append(details['original_inclusion_degree'])
        labels.append(f"{concept} ⊑ {parent}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(original, trained, s=100, color='blue', alpha=0.7)
    ax.set(xlabel="Original Inclusion Degree", ylabel="Learned Typicality Score",
           title="Learned Typicality vs Original Inclusion", xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], color='red', linestyle='--')
    for i, label in enumerate(labels):
        ax.annotate(label, (original[i], trained[i]), textcoords="offset points", xytext=(5, 5))
    ax.grid(True)
    save_plot(fig, output_path)


def bar_plot(discrepancies, output_path):
    labels = [f"{c} ⊑ {discrepancies[c]['parent']}" for c in discrepancies]
    trained = [discrepancies[c]['trained_inclusion_degree'] for c in discrepancies]
    original = [discrepancies[c]['original_inclusion_degree'] for c in discrepancies]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.175, trained, 0.35, label='Trained')
    ax.bar(x + 0.175, original, 0.35, label='Original')
    ax.set(ylabel="Inclusion / Typicality Score",
           title="Original Inclusion Degree & Trained Typicality Scores Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    save_plot(fig, output_path)


def difference_barplot(discrepancies, output_path):
    sorted_items = sorted(discrepancies.items(),
                          key=lambda x: abs(x[1]['trained_inclusion_degree'] - x[1]['original_inclusion_degree']),
                          reverse=True)
    labels = [f"{c} ⊑ {details['parent']}" for c, details in sorted_items]
    differences = [abs(details['trained_inclusion_degree'] - details['original_inclusion_degree']) for _, details in
                   sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, differences, color='orange')
    ax.set(ylabel="Absolute Difference", title="Absolute Differences (Sorted)")
    ax.tick_params(axis='x', rotation=45)
    save_plot(fig, output_path)


def embed_image(image_path):
    """
    Reads an image from the provided path and returns an HTML <img> tag with the
    image embedded as a base64-encoded data URL.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded_string}" class="img-fluid" />'


def generate_full_report(data, concept_relationships, output_path, plot_paths, tb_metrics, metadata, test_metrics=None):
    """
    Generates an HTML report that includes tables and, optionally, embedded plots.
    """
    def generate_section(title, content):
        return f"<h2 class='mt-4'>{title}</h2>{content}"

    def format_dict_table(name, dictionary):
        if not dictionary:
            return f"<p>No {name} concepts.</p>"

        rows = []
        for concept, details in dictionary.items():
            parents = details.get('parents', [])

            if name == "missing":
                row = {
                    "Concept": concept,
                    "Parent": ', '.join(parents) if isinstance(parents, list) else str(parents),
                    "Inclusion Degree": round(details.get('inclusion_degree', 0), 3) if details.get(
                        'inclusion_degree') is not None else "N/A"
                }
                rows.append(row)
            else:
                inclusion_degrees = details.get('inclusion_degrees', {})
                if parents:
                    for parent in parents:
                        degree = inclusion_degrees.get(parent, "N/A")
                        if isinstance(degree, (int, float)):
                            degree = round(degree, 3)

                        row = {
                            "Concept": concept,
                            "Parent": parent,
                            "Inclusion Degree": degree
                        }
                        rows.append(row)
                else:
                    rows.append({"Concept": concept, "Parent": "N/A", "Inclusion Degree": "N/A"})

        df = pd.DataFrame(rows)
        return df.to_html(classes='table table-striped', index=False, border=0)

    def generate_relationship_section(relationships, threshold=0.8):
        lines = []
        for concept, details in relationships.items():
            parents = details.get("parents", [])
            for p in parents:
                parent = p.get("parent", "")
                typ = p.get("inclusion_degree", 0)
                if threshold <= typ:
                    if typ > 1.0:
                        lines.append(f"{concept} &#8838; {parent}")
                    else:
                        lines.append(f"T({concept}) &#8838; {parent}")
        return "<ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>"

    html = """
    <html>
    <head>
    <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css'>
    <style>
    table.table {
        table-layout: auto;
        width: 100%;
    }
    table.table th, table.table td {
        text-align: left;
        vertical-align: middle;
        white-space: nowrap;
    }
    table.table td {
        padding: 8px;
    }
</style>
    <title>Full Report</title>
    </head>
    <body>
    <div class='container'>
    """
    if metadata:
        html += generate_metadata_section(metadata)

    html += generate_section("Missing Concepts", format_dict_table("missing", data.get("missing_in_trained", {})))
    html += generate_section("New Concepts", format_dict_table("new", data.get("new_in_trained", {})))
    html += generate_section("Concept Relationships", generate_relationship_section(concept_relationships,
                                                                                    metadata.get("Typicality Threshold",
                                                                                                 0.8)))
    if tb_metrics:
        full_table_df = create_full_metrics_table(tb_metrics, test_metrics)
        html += generate_section("Training Metrics Over Epochs",
                                 generate_metric_table_section("Logged Metrics", full_table_df))
    if plot_paths:
        plot_items = list(plot_paths.items())
        plots_html = "<div class='row'>"

        for idx, (plot_name, plot_path) in enumerate(plot_items):
            pretty_title = plot_name.replace("_", " ").title()
            plots_html += f"""
            <div class='col-md-6 mb-4'>
                <h4>{pretty_title}</h4>
                {embed_image(plot_path)}
            </div>
            """
            if (idx + 1) % 2 == 0 and idx + 1 < len(plot_items):
                plots_html += "</div><div class='row'>"

        plots_html += "</div>"
        html += generate_section("Plots", plots_html)

    html += "</div></body></html>"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def load_scalars(log_dir, tags_to_extract):
    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    extracted = {}
    for tag in tags_to_extract:
        if tag in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            extracted[tag] = (steps, values)
    return extracted


def load_test_scalars(log_dir):
    """Extract only the test metrics (those with 'TEST_' prefix)"""
    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    test_metrics = {}
    for tag in accumulator.Tags().get("scalars", []):
        if tag.startswith('TEST_'):
            events = accumulator.Scalars(tag)
            if events:
                clean_tag = tag.replace('TEST_', '')
                test_metrics[clean_tag] = events[-1].value

    return test_metrics


def plot_selected_metrics(metrics_dict, selected_tags, output_path, title="Metric Trends", ylabel="Value"):
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag in selected_tags:
        if tag in metrics_dict:
            steps, values = metrics_dict[tag]
            if not tag.startswith('TEST_'):
                ax.plot(steps, values, label=tag)
    ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(True)
    save_plot(fig, output_path)


def create_full_metrics_table(tb_metrics, test_metrics=None):
    metric_name_short = {
        "Hits@1": "H@1",
        "Hits@3": "H@3",
        "Hits@5": "H@5",
        "Hits@10": "H@10",
        "Adjusted Mean Rank Index": "AMRI",
        "Rank Biased Precision": "RBP",
        "Mean First Relevant": "MFR",
    }

    all_steps = sorted({step for (steps, _) in tb_metrics.values() for step in steps})
    if not all_steps:
        return pd.DataFrame()

    rows = []
    for step in all_steps:
        row = {'Epoch': step}
        for metric, (steps, values) in tb_metrics.items():
            if metric.startswith('TEST_'):
                continue

            metric_name = metric_name_short.get(metric, metric)
            if step in steps:
                idx = steps.index(step)
                row[metric_name] = round(values[idx], 2)
            else:
                row[metric_name] = "-"
        rows.append(row)

    if test_metrics and len(test_metrics) > 0:
        test_row = {'Epoch': 'TEST'}
        for metric, value in test_metrics.items():
            metric_name = metric_name_short.get(metric, metric)
            test_row[metric_name] = round(value, 2)
        rows.append(test_row)

    df = pd.DataFrame(rows)
    return df


def generate_metric_table_section(title, df):
    return f"<h3 class='mt-4'>{title}</h3>" + df.to_html(classes='table table-bordered table-sm', index=False)


def generate_metadata_section(metadata_dict):
    rows_html = ""
    for key, value in metadata_dict.items():
        rows_html += f"<tr><th scope='row'>{key}</th><td>{value}</td></tr>"
    return f"""
    <h2 class='mt-4'>Experiment Info</h2>
    <table class='table table-bordered w-50'>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


def format_dict_table_latex(name, dictionary):
    if not dictionary:
        return f"% No {name} concepts.\n"

    rows = []
    for concept, details in dictionary.items():
        parents = details.get('parents', [])

        if name == "missing":
            row = {
                "Concept": concept,
                "Parent": ', '.join(parents) if isinstance(parents, list) else str(parents),
                "Inclusion Degree": round(details.get('inclusion_degree', 0), 3) if details.get(
                    'inclusion_degree') is not None else "N/A"
            }
            rows.append(row)
        else:
            inclusion_degrees = details.get('inclusion_degrees', {})
            if parents:
                for parent in parents:
                    degree = inclusion_degrees.get(parent, "N/A")
                    if isinstance(degree, (int, float)):
                        degree = round(degree, 3)

                    row = {
                        "Concept": concept,
                        "Parent": parent,
                        "Inclusion Degree": degree
                    }
                    rows.append(row)
            else:
                rows.append({"Concept": concept, "Parent": "N/A", "Inclusion Degree": "N/A"})

    df = pd.DataFrame(rows)

    df['Concept'] = df['Concept'].apply(lambda x: f"${str(x).replace('_', '\\_')}$")
    df['Parent'] = df['Parent'].apply(lambda x: f"${str(x).replace('_', '\\_')}$" if x != "N/A" else x)

    return df.to_latex(index=False, longtable=True, caption=f"{name.capitalize()} concepts")


def shorten_path(full_path):
    p = Path(full_path)
    parent_dir = p.parent.name
    file_stem = p.stem
    return str(Path(parent_dir) / file_stem).replace("\\", "/")


def embed_image_latex(image_path):
    image_path = shorten_path(image_path)
    image_stem = Path(image_path).stem
    caption = image_stem.replace('_', ' ').title()
    return (
        f"\\begin{{figure}}[ht]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{image_path}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"\\end{{figure}}"
    )


def generate_relationships_latex(concept_relationships, threshold=0.8):
    lines = []
    for concept, details in concept_relationships.items():
        parents = details.get('parents', [])
        for p in parents:
            parent = p.get('parent', '')
            typ = p.get('inclusion_degree', 0)

            escaped_concept = concept.replace('_', '\\_')
            escaped_parent = parent.replace('_', '\\_')

            if threshold <= typ:
                if typ > 1.0:
                    lines.append(f"${escaped_concept} \\sqsubseteq {escaped_parent}$")
                else:
                    lines.append(f"$T({escaped_concept}) \\sqsubseteq {escaped_parent}$")

    if not lines:
        return "% No concept relationship items.\n"

    content = "\\begin{itemize}\n"
    for line in lines:
        content += f"  \\item {line}\n"
    content += "\\end{itemize}\n"
    return content


def generate_full_report_latex(data, concept_relationships, output_path, plot_paths, metrics, metadata,
                               test_metrics=None):
    lines = []
    lines.append("\\documentclass{article}")
    lines.append("\\usepackage[utf8]{inputenc}")
    lines.append("\\usepackage{graphicx}")
    lines.append("\\usepackage{longtable}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{amssymb}")
    lines.append("\\usepackage{placeins}")
    lines.append("\\begin{document}\n")

    lines.append("\\section*{Experiment Info}")
    lines.append("\\begin{tabular}{ll}")
    for k, v in metadata.items():
        escaped_v = str(v).replace('_', '\\_')
        lines.append(f"{k} & {escaped_v} \\\\")
    lines.append("\\end{tabular}\n")

    # Missing
    lines.append("\\section{Missing Concepts}")
    lines.append(format_dict_table_latex('missing', data.get('missing_in_trained', {})))

    # New
    lines.append("\\section{New Concepts}")
    lines.append(format_dict_table_latex('new', data.get('new_in_trained', {})))

    # Relationships
    lines.append("\\section{Concept Relationships}")
    lines.append(generate_relationships_latex(concept_relationships, metadata.get("Typicality Threshold", 0.8)))

    # TensorBoard metrics tables
    df = create_full_metrics_table(metrics, test_metrics)

    # Convert dataframe to LaTeX table
    latex_table = df.to_latex(
        index=False,
        longtable=False,
        float_format="%.2f",
        header=True,
        escape=False
    )

    lines.append("\\begin{table}[ht]")
    lines.append("  \\centering")
    lines.append("  \\caption{Logged Metrics}")
    lines.append("  \\resizebox{\\textwidth}{!}{%")
    lines.append(latex_table.strip())
    lines.append("  }")
    lines.append("\\end{table}")

    # Discrepancies plots
    lines.append("\\section{Discrepancies Plots}")
    lines.append("% Each plot is in its own figure environment so LaTeX can break pages naturally")
    for _, path in plot_paths.items():
        lines.append(embed_image_latex(path))
        lines.append("\\vspace{0.5cm}")
    lines.append("\\FloatBarrier")

    lines.append("\\end{document}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def generate_all_reports(comparison_file, concept_relationships, results_dir, summary_dir, metadata):
    data = load_data(comparison_file)
    discrepancies = data.get("discrepancies", {})

    plots_dir = os.path.join(results_dir, f"plots_{metadata["Date"]}")
    os.makedirs(plots_dir, exist_ok=True)

    scatter_path = os.path.join(plots_dir, "scatter_plot.png")
    bar_path = os.path.join(plots_dir, "bar_plot.png")
    diff_path = os.path.join(plots_dir, "difference_barplot.png")
    scatter_plot(discrepancies, scatter_path)
    bar_plot(discrepancies, bar_path)
    difference_barplot(discrepancies, diff_path)

    plot_paths = {
        "scatter": scatter_path,
        "bar": bar_path,
        "difference": diff_path
    }

    tags = [
        'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10',
        'Adjusted Mean Rank Index', 'Rank Biased Precision', 'Mean First Relevant'
    ]

    metrics = load_scalars(summary_dir, tags)

    test_metrics = load_test_scalars(summary_dir)

    hits_plot_path = os.path.join(plots_dir, "hits_at_k.png")
    ranks_plot_path = os.path.join(plots_dir, "ranking_metrics.png")
    amri_plot_path = os.path.join(plots_dir, "adjusted_mean_rank_index.png")

    plot_selected_metrics(
        metrics,
        ['Hits@1', 'Hits@3', 'Hits@5', 'Hits@10'],
        hits_plot_path,
        title="Hits@K Over Epochs",
        ylabel="Hits@K"
    )

    plot_selected_metrics(
        metrics,
        ['Mean First Relevant'],
        ranks_plot_path,
        title="Ranking Metrics Over Epochs",
        ylabel="Rank"
    )

    plot_selected_metrics(
        metrics,
        ['Adjusted Mean Rank Index', 'Rank Biased Precision'],
        amri_plot_path,
        title="Ranking Metrics Over Epochs",
        ylabel="Metric Value"
    )

    plot_paths["hits_at_k"] = hits_plot_path
    plot_paths["ranking_metrics"] = ranks_plot_path
    plot_paths["adjusted_mean_rank_index"] = amri_plot_path

    generate_full_report(data, concept_relationships,
                         os.path.join(results_dir, "full_report.html"),
                         plot_paths, metrics, metadata, test_metrics)

    generate_full_report_latex(data, concept_relationships,
                               os.path.join(results_dir, "full_report.tex"),
                               plot_paths, metrics, metadata, test_metrics)
