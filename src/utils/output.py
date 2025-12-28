"""Output utilities for saving results in PRISM-compatible format."""
import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.utils import constants

logger = get_logger("utils.output")


def save_description_csv(
    descriptions: List[Dict[str, Any]],
    model_name: str,
    target_model: str,
    layer_id: int,
    unit_id: int,
    output_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Save feature descriptions to CSV in PRISM format.

    Output path: descriptions/{model_name}/{target_model}/{model}_layer-{L}_unit-{U}_{timestamp}.csv

    Args:
        descriptions: List of description dictionaries with keys:
            - layer: Layer ID
            - unit: Unit ID
            - description: Text description
            - mean_activation: Mean activation value
            - highlights: List of highlighted tokens
        model_name: Name of the model generating descriptions
        target_model: Name of the target model being analyzed
        layer_id: Layer ID being analyzed
        unit_id: Unit ID being analyzed
        output_dir: Optional output directory override
        timestamp: Optional timestamp string (defaults to current time)

    Returns:
        Path to the saved CSV file
    """
    logger.debug(f"Saving descriptions for layer {layer_id}, unit {unit_id}")
    logger.debug(f"  Model: {model_name}")
    logger.debug(f"  Target: {target_model}")
    logger.debug(f"  Number of descriptions: {len(descriptions)}")

    # Generate output path
    if output_dir is None:
        output_dir = constants.get_descriptions_dir(model_name, target_model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"  Output directory: {output_dir}")

    # Generate timestamp
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    # Clean model name for filename
    target_model_clean = target_model.replace("/", "-").replace(".", "-")
    filename = f"{target_model_clean}_layer-{layer_id}_unit-{unit_id}_{timestamp}.csv"
    filepath = output_dir / filename

    logger.info(f"Saving descriptions to: {filepath}")

    # Write CSV
    fieldnames = ["layer", "unit", "description", "mean_activation", "highlights"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, desc in enumerate(descriptions):
            # Convert highlights to string if it's a list
            highlights = desc.get("highlights", [])
            if isinstance(highlights, list):
                highlights_str = str(highlights)
            else:
                highlights_str = str(highlights)

            row = {
                "layer": desc.get("layer", layer_id),
                "unit": desc.get("unit", unit_id),
                "description": desc.get("description", ""),
                "mean_activation": desc.get("mean_activation", ""),
                "highlights": highlights_str,
            }
            writer.writerow(row)
            logger.debug(
                f"  Wrote description {i}: {desc.get('description', '')[:50]}..."
            )

    logger.info(f"Saved {len(descriptions)} descriptions to {filepath}")
    return filepath


def save_explanation_csv(
    explanations: List[Dict[str, Any]],
    model_name: str,
    method_name: str,
    layers: List[int],
    n_samples: int,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save feature explanations to CSV in PRISM format.

    Output path: assets/explanations/{method_name}/{model}_layer{L1}_layer{L2}_{N}-samples.csv

    Args:
        explanations: List of explanation dictionaries
        model_name: Name of the model
        method_name: Name of the explanation method
        layers: List of layer IDs analyzed
        n_samples: Number of samples used
        output_dir: Optional output directory override

    Returns:
        Path to the saved CSV file
    """
    logger.debug(f"Saving explanations for {model_name}")
    logger.debug(f"  Method: {method_name}")
    logger.debug(f"  Layers: {layers}")
    logger.debug(f"  Samples: {n_samples}")
    logger.debug(f"  Number of explanations: {len(explanations)}")

    if output_dir is None:
        output_dir = constants.get_explanations_dir(method_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"  Output directory: {output_dir}")

    # Generate filename
    model_name_clean = model_name.replace("/", "-").replace(".", "-")
    layer_str = "_".join(f"layer{l}" for l in layers)
    filename = f"{model_name_clean}_{layer_str}_{n_samples}-samples.csv"
    filepath = output_dir / filename

    logger.info(f"Saving explanations to: {filepath}")

    # Write CSV
    if explanations:
        fieldnames = list(explanations[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in explanations:
                writer.writerow(row)
    else:
        # Write empty file with header
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            f.write("layer,unit,description,input_success\n")
        logger.warning("No explanations to save, wrote empty file")

    logger.info(f"Saved {len(explanations)} explanations to {filepath}")
    return filepath


def save_evaluation_csv(
    results: List[Dict[str, Any]],
    method_name: str,
    target_model: str,
    text_gen_model: str,
    eval_gen_model: str,
    aggregation: str,
    dataset: str,
    n_samples: int,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save evaluation results in PRISM format.

    Output path: results/cosy-evaluation_{method}_{target}_{textgen}_{agg}_{evalgen}_{dataset}_{n}.csv

    Args:
        results: List of evaluation result dictionaries
        method_name: Name of the evaluation method
        target_model: Name of the target model
        text_gen_model: Name of the text generation model
        eval_gen_model: Name of the evaluation model
        aggregation: Aggregation method used
        dataset: Dataset name
        n_samples: Number of samples
        output_dir: Optional output directory override

    Returns:
        Path to the saved CSV file
    """
    logger.debug(f"Saving evaluation results")
    logger.debug(f"  Method: {method_name}")
    logger.debug(f"  Target model: {target_model}")
    logger.debug(f"  Text gen model: {text_gen_model}")
    logger.debug(f"  Eval gen model: {eval_gen_model}")
    logger.debug(f"  Aggregation: {aggregation}")
    logger.debug(f"  Dataset: {dataset}")
    logger.debug(f"  Samples: {n_samples}")
    logger.debug(f"  Number of results: {len(results)}")

    if output_dir is None:
        output_dir = constants.get_results_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"  Output directory: {output_dir}")

    # Clean names for filename
    target_clean = target_model.replace("/", "-").replace(".", "-")
    textgen_clean = text_gen_model.replace("/", "-").replace(".", "-")
    evalgen_clean = eval_gen_model.replace("/", "-").replace(".", "-")

    filename = (
        f"cosy-evaluation_{method_name}_target-{target_clean}_"
        f"textgen-{textgen_clean}_{aggregation}_evalgen-{evalgen_clean}_"
        f"{dataset}_{n_samples}.csv"
    )
    filepath = output_dir / filename

    logger.info(f"Saving evaluation results to: {filepath}")

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    else:
        logger.warning("No results to save")

    logger.info(f"Saved {len(results)} evaluation results to {filepath}")
    return filepath


def save_cluster_statistics(
    statistics: Dict[str, Any],
    experiment_name: str,
    output_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Save cluster statistics to JSON.

    Args:
        statistics: Dictionary of cluster statistics
        experiment_name: Name of the experiment
        output_dir: Optional output directory override
        timestamp: Optional timestamp string

    Returns:
        Path to the saved JSON file
    """
    logger.debug(f"Saving cluster statistics for {experiment_name}")

    if output_dir is None:
        output_dir = constants.get_visualizations_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    filename = f"cluster_statistics_{experiment_name}_{timestamp}.json"
    filepath = output_dir / filename

    logger.info(f"Saving cluster statistics to: {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, default=str)

    logger.info(f"Saved cluster statistics to {filepath}")
    return filepath


def save_representative_sequences(
    sequences: Dict[int, List[Dict[str, Any]]],
    experiment_name: str,
    output_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Save representative sequences per cluster to text file.

    Args:
        sequences: Dictionary mapping cluster ID to list of sequence dicts
        experiment_name: Name of the experiment
        output_dir: Optional output directory override
        timestamp: Optional timestamp string

    Returns:
        Path to the saved text file
    """
    logger.debug(f"Saving representative sequences for {experiment_name}")
    logger.debug(f"  Number of clusters: {len(sequences)}")

    if output_dir is None:
        output_dir = constants.get_visualizations_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    filename = f"representative_sequences_{experiment_name}_{timestamp}.txt"
    filepath = output_dir / filename

    logger.info(f"Saving representative sequences to: {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"REPRESENTATIVE SEQUENCES - {experiment_name}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        for cluster_id in sorted(sequences.keys()):
            seqs = sequences[cluster_id]
            f.write(f"CLUSTER {cluster_id} ({len(seqs)} sequences)\n")
            f.write("-" * 80 + "\n")

            for i, seq_dict in enumerate(seqs[:10], 1):  # Max 10 per cluster
                seq_id = seq_dict.get("id", f"seq_{i}")
                sequence = seq_dict.get("sequence", "")
                # Truncate long sequences
                if len(sequence) > 100:
                    sequence = sequence[:50] + "..." + sequence[-50:]
                f.write(f"  {i}. {seq_id}: {sequence}\n")

            f.write("\n")

    logger.info(f"Saved representative sequences to {filepath}")
    return filepath


def save_go_enrichment_results(
    enrichment_results: List[Dict[str, Any]],
    cluster_id: int,
    experiment_name: str,
    output_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Save GO enrichment results for a cluster.

    Args:
        enrichment_results: List of GO enrichment result dictionaries
        cluster_id: ID of the cluster
        experiment_name: Name of the experiment
        output_dir: Optional output directory override
        timestamp: Optional timestamp string

    Returns:
        Path to the saved CSV file
    """
    logger.debug(f"Saving GO enrichment results for cluster {cluster_id}")
    logger.debug(f"  Experiment: {experiment_name}")
    logger.debug(f"  Number of results: {len(enrichment_results)}")

    if output_dir is None:
        output_dir = constants.get_results_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    filename = f"go_enrichment_{experiment_name}_cluster{cluster_id}_{timestamp}.csv"
    filepath = output_dir / filename

    logger.info(f"Saving GO enrichment results to: {filepath}")

    if enrichment_results:
        fieldnames = list(enrichment_results[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enrichment_results)
    else:
        logger.warning("No enrichment results to save")

    logger.info(f"Saved {len(enrichment_results)} GO enrichment results to {filepath}")
    return filepath

