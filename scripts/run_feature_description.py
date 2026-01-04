#!/usr/bin/env python3
"""Feature description script for PRISM-Bio.

This script performs mechanistic interpretability analysis on protein language models.
It loads protein sequences, extracts activations, clusters them, and generates
descriptions of what features represent.

Usage:
    python scripts/run_feature_description.py --config configs/default.yaml
    python scripts/run_feature_description.py --config configs/experiments/quick_test.yaml
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.utils.logging_utils import setup_logging, log_system_info
from src.utils import constants, helpers
from src.utils.output import save_description_csv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PRISM-Bio Feature Description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment name",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name (e.g., facebook/esm2_t6_8M_UR50D)",
    )
    parser.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="Override layer IDs to analyze",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override maximum number of samples",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Override number of clusters",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )

    return parser.parse_args()


def build_cli_overrides(args) -> dict:
    """Build CLI overrides dictionary from parsed arguments."""
    overrides = {}

    if args.experiment_name:
        overrides["experiment_name"] = args.experiment_name

    if args.output_dir:
        overrides["output_dir"] = str(args.output_dir)

    if args.verbose:
        overrides["verbose"] = True

    if args.model_name or args.layer_ids:
        overrides["model"] = {}
        if args.model_name:
            overrides["model"]["model_name"] = args.model_name
        if args.layer_ids:
            overrides["model"]["layer_ids"] = args.layer_ids

    if args.max_samples:
        overrides["dataset"] = {"max_samples": args.max_samples}

    if args.n_clusters:
        overrides["clustering"] = {"n_clusters": args.n_clusters}

    return overrides


def main():
    """Main entry point."""
    args = parse_args()

    # Build CLI overrides
    cli_overrides = build_cli_overrides(args)

    # Load configuration
    config = load_config(
        config_path=args.config,
        cli_overrides=cli_overrides,
    )

    # Ensure output directories exist
    constants.set_base_output_dir(Path(config.output_dir))
    constants.ensure_output_dirs()

    # Setup logging
    logger, log_file = setup_logging(
        config.to_dict(),
        mode="feature_description",
        verbose=config.verbose,
    )

    # Log system info
    log_system_info(logger)

    # Set random seed
    helpers.set_seed(config.random_seed)

    logger.info("=" * 80)
    logger.info("PRISM-BIO FEATURE DESCRIPTION")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Layers: {config.model.layer_ids}")
    logger.info(f"Dataset: {config.dataset.hf_dataset_name}")
    logger.info(f"Max samples: {config.dataset.max_samples}")
    logger.info(f"Clustering: {config.clustering.algorithm}")

    try:
        # Step 1: Load dataset
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING DATASET")
        logger.info("=" * 80)

        from src.data import get_uniref50, HuggingFaceProteinLoader, ProteinDataset

        if config.dataset.source == "huggingface":
            loader = HuggingFaceProteinLoader(
                hf_dataset_name=config.dataset.hf_dataset_name,
                hf_subset=config.dataset.hf_subset,
                hf_split=config.dataset.hf_split,
                streaming=config.dataset.streaming,
                max_samples=config.dataset.max_samples,
                sequence_column=config.dataset.sequence_column,
                id_column=config.dataset.id_column,
            )
            sequences = list(loader.load())
        else:
            logger.warning(f"Dataset source {config.dataset.source} not yet implemented")
            sequences = []

        logger.info(f"Loaded {len(sequences)} sequences")

        if not sequences:
            logger.error("No sequences loaded. Exiting.")
            return 1

        # Step 2: Load model and tokenizer
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 2: LOADING MODEL")
        logger.info("=" * 80)

        from transformers import AutoTokenizer, AutoModel
        import torch

        device = helpers.get_device(config.model.device)
        logger.info(f"Using device: {device}")

        # Get token if available (for gated/private models)
        hf_token = helpers.get_huggingface_token()
        if hf_token:
            logger.debug(f"HuggingFace token found - will use for model download")
        else:
            logger.debug(f"No HuggingFace token set - using public access")

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            token=hf_token,  # Pass token explicitly (None is fine for public models)
        )
        model = AutoModel.from_pretrained(
            config.model.model_name,
            torch_dtype=torch.float16 if config.model.dtype == "float16" else torch.float32,
            token=hf_token,  # Pass token explicitly (None is fine for public models)
        )
        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded: {config.model.model_name}")

        # Step 3: Create dataset and extract activations
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 3: EXTRACTING ACTIVATIONS")
        logger.info("=" * 80)

        from torch.utils.data import DataLoader
        from src.data import ProteinCollator

        dataset = ProteinDataset(
            sequences,
            tokenizer,
            max_length=config.dataset.max_sequence_length,
            min_length=config.dataset.min_sequence_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config.model.batch_size,
            shuffle=False,
            collate_fn=ProteinCollator(),
        )

        # Extract embeddings
        # Process only the first layer to avoid metadata/embedding mismatch
        # If multiple layers are specified, we'll use the first one
        primary_layer_id = config.model.layer_ids[0]
        if len(config.model.layer_ids) > 1:
            logger.warning(
                f"Multiple layers specified ({config.model.layer_ids}). "
                f"Using only the first layer ({primary_layer_id}) for clustering to avoid metadata mismatch."
            )
        
        all_embeddings = []
        all_metadata = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Get embeddings from primary layer only
                hidden_states = outputs.hidden_states[primary_layer_id]

                # Aggregate over sequence
                if config.model.aggregation == "mean":
                    # Mean pool over non-padding tokens
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                elif config.model.aggregation == "first":
                    embeddings = hidden_states[:, 0]  # CLS token
                else:
                    embeddings = hidden_states.mean(dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

                # Store metadata with sequence info (once per batch, matching embeddings)
                for i, seq_id in enumerate(batch["sequence_id"]):
                    # Find the original sequence data
                    seq_data = next((s for s in sequences if s.get("id") == seq_id), None)
                    if seq_data:
                        all_metadata.append({
                            "id": seq_id,
                            "sequence": seq_data.get("sequence", ""),
                        })
                    else:
                        all_metadata.append({"id": seq_id})

                # Clear GPU cache more frequently to prevent OOM
                if batch_idx % 5 == 0:
                    helpers.clear_gpu_cache(device)

        import numpy as np
        
        # Check if we have any embeddings
        if len(all_embeddings) == 0:
            raise ValueError(
                "No embeddings were extracted. This usually means:\n"
                "1. No sequences passed filtering (check sequence_column and id_column match dataset)\n"
                "2. All sequences were filtered out (check min/max_sequence_length)\n"
                "3. Dataset is empty or not loading correctly"
            )
        
        all_embeddings = np.vstack(all_embeddings)
        logger.info(f"Extracted embeddings: {all_embeddings.shape}")
        
        # Verify metadata and embeddings are aligned
        if len(all_metadata) != len(all_embeddings):
            raise ValueError(
                f"Metadata/embeddings mismatch: {len(all_metadata)} metadata items "
                f"but {len(all_embeddings)} embeddings. This can happen if multiple "
                f"layers are processed. Using only the first layer: {primary_layer_id}"
            )

        # Check for and handle inf/nan values in embeddings
        if np.any(~np.isfinite(all_embeddings)):
            n_invalid = np.sum(~np.isfinite(all_embeddings))
            logger.warning(f"Found {n_invalid} inf/nan values in embeddings. Replacing with 0.")
            all_embeddings = np.nan_to_num(all_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 4: Clustering
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: CLUSTERING")
        logger.info("=" * 80)

        from src.analysis.clustering import cluster_embeddings, compute_cluster_statistics, compute_cluster_similarity

        labels = cluster_embeddings(
            all_embeddings,
            method=config.clustering.algorithm,
            n_clusters=config.clustering.n_clusters,
            seed=config.clustering.random_seed,
        )

        stats = compute_cluster_statistics(all_embeddings, labels)
        logger.info(f"Clustering complete. Found {len(stats)} clusters.")

        # Compute inter-cluster similarity for polysemanticity analysis
        similarity_results = compute_cluster_similarity(stats)
        logger.info(f"Inter-cluster similarity: overall avg = {similarity_results['overall_avg']:.4f}")

        # Get ALL sequences for each cluster (not just representatives)
        import numpy as np
        all_cluster_sequences = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            mask = labels == label
            indices = np.where(mask)[0]
            # Get all metadata for this cluster
            all_cluster_sequences[label] = [all_metadata[i] for i in indices]
            logger.info(f"Cluster {label}: {len(all_cluster_sequences[label])} sequences")
            
            # Log all sequences in this cluster
            logger.info(f"\n--- Cluster {label} Sequences ({len(all_cluster_sequences[label])} total) ---")
            for i, seq_data in enumerate(all_cluster_sequences[label], 1):
                seq_id = seq_data.get("id", f"seq_{i}")
                sequence = seq_data.get("sequence", "")
                logger.info(f"  {i}. {seq_id}: {sequence}")
            logger.info("")

        # Step 5: Visualization
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 5: VISUALIZATION")
        logger.info("=" * 80)

        from src.visualization import ClusterVisualizer

        viz = ClusterVisualizer(
            output_dir=str(Path(config.output_dir) / "visualizations"),
            dpi=config.visualization.dpi,
        )

        viz.create_full_report(
            all_embeddings,
            labels,
            all_metadata,
            reduction_method=config.visualization.reduction_method,
            experiment_name=config.experiment_name,
        )

        # Step 5.5: GO Enrichment Analysis (if enabled)
        go_enrichment_results = {}
        if config.go_enrichment.enabled:
            logger.info("")
            logger.info("=" * 80)
            logger.info("STEP 5.5: GO ENRICHMENT ANALYSIS")
            logger.info("=" * 80)

            from src.analysis.go_enrichment import run_cluster_go_enrichment

            # Build cluster sequences dict with full sequence data for GO extraction
            cluster_sequences_for_go = {}
            for label in unique_labels:
                if label == -1:
                    continue
                mask = labels == label
                indices = np.where(mask)[0]
                # Get full sequence data (with GO annotations) for this cluster
                cluster_sequences_for_go[label] = []
                for idx in indices:
                    seq_id = all_metadata[idx].get("id", "")
                    # Find full sequence data from original sequences
                    seq_data = next((s for s in sequences if s.get("id") == seq_id), None)
                    if seq_data:
                        cluster_sequences_for_go[label].append(seq_data)
                    else:
                        cluster_sequences_for_go[label].append(all_metadata[idx])

            try:
                go_enrichment_results = run_cluster_go_enrichment(
                    cluster_sequences=cluster_sequences_for_go,
                    all_sequences=sequences,
                    config=config,
                    experiment_name=config.experiment_name,
                    output_dir=Path(config.output_dir) / "results",
                )
                logger.info(f"GO enrichment completed for {len(go_enrichment_results)} clusters")
            except Exception as e:
                logger.error(f"GO enrichment failed: {e}")
                logger.warning("Continuing without GO enrichment results")
                go_enrichment_results = {}
        else:
            logger.info("")
            logger.info("GO enrichment is disabled in config. Skipping Step 5.5.")

        # Step 6: Save results
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("=" * 80)

        # Save ALL sequences to a separate file
        from src.utils.output import save_representative_sequences
        seq_filepath = save_representative_sequences(
            all_cluster_sequences,
            experiment_name=config.experiment_name,
            output_dir=Path(config.output_dir) / "visualizations",
        )
        logger.info(f"Saved all sequences to: {seq_filepath}")

        # Create descriptions with sequence IDs in highlights
        # Use GO-based descriptions if enrichment was performed
        descriptions = []
        for cluster_id in sorted(stats.keys()):
            # Get ALL sequence IDs for this cluster
            cluster_sequences = all_cluster_sequences.get(cluster_id, [])
            sequence_ids = [seq.get("id", "") for seq in cluster_sequences]  # All sequences

            # Use GO-based description if available, otherwise fallback to default
            if cluster_id in go_enrichment_results and go_enrichment_results[cluster_id].get("description"):
                description = go_enrichment_results[cluster_id]["description"]
            else:
                description = f"Cluster {cluster_id} with {stats[cluster_id]['size']} samples"

            # Get cluster similarity (avg similarity to other clusters)
            cluster_similarity = similarity_results["per_cluster_avg"].get(cluster_id, 1.0)

            descriptions.append({
                "layer": config.model.layer_ids[0],
                "unit": cluster_id,
                "description": description,
                "mean_activation": stats[cluster_id]["mean_distance"],
                "highlights": sequence_ids,  # Store sequence IDs instead of empty list
                "cluster_similarity": cluster_similarity,  # Avg cosine similarity to other clusters
            })

        output_path = save_description_csv(
            descriptions,
            model_name=config.experiment_name,
            target_model=config.model.model_name.split("/")[-1],
            layer_id=config.model.layer_ids[0],
            unit_id=0,
            output_dir=Path(config.output_dir) / "descriptions",
        )

        logger.info(f"Saved descriptions to: {output_path}")

        # Save the full similarity matrix to a JSON file
        import json
        from datetime import datetime
        import numpy as np
        
        # Helper function to convert numpy types to Python native types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert numpy types to Python native types for JSON serialization
        cluster_ids = [int(cid) for cid in similarity_results["cluster_ids"]]
        similarity_matrix = convert_numpy_types(similarity_results["similarity_matrix"])
        
        similarity_output = {
            "cluster_ids": cluster_ids,
            "similarity_matrix": similarity_matrix,
            "per_cluster_avg": {str(k): float(v) for k, v in similarity_results["per_cluster_avg"].items()},
            "overall_avg": float(similarity_results["overall_avg"]),
            "interpretation": "Lower avg similarity = more distinct clusters = potentially more polysemantic neuron",
            "metadata": {
                "experiment_name": config.experiment_name,
                "model": config.model.model_name,
                "layer": int(config.model.layer_ids[0]),
                "n_clusters": int(len(similarity_results["cluster_ids"])),
                "timestamp": datetime.now().isoformat(),
            }
        }
        
        results_dir = Path(config.output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        similarity_filepath = results_dir / f"similarity_matrix_{config.experiment_name}_{timestamp}.json"
        
        with open(similarity_filepath, "w") as f:
            json.dump(similarity_output, f, indent=2)
        
        logger.info(f"Saved similarity matrix to: {similarity_filepath}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Output directory: {config.output_dir}")

        return 0

    except Exception as e:
        logger.exception(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())




