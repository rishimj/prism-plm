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

        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        model = AutoModel.from_pretrained(
            config.model.model_name,
            torch_dtype=torch.float16 if config.model.dtype == "float16" else torch.float32,
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

                # Get embeddings from specified layer
                for layer_id in config.model.layer_ids:
                    hidden_states = outputs.hidden_states[layer_id]

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

                # Store metadata
                for seq_id in batch["sequence_id"]:
                    all_metadata.append({"id": seq_id})

                # Clear GPU cache periodically
                if batch_idx % 10 == 0:
                    helpers.clear_gpu_cache(device)

        import numpy as np
        all_embeddings = np.vstack(all_embeddings)
        logger.info(f"Extracted embeddings: {all_embeddings.shape}")

        # Step 4: Clustering
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: CLUSTERING")
        logger.info("=" * 80)

        from src.analysis.clustering import cluster_embeddings, compute_cluster_statistics

        labels = cluster_embeddings(
            all_embeddings,
            method=config.clustering.algorithm,
            n_clusters=config.clustering.n_clusters,
            seed=config.clustering.random_seed,
        )

        stats = compute_cluster_statistics(all_embeddings, labels)
        logger.info(f"Clustering complete. Found {len(stats)} clusters.")

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

        # Step 6: Save results
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("=" * 80)

        # Create descriptions (placeholder - would use LLM in full implementation)
        descriptions = []
        for cluster_id in sorted(stats.keys()):
            descriptions.append({
                "layer": config.model.layer_ids[0],
                "unit": cluster_id,
                "description": f"Cluster {cluster_id} with {stats[cluster_id]['size']} samples",
                "mean_activation": stats[cluster_id]["mean_distance"],
                "highlights": [],
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

