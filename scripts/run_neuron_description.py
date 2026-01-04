#!/usr/bin/env python3
"""Per-neuron feature description script for PRISM-Bio.

This script performs mechanistic interpretability analysis on individual neurons
in protein language models. For each neuron (dimension) in a layer, it:
1. Extracts activation values across all sequences
2. Finds sequences that highly activate that neuron
3. Optionally clusters those sequences to find sub-patterns
4. Generates descriptions for what each neuron responds to

Usage:
    # Analyze neurons 0-9 in layer 18
    python scripts/run_neuron_description.py --config configs/models/esm2_3b.yaml --neuron-ids 0 1 2 3 4 5 6 7 8 9

    # Analyze top 10 most variable neurons
    python scripts/run_neuron_description.py --config configs/models/esm2_3b.yaml --top-variable 10

    # Analyze all neurons (warning: expensive!)
    python scripts/run_neuron_description.py --config configs/models/esm2_3b.yaml --all-neurons
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.utils.logging_utils import setup_logging, log_system_info
from src.utils import constants, helpers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PRISM-Bio Per-Neuron Feature Description",
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
        "--layer-id",
        type=int,
        default=None,
        help="Layer ID to analyze (default: first in config)",
    )

    # Neuron selection (mutually exclusive)
    neuron_group = parser.add_mutually_exclusive_group()
    neuron_group.add_argument(
        "--neuron-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific neuron indices to analyze",
    )
    neuron_group.add_argument(
        "--neuron-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of neuron indices to analyze (inclusive)",
    )
    neuron_group.add_argument(
        "--top-variable",
        type=int,
        default=None,
        help="Analyze top N most variable neurons",
    )
    neuron_group.add_argument(
        "--top-active",
        type=int,
        default=None,
        help="Analyze top N neurons by mean activation",
    )
    neuron_group.add_argument(
        "--random-neurons",
        type=int,
        default=None,
        help="Analyze N randomly selected neurons",
    )
    neuron_group.add_argument(
        "--all-neurons",
        action="store_true",
        help="Analyze all neurons (warning: very expensive!)",
    )

    # Per-neuron analysis settings
    parser.add_argument(
        "--top-k-sequences",
        type=int,
        default=100,
        help="Number of top-activating sequences to consider per neuron",
    )
    parser.add_argument(
        "--cluster-per-neuron",
        action="store_true",
        help="Cluster top-activating sequences for each neuron",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters per neuron (if clustering)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override maximum number of sequences to load",
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

    if args.model_name:
        overrides["model"] = {"model_name": args.model_name}

    if args.max_samples:
        overrides["dataset"] = {"max_samples": args.max_samples}

    return overrides


def select_neurons(
    all_activations,  # Shape: (n_sequences, n_neurons)
    args,
    logger,
    random_seed: int = 42,
) -> List[int]:
    """Select which neurons to analyze based on CLI arguments."""
    import numpy as np
    
    n_neurons = all_activations.shape[1]
    
    if args.neuron_ids:
        # Specific neuron indices
        neuron_ids = [n for n in args.neuron_ids if 0 <= n < n_neurons]
        if len(neuron_ids) != len(args.neuron_ids):
            logger.warning(f"Some neuron IDs out of range (max: {n_neurons - 1})")
        return neuron_ids
    
    elif args.neuron_range:
        start, end = args.neuron_range
        start = max(0, start)
        end = min(n_neurons - 1, end)
        return list(range(start, end + 1))
    
    elif args.top_variable:
        # Select neurons with highest variance
        variances = np.var(all_activations, axis=0)
        top_indices = np.argsort(variances)[-args.top_variable:][::-1]
        logger.info(f"Top {args.top_variable} most variable neurons: {top_indices.tolist()}")
        return top_indices.tolist()
    
    elif args.top_active:
        # Select neurons with highest mean activation
        means = np.mean(all_activations, axis=0)
        top_indices = np.argsort(means)[-args.top_active:][::-1]
        logger.info(f"Top {args.top_active} most active neurons: {top_indices.tolist()}")
        return top_indices.tolist()
    
    elif args.random_neurons:
        # Random selection
        np.random.seed(random_seed)
        indices = np.random.choice(n_neurons, size=min(args.random_neurons, n_neurons), replace=False)
        logger.info(f"Randomly selected {len(indices)} neurons: {sorted(indices.tolist())}")
        return sorted(indices.tolist())
    
    elif args.all_neurons:
        logger.warning(f"Analyzing ALL {n_neurons} neurons - this may take a long time!")
        return list(range(n_neurons))
    
    else:
        # Default: top 10 most variable
        logger.info("No neuron selection specified. Defaulting to top 10 most variable.")
        variances = np.var(all_activations, axis=0)
        top_indices = np.argsort(variances)[-10:][::-1]
        return top_indices.tolist()


def analyze_neuron(
    neuron_id: int,
    activations: "np.ndarray",  # 1D array of activations for this neuron
    all_metadata: List[Dict],
    sequences: List[Dict],
    top_k: int,
    do_cluster: bool,
    n_clusters: int,
    logger,
) -> Dict[str, Any]:
    """Analyze a single neuron: find top activating sequences and optionally cluster."""
    import numpy as np
    
    # Get top-k activating sequence indices
    top_indices = np.argsort(activations)[-top_k:][::-1]
    top_activations = activations[top_indices]
    
    # Get metadata for top sequences
    top_sequences = []
    for idx in top_indices:
        meta = all_metadata[idx]
        seq_id = meta.get("id", f"seq_{idx}")
        # Find full sequence data
        seq_data = next((s for s in sequences if s.get("id") == seq_id), None)
        
        top_sequences.append({
            "id": seq_id,
            "sequence": meta.get("sequence", seq_data.get("sequence", "") if seq_data else ""),
            "activation": float(activations[idx]),
            "rank": len(top_sequences) + 1,
        })
    
    result = {
        "neuron_id": neuron_id,
        "n_top_sequences": len(top_sequences),
        "mean_top_activation": float(np.mean(top_activations)),
        "max_activation": float(np.max(activations)),
        "min_activation": float(np.min(activations)),
        "overall_mean": float(np.mean(activations)),
        "overall_std": float(np.std(activations)),
        "top_sequences": top_sequences,
        "clusters": None,
    }
    
    # Optionally cluster the top sequences
    if do_cluster and len(top_sequences) >= n_clusters:
        try:
            from src.analysis.clustering import cluster_embeddings
            from sentence_transformers import SentenceTransformer
            
            # Embed sequences for clustering (using their string representations)
            # Note: This uses text embeddings, not the protein model activations
            texts = [f"{s['id']}: {s['sequence'][:100]}" for s in top_sequences]
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = embedder.encode(texts)
            
            labels = cluster_embeddings(
                embeddings,
                method="kmeans",
                n_clusters=n_clusters,
                seed=42,
            )
            
            # Group sequences by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(top_sequences[i])
            
            result["clusters"] = {
                int(k): {
                    "sequences": v,
                    "mean_activation": np.mean([s["activation"] for s in v]),
                }
                for k, v in clusters.items()
            }
        except Exception as e:
            logger.warning(f"Clustering failed for neuron {neuron_id}: {e}")
    
    return result


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
        mode="neuron_description",
        verbose=config.verbose,
    )

    # Log system info
    log_system_info(logger)

    # Set random seed
    helpers.set_seed(config.random_seed)

    logger.info("=" * 80)
    logger.info("PRISM-BIO PER-NEURON FEATURE DESCRIPTION")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Dataset: {config.dataset.hf_dataset_name}")
    logger.info(f"Max samples: {config.dataset.max_samples}")
    logger.info(f"Top-k sequences per neuron: {args.top_k_sequences}")
    logger.info(f"Cluster per neuron: {args.cluster_per_neuron}")

    try:
        import numpy as np
        import torch
        
        # Step 1: Load dataset
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING DATASET")
        logger.info("=" * 80)

        from src.data import HuggingFaceProteinLoader, ProteinDataset

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

        # Step 2: Load model
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 2: LOADING MODEL")
        logger.info("=" * 80)

        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import os

        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

        # Set dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(config.model.dtype, torch.float32)

        device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        logger.info(f"Using dtype: {torch_dtype}")

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            token=hf_token,
        )

        model = AutoModelForMaskedLM.from_pretrained(
            config.model.model_name,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded: {config.model.model_name}")

        # Step 3: Extract per-neuron activations
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 3: EXTRACTING PER-NEURON ACTIVATIONS")
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

        # Use first layer or specified layer
        layer_id = args.layer_id if args.layer_id is not None else config.model.layer_ids[0]
        logger.info(f"Analyzing layer: {layer_id}")

        # Extract full hidden states (all neurons)
        all_activations = []  # Shape will be (n_sequences, n_neurons)
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

                # Get hidden states from specified layer
                hidden_states = outputs.hidden_states[layer_id]

                # Aggregate over sequence (mean pooling)
                if config.model.aggregation == "mean":
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                    activations = sum_embeddings / sum_mask
                elif config.model.aggregation == "first":
                    activations = hidden_states[:, 0]
                elif config.model.aggregation == "max":
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    hidden_states[~mask_expanded.bool()] = float('-inf')
                    activations = hidden_states.max(dim=1)[0]
                else:
                    activations = hidden_states.mean(dim=1)

                all_activations.append(activations.cpu().numpy())

                # Store metadata
                for i, seq_id in enumerate(batch["sequence_id"]):
                    seq_data = next((s for s in sequences if s.get("id") == seq_id), None)
                    if seq_data:
                        all_metadata.append({
                            "id": seq_id,
                            "sequence": seq_data.get("sequence", ""),
                        })
                    else:
                        all_metadata.append({"id": seq_id})

                # Clear GPU cache periodically
                if batch_idx % 5 == 0:
                    helpers.clear_gpu_cache(device)

        # Stack all activations
        all_activations = np.vstack(all_activations)
        logger.info(f"Extracted activations shape: {all_activations.shape}")
        logger.info(f"  - {all_activations.shape[0]} sequences")
        logger.info(f"  - {all_activations.shape[1]} neurons (dimensions)")

        # Step 4: Select neurons to analyze
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: SELECTING NEURONS")
        logger.info("=" * 80)

        neuron_ids = select_neurons(all_activations, args, logger, config.random_seed)
        logger.info(f"Selected {len(neuron_ids)} neurons to analyze")

        # Step 5: Analyze each neuron
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 5: ANALYZING NEURONS")
        logger.info("=" * 80)

        neuron_results = []
        for i, neuron_id in enumerate(neuron_ids):
            logger.info(f"Analyzing neuron {neuron_id} ({i + 1}/{len(neuron_ids)})")
            
            result = analyze_neuron(
                neuron_id=neuron_id,
                activations=all_activations[:, neuron_id],
                all_metadata=all_metadata,
                sequences=sequences,
                top_k=args.top_k_sequences,
                do_cluster=args.cluster_per_neuron,
                n_clusters=args.n_clusters,
                logger=logger,
            )
            neuron_results.append(result)
            
            # Log summary
            logger.info(f"  Mean top activation: {result['mean_top_activation']:.4f}")
            logger.info(f"  Overall mean: {result['overall_mean']:.4f} ± {result['overall_std']:.4f}")
            if result['clusters']:
                logger.info(f"  Found {len(result['clusters'])} clusters")

        # Step 6: Save results
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("=" * 80)

        # Save detailed JSON results
        output_dir = Path(config.output_dir) / "neuron_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_short = config.model.model_name.split("/")[-1]
        
        # Full results (JSON)
        json_path = output_dir / f"{model_short}_layer-{layer_id}_neurons_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(x) for x in obj]
            return obj
        
        output_data = {
            "metadata": {
                "model": config.model.model_name,
                "layer_id": layer_id,
                "n_sequences": len(all_metadata),
                "n_neurons_analyzed": len(neuron_ids),
                "top_k_sequences": args.top_k_sequences,
                "clustered": args.cluster_per_neuron,
                "timestamp": timestamp,
            },
            "neurons": convert_numpy(neuron_results),
        }
        
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved detailed results to: {json_path}")

        # Save CSV summary
        csv_path = output_dir / f"{model_short}_layer-{layer_id}_summary_{timestamp}.csv"
        
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer", "neuron_id", "mean_activation", "std_activation",
                "max_activation", "top_k_mean", "n_top_sequences",
                "top_3_sequence_ids"
            ])
            for r in neuron_results:
                top_3_ids = ", ".join([s["id"] for s in r["top_sequences"][:3]])
                writer.writerow([
                    layer_id,
                    r["neuron_id"],
                    f"{r['overall_mean']:.6f}",
                    f"{r['overall_std']:.6f}",
                    f"{r['max_activation']:.6f}",
                    f"{r['mean_top_activation']:.6f}",
                    r["n_top_sequences"],
                    top_3_ids,
                ])
        logger.info(f"Saved summary CSV to: {csv_path}")

        # Save per-neuron sequence files
        seq_dir = output_dir / f"{model_short}_layer-{layer_id}_{timestamp}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        for r in neuron_results:
            neuron_file = seq_dir / f"neuron_{r['neuron_id']:05d}.txt"
            with open(neuron_file, "w") as f:
                f.write(f"NEURON {r['neuron_id']} - LAYER {layer_id}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Overall mean activation: {r['overall_mean']:.6f}\n")
                f.write(f"Overall std: {r['overall_std']:.6f}\n")
                f.write(f"Max activation: {r['max_activation']:.6f}\n")
                f.write(f"Mean of top-{args.top_k_sequences}: {r['mean_top_activation']:.6f}\n")
                f.write("\n")
                f.write(f"TOP {len(r['top_sequences'])} ACTIVATING SEQUENCES:\n")
                f.write("-" * 80 + "\n")
                for seq in r["top_sequences"]:
                    f.write(f"  {seq['rank']:3d}. [{seq['activation']:.4f}] {seq['id']}\n")
                    f.write(f"       {seq['sequence'][:100]}...\n" if len(seq['sequence']) > 100 else f"       {seq['sequence']}\n")
        
        logger.info(f"Saved per-neuron files to: {seq_dir}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Analyzed {len(neuron_ids)} neurons from layer {layer_id}")
        logger.info(f"Results saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.exception(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

