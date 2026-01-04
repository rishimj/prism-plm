#!/usr/bin/env python3
"""Batch neuron analysis for PRISM-Bio.

This script runs single neuron analysis across MANY neurons to build a comprehensive
sample space, similar to how prism/src/evaluation.py processes multiple layer-unit pairs.

It can:
1. Process a list of specific neurons from a CSV file
2. Auto-select neurons based on variance/activity
3. Process a range of neurons

Usage:
    # Process neurons listed in a CSV file (with columns: layer, unit)
    python scripts/run_neuron_batch.py --neuron-file neurons_to_analyze.csv

    # Auto-select top 50 most variable neurons from layer 18
    python scripts/run_neuron_batch.py --layer-id 18 --auto-select 50 --selection-method variance

    # Process neurons 0-99 in layer 18
    python scripts/run_neuron_batch.py --layer-id 18 --unit-range 0 99

    # Process specific neurons
    python scripts/run_neuron_batch.py --layer-id 18 --unit-ids 10 50 100 200 300
"""
import csv
import gc
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json

# Set temp directory and HuggingFace cache
os.environ["TMPDIR"] = "/tmp"
os.environ["TMP"] = "/tmp"
os.environ["TEMP"] = "/tmp"
hf_cache_dir = "/tmp/huggingface_cache"
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.utils.logging_utils import setup_logging, log_system_info
from src.utils import constants, helpers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PRISM-Bio Batch Neuron Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    
    # Neuron selection options (mutually exclusive groups)
    neuron_source = parser.add_mutually_exclusive_group(required=True)
    neuron_source.add_argument(
        "--neuron-file",
        type=Path,
        help="CSV file with 'layer' and 'unit' columns specifying neurons to analyze",
    )
    neuron_source.add_argument(
        "--layer-id",
        type=int,
        help="Layer ID (use with --unit-ids, --unit-range, or --auto-select)",
    )
    
    # Unit selection (when using --layer-id)
    parser.add_argument(
        "--unit-ids",
        type=int,
        nargs="+",
        help="Specific unit IDs to analyze",
    )
    parser.add_argument(
        "--unit-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of unit IDs to analyze (inclusive)",
    )
    parser.add_argument(
        "--auto-select",
        type=int,
        help="Auto-select N neurons based on selection method",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        choices=["variance", "mean", "random"],
        default="variance",
        help="Method for auto-selecting neurons",
    )
    
    # Analysis parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top-activating sequences per neuron",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters for polysemanticity analysis",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of sequences to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual neuron results (in addition to summary)",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=None,
        help="Reduce embeddings to this dimensionality via PCA before clustering (normalizes across models)",
    )

    return parser.parse_args()


def extract_all_neuron_activations(
    model,
    dataloader,
    layer_id: int,
    device,
    aggregation: str = "mean",
    logger=None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Extract activations for ALL neurons in a layer and full embeddings.
    
    Returns:
        all_activations: 2D array (n_sequences, n_neurons) of all neuron activations
        full_embeddings: 2D array (n_sequences, hidden_dim) for clustering
        metadata: List of dicts with sequence info
    """
    all_activations = []
    metadata = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if logger and batch_idx % 10 == 0:
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
            
            # Aggregate over sequence
            if aggregation == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                aggregated = sum_embeddings / sum_mask
            elif aggregation == "first":
                aggregated = hidden_states[:, 0]
            elif aggregation == "max":
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_masked = hidden_states.clone()
                hidden_masked[~mask_expanded.bool()] = float('-inf')
                aggregated = hidden_masked.max(dim=1)[0]
            else:
                aggregated = hidden_states.mean(dim=1)
            
            all_activations.append(aggregated.cpu().numpy())
            
            # Store metadata
            for i, seq_id in enumerate(batch["sequence_id"]):
                metadata.append({
                    "id": seq_id,
                    "sequence": batch.get("sequence", [""] * len(batch["sequence_id"]))[i] if "sequence" in batch else "",
                })
            
            if batch_idx % 5 == 0:
                helpers.clear_gpu_cache(device)
    
    all_activations = np.vstack(all_activations)  # Shape: (n_sequences, hidden_dim)
    
    # Handle inf/nan values (can occur with float16)
    if np.any(~np.isfinite(all_activations)):
        n_invalid = np.sum(~np.isfinite(all_activations))
        if logger:
            logger.warning(f"Found {n_invalid} inf/nan values in activations. Replacing with 0.")
        all_activations = np.nan_to_num(all_activations, nan=0.0, posinf=0.0, neginf=0.0)
    
    return all_activations, metadata


def select_neurons_to_analyze(
    all_activations: np.ndarray,
    args,
    logger,
) -> List[int]:
    """Select which neurons to analyze based on CLI arguments."""
    n_neurons = all_activations.shape[1]
    
    if args.unit_ids:
        # Specific unit IDs
        unit_ids = [u for u in args.unit_ids if 0 <= u < n_neurons]
        if len(unit_ids) != len(args.unit_ids):
            logger.warning(f"Some unit IDs out of range (max: {n_neurons - 1})")
        return unit_ids
    
    elif args.unit_range:
        start, end = args.unit_range
        start = max(0, start)
        end = min(n_neurons - 1, end)
        return list(range(start, end + 1))
    
    elif args.auto_select:
        n_select = min(args.auto_select, n_neurons)
        
        if args.selection_method == "variance":
            variances = np.var(all_activations, axis=0)
            selected = np.argsort(variances)[-n_select:][::-1]
            logger.info(f"Selected top {n_select} neurons by variance")
        elif args.selection_method == "mean":
            means = np.mean(all_activations, axis=0)
            selected = np.argsort(means)[-n_select:][::-1]
            logger.info(f"Selected top {n_select} neurons by mean activation")
        else:  # random
            np.random.seed(42)
            selected = np.random.choice(n_neurons, size=n_select, replace=False)
            logger.info(f"Randomly selected {n_select} neurons")
        
        return selected.tolist()
    
    else:
        # Default: top 10 by variance
        variances = np.var(all_activations, axis=0)
        selected = np.argsort(variances)[-10:][::-1]
        logger.info(f"Default: selected top 10 neurons by variance")
        return selected.tolist()


def analyze_single_neuron(
    neuron_activations: np.ndarray,
    full_embeddings: np.ndarray,
    metadata: List[Dict],
    sequences: List[Dict],
    layer_id: int,
    unit_id: int,
    top_k: int,
    n_clusters: int,
    config,
    logger,
    pca_dims: Optional[int] = None,
    pca_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Analyze a single neuron and return results."""
    from src.analysis.clustering import cluster_embeddings, compute_cluster_statistics
    
    # Select top-k activating sequences
    top_k = min(top_k, len(neuron_activations))
    top_indices = np.argsort(neuron_activations)[-top_k:][::-1]
    
    top_activations = neuron_activations[top_indices]
    top_embeddings = full_embeddings[top_indices]
    top_metadata = [metadata[i] for i in top_indices]
    
    # Handle inf/nan in top embeddings before clustering
    if np.any(~np.isfinite(top_embeddings)):
        top_embeddings = np.nan_to_num(top_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(~np.isfinite(top_activations)):
        top_activations = np.nan_to_num(top_activations, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply PCA dimensionality reduction if specified (normalizes across models)
    if pca_dims is not None and pca_model is not None:
        top_embeddings = pca_model.transform(top_embeddings)
    
    # Add sequence data
    for i, idx in enumerate(top_indices):
        seq_id = top_metadata[i]["id"]
        seq_data = next((s for s in sequences if s.get("id") == seq_id), None)
        if seq_data:
            top_metadata[i]["sequence"] = seq_data.get("sequence", "")
    
    # Cluster
    n_clust = min(n_clusters, top_k)
    labels = cluster_embeddings(
        top_embeddings,
        method=config.clustering.algorithm,
        n_clusters=n_clust,
        seed=config.random_seed,
    )
    
    # Organize by cluster
    cluster_sequences = {}
    for i, label in enumerate(labels):
        if label not in cluster_sequences:
            cluster_sequences[label] = []
        cluster_sequences[label].append({
            "id": top_metadata[i]["id"],
            "sequence": top_metadata[i].get("sequence", ""),
            "activation": float(top_activations[i]),
        })
    
    # Compute polysemanticity
    unique_labels = [l for l in np.unique(labels) if l != -1]
    
    if len(unique_labels) >= 2:
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[int(label)] = top_embeddings[mask].mean(axis=0)
        
        centroid_matrix = np.vstack([centroids[l] for l in sorted(centroids.keys())])
        sim_matrix = cosine_similarity(centroid_matrix)
        
        n = len(centroids)
        overall_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
        polysemanticity_score = 1.0 - overall_sim
        
        per_cluster_sim = {}
        for i, label in enumerate(sorted(centroids.keys())):
            other_sims = [sim_matrix[i, j] for j in range(n) if i != j]
            per_cluster_sim[int(label)] = float(np.mean(other_sims)) if other_sims else 1.0
    else:
        overall_sim = 1.0
        polysemanticity_score = 0.0
        per_cluster_sim = {}
    
    return {
        "layer_id": layer_id,
        "unit_id": unit_id,
        "n_sequences_total": len(neuron_activations),
        "neuron_stats": {
            "mean": float(neuron_activations.mean()),
            "std": float(neuron_activations.std()),
            "max": float(neuron_activations.max()),
            "min": float(neuron_activations.min()),
            "top_k_mean": float(top_activations.mean()),
        },
        "polysemanticity": {
            "score": polysemanticity_score,
            "overall_similarity": overall_sim,
            "per_cluster_similarity": per_cluster_sim,
        },
        "clusters": {
            int(k): {
                "n_sequences": len(v),
                "mean_activation": float(np.mean([s["activation"] for s in v])),
                "sequence_ids": [s["id"] for s in v],
            }
            for k, v in cluster_sequences.items()
        },
    }


def main():
    """Main entry point for batch neuron analysis."""
    args = parse_args()
    
    # Build CLI overrides
    cli_overrides = {}
    if args.output_dir:
        cli_overrides["output_dir"] = str(args.output_dir)
    if args.verbose:
        cli_overrides["verbose"] = True
    if args.max_samples:
        cli_overrides["dataset"] = {"max_samples": args.max_samples}
    
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
        mode="neuron_batch",
        verbose=config.verbose,
    )
    
    log_system_info(logger)
    helpers.set_seed(config.random_seed)
    
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logger.info("=" * 80)
    logger.info("PRISM-BIO BATCH NEURON ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Top-k sequences per neuron: {args.top_k}")
    logger.info(f"Number of clusters: {args.n_clusters}")
    
    try:
        # Determine neurons to analyze
        if args.neuron_file:
            # Load from CSV
            neuron_df = pd.read_csv(args.neuron_file)
            if "layer" not in neuron_df.columns or "unit" not in neuron_df.columns:
                logger.error("Neuron file must have 'layer' and 'unit' columns")
                return 1
            neurons_to_analyze = list(zip(neuron_df["layer"], neuron_df["unit"]))
            logger.info(f"Loaded {len(neurons_to_analyze)} neurons from {args.neuron_file}")
        else:
            # Will determine after extracting activations
            neurons_to_analyze = None
            layer_id = args.layer_id
            logger.info(f"Will analyze neurons from layer {layer_id}")
        
        # Step 1: Load dataset
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING DATASET")
        logger.info("=" * 80)
        
        from src.data import HuggingFaceProteinLoader, ProteinDataset, ProteinCollator
        
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
        
        from transformers import AutoTokenizer, AutoModel
        
        device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(config.model.dtype, torch.float32)
        
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, token=hf_token)
        model = AutoModel.from_pretrained(config.model.model_name, torch_dtype=torch_dtype, token=hf_token)
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded: {config.model.model_name}")
        
        # Step 3: Create dataloader
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
        
        # Step 4: Extract activations
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 3: EXTRACTING ACTIVATIONS")
        logger.info("=" * 80)
        
        if neurons_to_analyze:
            # Multiple layers - need to extract per layer
            unique_layers = list(set([n[0] for n in neurons_to_analyze]))
            layer_activations = {}
            
            for lid in unique_layers:
                logger.info(f"Extracting activations for layer {lid}")
                activations, metadata = extract_all_neuron_activations(
                    model, dataloader, lid, device, config.model.aggregation, logger
                )
                layer_activations[lid] = activations
                logger.info(f"  Shape: {activations.shape}")
        else:
            # Single layer
            all_activations, metadata = extract_all_neuron_activations(
                model, dataloader, layer_id, device, config.model.aggregation, logger
            )
            logger.info(f"Extracted activations shape: {all_activations.shape}")
            
            # Select neurons to analyze
            unit_ids = select_neurons_to_analyze(all_activations, args, logger)
            neurons_to_analyze = [(layer_id, u) for u in unit_ids]
            layer_activations = {layer_id: all_activations}
        
        logger.info(f"Will analyze {len(neurons_to_analyze)} neurons")
        
        # Free model memory
        del model
        helpers.clear_gpu_cache(device)
        gc.collect()
        
        # Step 5: Analyze each neuron
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: ANALYZING NEURONS")
        logger.info("=" * 80)
        
        # Initialize PCA for dimensionality reduction if specified
        pca_models = {}
        if args.pca_dims:
            from sklearn.decomposition import PCA
            logger.info(f"Using PCA reduction to {args.pca_dims} dimensions for normalized comparison")
            for lid in layer_activations.keys():
                activations = layer_activations[lid]
                n_components = min(args.pca_dims, activations.shape[1], activations.shape[0])
                pca = PCA(n_components=n_components, random_state=config.random_seed)
                pca.fit(activations)
                pca_models[lid] = pca
                logger.info(f"  Layer {lid}: {activations.shape[1]}D -> {n_components}D (variance explained: {pca.explained_variance_ratio_.sum():.2%})")
        
        all_results = []
        
        for idx, (lid, uid) in enumerate(neurons_to_analyze):
            logger.info(f"Analyzing layer {lid}, unit {uid} ({idx + 1}/{len(neurons_to_analyze)})")
            
            # Get activations for this layer
            activations = layer_activations[lid]
            neuron_activations = activations[:, uid]
            
            result = analyze_single_neuron(
                neuron_activations=neuron_activations,
                full_embeddings=activations,
                metadata=metadata,
                sequences=sequences,
                layer_id=lid,
                unit_id=uid,
                top_k=args.top_k,
                n_clusters=args.n_clusters,
                config=config,
                logger=logger,
                pca_dims=args.pca_dims,
                pca_model=pca_models.get(lid) if args.pca_dims else None,
            )
            
            all_results.append(result)
            
            logger.info(f"  Polysemanticity score: {result['polysemanticity']['score']:.4f}")
            logger.info(f"  Mean activation: {result['neuron_stats']['mean']:.4f}")
        
        # Step 6: Save results
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 5: SAVING RESULTS")
        logger.info("=" * 80)
        
        output_dir = Path(config.output_dir) / "neuron_batch"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_short = config.model.model_name.split("/")[-1]
        base_name = f"{model_short}_batch_{timestamp}"
        
        # Save summary CSV (like evaluation.py output)
        csv_path = output_dir / f"{base_name}_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer", "unit", "mean_activation", "std_activation", "max_activation",
                "top_k_mean", "polysemanticity_score", "overall_similarity", 
                "n_clusters", "cluster_sizes"
            ])
            
            for r in all_results:
                cluster_sizes = ",".join([str(c["n_sequences"]) for c in r["clusters"].values()])
                writer.writerow([
                    r["layer_id"],
                    r["unit_id"],
                    f"{r['neuron_stats']['mean']:.6f}",
                    f"{r['neuron_stats']['std']:.6f}",
                    f"{r['neuron_stats']['max']:.6f}",
                    f"{r['neuron_stats']['top_k_mean']:.6f}",
                    f"{r['polysemanticity']['score']:.6f}",
                    f"{r['polysemanticity']['overall_similarity']:.6f}",
                    len(r["clusters"]),
                    cluster_sizes,
                ])
        
        logger.info(f"Saved summary CSV to: {csv_path}")
        
        # Save full JSON
        json_path = output_dir / f"{base_name}_full.json"
        
        # Helper to convert numpy types
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
                "n_neurons_analyzed": len(all_results),
                "top_k": args.top_k,
                "n_clusters": args.n_clusters,
                "n_sequences": len(metadata),
                "pca_dims": args.pca_dims,
                "timestamp": timestamp,
            },
            "summary_stats": {
                "mean_polysemanticity": float(np.mean([r["polysemanticity"]["score"] for r in all_results])),
                "std_polysemanticity": float(np.std([r["polysemanticity"]["score"] for r in all_results])),
                "max_polysemanticity": float(np.max([r["polysemanticity"]["score"] for r in all_results])),
                "min_polysemanticity": float(np.min([r["polysemanticity"]["score"] for r in all_results])),
            },
            "neurons": convert_numpy(all_results),
        }
        
        with open(json_path, "w") as f:
            json.dump(convert_numpy(output_data), f, indent=2)
        
        logger.info(f"Saved full JSON to: {json_path}")
        
        # Save individual neuron results if requested
        if args.save_individual:
            individual_dir = output_dir / f"{base_name}_individual"
            individual_dir.mkdir(parents=True, exist_ok=True)
            
            for r in all_results:
                neuron_file = individual_dir / f"layer-{r['layer_id']}_unit-{r['unit_id']}.json"
                with open(neuron_file, "w") as f:
                    json.dump(convert_numpy(r), f, indent=2)
            
            logger.info(f"Saved individual results to: {individual_dir}")
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Analyzed {len(all_results)} neurons")
        logger.info(f"Mean polysemanticity: {output_data['summary_stats']['mean_polysemanticity']:.4f}")
        logger.info(f"Std polysemanticity: {output_data['summary_stats']['std_polysemanticity']:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
        # Show top 5 most polysemantic neurons
        sorted_results = sorted(all_results, key=lambda x: x["polysemanticity"]["score"], reverse=True)
        logger.info("")
        logger.info("Top 5 most polysemantic neurons:")
        for i, r in enumerate(sorted_results[:5]):
            logger.info(f"  {i+1}. Layer {r['layer_id']}, Unit {r['unit_id']}: score={r['polysemanticity']['score']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

