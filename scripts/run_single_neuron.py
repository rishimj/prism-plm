#!/usr/bin/env python3
"""Single neuron feature description for PRISM-Bio.

This script analyzes ONE specific neuron in a protein language model:
1. Pass protein sequences through the model
2. Extract activations for the specified neuron (layer + unit)
3. Find top-k highly activating sequences
4. Embed those sequences using the model's full representation
5. Cluster the embeddings to find sub-patterns
6. Compute polysemanticity (cosine similarities between cluster centroids)

This mirrors the structure of prism/src/feature_description.py but for proteins.

Usage:
    python scripts/run_single_neuron.py --layer-id 18 --unit-id 100
    python scripts/run_single_neuron.py --layer-id 18 --unit-id 100 --config configs/models/esm2_3b.yaml
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
        description="PRISM-Bio Single Neuron Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        required=True,
        help="Layer index to analyze",
    )
    parser.add_argument(
        "--unit-id",
        type=int,
        required=True,
        help="Neuron/unit index within the layer",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top-activating sequences to analyze",
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

    return parser.parse_args()


def extract_neuron_activations(
    model,
    dataloader,
    layer_id: int,
    unit_id: int,
    device,
    aggregation: str = "mean",
    logger=None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Extract activations for a specific neuron and full embeddings for clustering.
    
    Returns:
        neuron_activations: 1D array of neuron activations per sequence
        full_embeddings: 2D array of full embeddings for clustering
        metadata: List of dicts with sequence info
    """
    neuron_activations = []
    full_embeddings = []
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
            hidden_states = outputs.hidden_states[layer_id]  # Shape: (batch, seq_len, hidden_dim)
            
            # Aggregate over sequence
            if aggregation == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                aggregated = sum_embeddings / sum_mask  # Shape: (batch, hidden_dim)
            elif aggregation == "first":
                aggregated = hidden_states[:, 0]
            elif aggregation == "max":
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_masked = hidden_states.clone()
                hidden_masked[~mask_expanded.bool()] = float('-inf')
                aggregated = hidden_masked.max(dim=1)[0]
            else:
                aggregated = hidden_states.mean(dim=1)
            
            # Extract neuron-specific activation (single dimension)
            neuron_acts = aggregated[:, unit_id].cpu().numpy()  # Shape: (batch,)
            neuron_activations.extend(neuron_acts.tolist())
            
            # Store full embeddings for clustering
            full_embeddings.append(aggregated.cpu().numpy())
            
            # Store metadata
            for i, seq_id in enumerate(batch["sequence_id"]):
                metadata.append({
                    "id": seq_id,
                    "sequence": batch.get("sequence", [""] * len(batch["sequence_id"]))[i] if "sequence" in batch else "",
                })
            
            # Clear GPU cache periodically
            if batch_idx % 5 == 0:
                helpers.clear_gpu_cache(device)
    
    neuron_activations = np.array(neuron_activations)
    full_embeddings = np.vstack(full_embeddings)
    
    # Handle inf/nan values (can occur with float16)
    if np.any(~np.isfinite(full_embeddings)):
        n_invalid = np.sum(~np.isfinite(full_embeddings))
        if logger:
            logger.warning(f"Found {n_invalid} inf/nan values in embeddings. Replacing with 0.")
        full_embeddings = np.nan_to_num(full_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        neuron_activations = np.nan_to_num(neuron_activations, nan=0.0, posinf=0.0, neginf=0.0)
    
    return neuron_activations, full_embeddings, metadata


def compute_polysemanticity(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute polysemanticity score based on cluster centroid similarities.
    
    Lower average similarity = more distinct clusters = potentially more polysemantic neuron
    (neuron responds to multiple distinct concepts)
    """
    unique_labels = [l for l in np.unique(labels) if l != -1]
    
    if len(unique_labels) < 2:
        return {
            "cluster_centroids": {},
            "similarity_matrix": [],
            "per_cluster_avg": {},
            "overall_avg": 1.0,
            "polysemanticity_score": 0.0,
        }
    
    # Compute centroids
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[int(label)] = embeddings[mask].mean(axis=0)
    
    # Stack centroids for similarity computation
    centroid_matrix = np.vstack([centroids[l] for l in sorted(centroids.keys())])
    
    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(centroid_matrix)
    
    # Compute per-cluster average (excluding self-similarity)
    per_cluster_avg = {}
    for i, label in enumerate(sorted(centroids.keys())):
        other_sims = [sim_matrix[i, j] for j in range(len(centroids)) if i != j]
        per_cluster_avg[int(label)] = float(np.mean(other_sims)) if other_sims else 1.0
    
    # Overall average similarity (excluding diagonal)
    n = len(centroids)
    if n > 1:
        overall_avg = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
    else:
        overall_avg = 1.0
    
    # Polysemanticity score: 1 - avg_similarity
    # Higher score = more polysemantic (clusters are more different)
    polysemanticity_score = 1.0 - overall_avg
    
    return {
        "cluster_centroids": {int(k): v.tolist() for k, v in centroids.items()},
        "similarity_matrix": sim_matrix.tolist(),
        "per_cluster_avg": per_cluster_avg,
        "overall_avg": float(overall_avg),
        "polysemanticity_score": float(polysemanticity_score),
    }


def main():
    """Main entry point for single neuron analysis."""
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
        mode=f"single_neuron_L{args.layer_id}_U{args.unit_id}",
        verbose=config.verbose,
    )
    
    log_system_info(logger)
    helpers.set_seed(config.random_seed)
    
    # Set CUDA config
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logger.info("=" * 80)
    logger.info("PRISM-BIO SINGLE NEURON ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Layer: {args.layer_id}")
    logger.info(f"Unit/Neuron: {args.unit_id}")
    logger.info(f"Top-k sequences: {args.top_k}")
    logger.info(f"Number of clusters: {args.n_clusters}")
    logger.info(f"Max samples: {config.dataset.max_samples}")
    
    try:
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
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            token=hf_token,
        )
        
        model = AutoModel.from_pretrained(
            config.model.model_name,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded: {config.model.model_name}")
        
        # Step 3: Create dataloader
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 3: EXTRACTING ACTIVATIONS")
        logger.info("=" * 80)
        
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
        
        # Extract neuron activations and full embeddings
        neuron_activations, full_embeddings, metadata = extract_neuron_activations(
            model=model,
            dataloader=dataloader,
            layer_id=args.layer_id,
            unit_id=args.unit_id,
            device=device,
            aggregation=config.model.aggregation,
            logger=logger,
        )
        
        logger.info(f"Extracted activations for neuron {args.unit_id} in layer {args.layer_id}")
        logger.info(f"  Neuron activations shape: {neuron_activations.shape}")
        logger.info(f"  Full embeddings shape: {full_embeddings.shape}")
        logger.info(f"  Activation stats: mean={neuron_activations.mean():.4f}, std={neuron_activations.std():.4f}")
        
        # Free model memory
        del model
        helpers.clear_gpu_cache(device)
        gc.collect()
        
        # Step 4: Select top-k activating sequences
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: SELECTING TOP ACTIVATING SEQUENCES")
        logger.info("=" * 80)
        
        top_k = min(args.top_k, len(neuron_activations))
        top_indices = np.argsort(neuron_activations)[-top_k:][::-1]
        
        top_activations = neuron_activations[top_indices]
        top_embeddings = full_embeddings[top_indices]
        top_metadata = [metadata[i] for i in top_indices]
        
        # Add original sequence data
        for i, idx in enumerate(top_indices):
            seq_id = top_metadata[i]["id"]
            seq_data = next((s for s in sequences if s.get("id") == seq_id), None)
            if seq_data:
                top_metadata[i]["sequence"] = seq_data.get("sequence", "")
        
        logger.info(f"Selected top {top_k} activating sequences")
        logger.info(f"  Top activation: {top_activations[0]:.4f}")
        logger.info(f"  Bottom of top-k: {top_activations[-1]:.4f}")
        logger.info(f"  Mean of top-k: {top_activations.mean():.4f}")
        
        # Step 5: Cluster the top-activating sequences
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 5: CLUSTERING TOP SEQUENCES")
        logger.info("=" * 80)
        
        from src.analysis.clustering import cluster_embeddings, compute_cluster_statistics
        
        n_clusters = min(args.n_clusters, top_k)
        
        labels = cluster_embeddings(
            top_embeddings,
            method=config.clustering.algorithm,
            n_clusters=n_clusters,
            seed=config.random_seed,
        )
        
        cluster_stats = compute_cluster_statistics(top_embeddings, labels)
        logger.info(f"Clustering complete. Found {len(cluster_stats)} clusters.")
        
        # Organize sequences by cluster
        cluster_sequences = {}
        for i, label in enumerate(labels):
            if label not in cluster_sequences:
                cluster_sequences[label] = []
            cluster_sequences[label].append({
                "id": top_metadata[i]["id"],
                "sequence": top_metadata[i].get("sequence", ""),
                "activation": float(top_activations[i]),
                "rank": i + 1,
            })
        
        for cluster_id, seqs in cluster_sequences.items():
            logger.info(f"  Cluster {cluster_id}: {len(seqs)} sequences, "
                       f"mean activation: {np.mean([s['activation'] for s in seqs]):.4f}")
        
        # Step 6: Compute polysemanticity
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 6: COMPUTING POLYSEMANTICITY")
        logger.info("=" * 80)
        
        polysemanticity = compute_polysemanticity(top_embeddings, labels)
        
        logger.info(f"Polysemanticity analysis:")
        logger.info(f"  Overall cluster similarity: {polysemanticity['overall_avg']:.4f}")
        logger.info(f"  Polysemanticity score: {polysemanticity['polysemanticity_score']:.4f}")
        logger.info(f"  (Higher score = more distinct clusters = neuron responds to multiple concepts)")
        
        for cluster_id, avg_sim in polysemanticity["per_cluster_avg"].items():
            logger.info(f"  Cluster {cluster_id} avg similarity to others: {avg_sim:.4f}")
        
        # Step 7: Save results
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 7: SAVING RESULTS")
        logger.info("=" * 80)
        
        output_dir = Path(config.output_dir) / "single_neuron"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_short = config.model.model_name.split("/")[-1]
        base_name = f"{model_short}_layer-{args.layer_id}_unit-{args.unit_id}_{timestamp}"
        
        # Save CSV (matches prism/src/feature_description.py output format)
        csv_path = output_dir / f"{base_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer", "unit", "cluster", "n_sequences", "mean_activation", 
                "polysemanticity_score", "cluster_similarity", "highlights"
            ])
            
            for cluster_id in sorted(cluster_sequences.keys()):
                seqs = cluster_sequences[cluster_id]
                seq_ids = [s["id"] for s in seqs]
                mean_act = np.mean([s["activation"] for s in seqs])
                cluster_sim = polysemanticity["per_cluster_avg"].get(int(cluster_id), 1.0)
                
                writer.writerow([
                    args.layer_id,
                    args.unit_id,
                    cluster_id,
                    len(seqs),
                    f"{mean_act:.6f}",
                    f"{polysemanticity['polysemanticity_score']:.6f}",
                    f"{cluster_sim:.6f}",
                    str(seq_ids),
                ])
        
        logger.info(f"Saved CSV to: {csv_path}")
        
        # Save detailed JSON
        json_path = output_dir / f"{base_name}.json"
        
        result = {
            "metadata": {
                "model": config.model.model_name,
                "layer_id": args.layer_id,
                "unit_id": args.unit_id,
                "top_k": top_k,
                "n_clusters": n_clusters,
                "n_sequences_total": len(neuron_activations),
                "timestamp": timestamp,
            },
            "neuron_stats": {
                "mean_activation": float(neuron_activations.mean()),
                "std_activation": float(neuron_activations.std()),
                "max_activation": float(neuron_activations.max()),
                "min_activation": float(neuron_activations.min()),
                "top_k_mean": float(top_activations.mean()),
            },
            "polysemanticity": {
                "score": polysemanticity["polysemanticity_score"],
                "overall_cluster_similarity": polysemanticity["overall_avg"],
                "per_cluster_similarity": polysemanticity["per_cluster_avg"],
                "similarity_matrix": polysemanticity["similarity_matrix"],
            },
            "clusters": {
                int(k): {
                    "n_sequences": len(v),
                    "mean_activation": float(np.mean([s["activation"] for s in v])),
                    "sequences": v,
                }
                for k, v in cluster_sequences.items()
            },
        }
        
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved JSON to: {json_path}")
        
        # Save per-cluster sequence files
        for cluster_id, seqs in cluster_sequences.items():
            cluster_file = output_dir / f"{base_name}_cluster-{cluster_id}.txt"
            with open(cluster_file, "w") as f:
                f.write(f"LAYER {args.layer_id} - UNIT {args.unit_id} - CLUSTER {cluster_id}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Number of sequences: {len(seqs)}\n")
                f.write(f"Mean activation: {np.mean([s['activation'] for s in seqs]):.6f}\n")
                f.write(f"Cluster similarity to others: {polysemanticity['per_cluster_avg'].get(int(cluster_id), 1.0):.4f}\n")
                f.write("\n")
                f.write("SEQUENCES:\n")
                f.write("-" * 80 + "\n")
                for seq in seqs:
                    f.write(f"  {seq['rank']:3d}. [{seq['activation']:.4f}] {seq['id']}\n")
                    seq_str = seq['sequence']
                    if len(seq_str) > 100:
                        seq_str = seq_str[:50] + "..." + seq_str[-50:]
                    f.write(f"       {seq_str}\n")
        
        logger.info(f"Saved cluster files to: {output_dir}")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Layer: {args.layer_id}, Unit: {args.unit_id}")
        logger.info(f"Polysemanticity score: {polysemanticity['polysemanticity_score']:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

