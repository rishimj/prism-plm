#!/usr/bin/env python3
"""Activation Steering experiment script for ESM-2 models.

This script performs activation steering experiments by injecting concept vectors
into the ESM-2 residual stream and measuring the effects on model representations.

Usage:
    # Mode 1: Motif-based steering
    python scripts/run_activation_steering.py \
        --model facebook/esm2_t33_650M_UR50D \
        --layer-id 16 \
        --mode motif \
        --motif-pattern "C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H" \
        --target-sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQD..."

    # Mode 2: GO cluster-based steering
    python scripts/run_activation_steering.py \
        --model facebook/esm2_t33_650M_UR50D \
        --layer-id 16 \
        --mode go_cluster \
        --prism-results outputs/neuron_batch/esm2_t33_650M_batch_xxx.json \
        --cluster-id 3

    # Mode 3: Explicit ID lists
    python scripts/run_activation_steering.py \
        --model facebook/esm2_t6_8M_UR50D \
        --layer-id 3 \
        --mode id_lists \
        --positive-ids positive_proteins.txt \
        --negative-ids negative_proteins.txt
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Set environment variables before imports
os.environ["TMPDIR"] = "/tmp"
os.environ["TMP"] = "/tmp"
os.environ["TEMP"] = "/tmp"
hf_cache_dir = "/tmp/huggingface_cache"
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steering import SteeringVector, SteeringHook
from src.steering.analysis import (
    compute_steering_effect_by_layer,
    generate_steering_report,
    summarize_steering_results,
)
from src.utils.logging_utils import setup_logging, log_system_info
from src.utils import helpers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Activation Steering for ESM-2 Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="Model name or path (default: facebook/esm2_t6_8M_UR50D)",
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        required=True,
        help="Layer to inject steering vector at",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model dtype (default: float32)",
    )
    parser.add_argument(
        "--use-logits",
        action="store_true",
        help="Enable logit/probability comparison metrics (requires AutoModelForMaskedLM)",
    )
    
    # Steering mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["motif", "go_cluster", "id_lists"],
        required=True,
        help="Mode for deriving the steering vector",
    )
    
    # Motif mode options
    parser.add_argument(
        "--motif-pattern",
        type=str,
        default=None,
        help="Regex pattern for motif (required for motif mode)",
    )
    
    # GO cluster mode options
    parser.add_argument(
        "--prism-results",
        type=Path,
        default=None,
        help="Path to PRISM results JSON (required for go_cluster mode)",
    )
    parser.add_argument(
        "--cluster-id",
        type=int,
        default=None,
        help="Cluster ID to use as positive set (required for go_cluster mode)",
    )
    
    # ID list mode options
    parser.add_argument(
        "--positive-ids",
        type=Path,
        default=None,
        help="Path to file with positive protein IDs (required for id_lists mode)",
    )
    parser.add_argument(
        "--negative-ids",
        type=Path,
        default=None,
        help="Path to file with negative protein IDs (required for id_lists mode)",
    )
    
    # Target sequence for steering experiment
    parser.add_argument(
        "--target-sequence",
        type=str,
        default=None,
        help="Target protein sequence to steer (if not provided, uses random from dataset)",
    )
    parser.add_argument(
        "--target-id",
        type=str,
        default=None,
        help="Target protein ID to steer (alternative to --target-sequence)",
    )
    
    # Steering parameters
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 5.0],
        help="Multiplier values to test (default: 0.0 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument(
        "--n-positive",
        type=int,
        default=100,
        help="Number of positive samples for vector derivation (default: 100)",
    )
    parser.add_argument(
        "--n-negative",
        type=int,
        default=100,
        help="Number of negative samples for vector derivation (default: 100)",
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default="lhallee/SwissProt",
        help="HuggingFace dataset for sequences (default: lhallee/SwissProt)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum sequences to load from dataset (default: 5000)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/steering"),
        help="Output directory for results (default: outputs/steering)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for the experiment (default: auto-generated)",
    )
    parser.add_argument(
        "--save-vector",
        action="store_true",
        help="Save the computed steering vector",
    )
    parser.add_argument(
        "--load-vector",
        type=Path,
        default=None,
        help="Load a pre-computed steering vector instead of deriving one",
    )
    
    # Misc
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    return parser.parse_args()


def load_sequences(dataset_name: str, max_samples: int) -> List[dict]:
    """Load protein sequences from a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        max_samples: Maximum number of samples to load
        
    Returns:
        List of sequence dicts with 'id' and 'sequence' keys
    """
    from src.data import HuggingFaceProteinLoader
    
    # Determine column names based on dataset
    if "swissprot" in dataset_name.lower():
        sequence_column = "Sequence"
        id_column = "Entry"
    else:
        sequence_column = "sequence"
        id_column = "id"
    
    loader = HuggingFaceProteinLoader(
        hf_dataset_name=dataset_name,
        hf_split="train",
        streaming=True,
        max_samples=max_samples,
        sequence_column=sequence_column,
        id_column=id_column,
    )
    
    return list(loader.load())


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate mode-specific arguments
    if args.mode == "motif" and not args.motif_pattern:
        print("Error: --motif-pattern required for motif mode")
        return 1
    
    if args.mode == "go_cluster":
        if not args.prism_results:
            print("Error: --prism-results required for go_cluster mode")
            return 1
        if args.cluster_id is None:
            print("Error: --cluster-id required for go_cluster mode")
            return 1
    
    if args.mode == "id_lists":
        if not args.positive_ids or not args.negative_ids:
            print("Error: --positive-ids and --negative-ids required for id_lists mode")
            return 1
    
    # Setup
    helpers.set_seed(args.seed)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        model_short = args.model.split("/")[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.experiment_name = f"steering_{model_short}_layer{args.layer_id}_{args.mode}_{timestamp}"
    
    # Setup logging
    logger, log_file = setup_logging(
        {"experiment_name": args.experiment_name},
        mode="steering",
        verbose=args.verbose,
    )
    
    log_system_info(logger)
    
    logger.info("=" * 80)
    logger.info("ACTIVATION STEERING EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Layer: {args.layer_id}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Multipliers: {args.multipliers}")
    
    try:
        # Setup device
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")
        
        if torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Load model and tokenizer
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING MODEL")
        logger.info("=" * 80)
        
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[args.dtype]
        
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        
        # Use AutoModelForMaskedLM if logits are needed, otherwise AutoModel
        if args.use_logits:
            logger.info("Using AutoModelForMaskedLM to enable logit comparison")
            model = AutoModelForMaskedLM.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        else:
            model = AutoModel.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded: {args.model}")
        logger.info(f"Hidden size: {model.config.hidden_size}")
        logger.info(f"Num layers: {model.config.num_hidden_layers}")
        
        # Validate layer ID
        if args.layer_id >= model.config.num_hidden_layers:
            raise ValueError(
                f"Layer {args.layer_id} out of range. "
                f"Model has {model.config.num_hidden_layers} layers."
            )
        
        # Load or derive steering vector
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 2: DERIVING STEERING VECTOR")
        logger.info("=" * 80)
        
        if args.load_vector:
            logger.info(f"Loading pre-computed steering vector from {args.load_vector}")
            steering_vec = SteeringVector.load(args.load_vector, model, tokenizer, device)
        else:
            # Load sequences for vector derivation
            logger.info(f"Loading sequences from {args.dataset}...")
            sequences = load_sequences(args.dataset, args.max_samples)
            logger.info(f"Loaded {len(sequences)} sequences")
            
            # Create steering vector extractor
            steering_vec = SteeringVector(
                model=model,
                tokenizer=tokenizer,
                layer_id=args.layer_id,
                device=device,
                aggregation="mean",
                max_length=args.max_length,
            )
            
            # Derive vector based on mode
            if args.mode == "motif":
                logger.info(f"Deriving vector from motif pattern: {args.motif_pattern}")
                steering_vec.from_motif(
                    sequences=sequences,
                    motif_pattern=args.motif_pattern,
                    n_positive=args.n_positive,
                    n_negative=args.n_negative,
                    random_seed=args.seed,
                )
            
            elif args.mode == "go_cluster":
                logger.info(f"Deriving vector from PRISM cluster {args.cluster_id}")
                steering_vec.from_go_clusters(
                    prism_results_path=str(args.prism_results),
                    target_cluster_id=args.cluster_id,
                    sequences=sequences,
                    n_positive=args.n_positive,
                    n_negative=args.n_negative,
                    random_seed=args.seed,
                )
            
            elif args.mode == "id_lists":
                logger.info("Deriving vector from ID lists")
                steering_vec.from_id_lists(
                    positive_ids_path=str(args.positive_ids),
                    negative_ids_path=str(args.negative_ids),
                    sequences=sequences,
                )
            
            # Save vector if requested
            if args.save_vector:
                vector_path = args.output_dir / f"{args.experiment_name}_vector.pt"
                steering_vec.save(str(vector_path))
        
        logger.info(f"Steering vector: dim={steering_vec.hidden_dim}, "
                   f"norm={torch.norm(steering_vec.vector).item():.4f}")
        
        # Prepare target sequence
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 3: PREPARING TARGET SEQUENCE")
        logger.info("=" * 80)
        
        if args.target_sequence:
            target_seq = args.target_sequence
            target_id = "user_provided"
        elif args.target_id:
            # Find sequence by ID
            seq_lookup = {s.get("id"): s for s in sequences}
            if args.target_id not in seq_lookup:
                raise ValueError(f"Target ID {args.target_id} not found in dataset")
            target_seq = seq_lookup[args.target_id].get("sequence", "")
            target_id = args.target_id
        else:
            # Use a random sequence from the dataset
            np.random.seed(args.seed + 1)
            target_idx = np.random.randint(len(sequences))
            target_seq = sequences[target_idx].get("sequence", "")
            target_id = sequences[target_idx].get("id", f"sample_{target_idx}")
        
        logger.info(f"Target sequence ID: {target_id}")
        logger.info(f"Target sequence length: {len(target_seq)}")
        
        # Tokenize target
        encoded = tokenizer(
            target_seq,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        logger.info(f"Tokenized length: {input_ids.shape[1]}")
        
        # Run steering experiment
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: RUNNING STEERING EXPERIMENT")
        logger.info("=" * 80)
        
        # Create concept evaluator if motif pattern provided
        concept_evaluator = None
        if args.mode == "motif" and args.motif_pattern:
            from src.steering.analysis import create_motif_evaluator
            concept_evaluator = create_motif_evaluator(args.motif_pattern)
            logger.info("Concept probability shift evaluation enabled for motif")
        
        results = compute_steering_effect_by_layer(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            steering_vector=steering_vec.vector,
            target_layer=args.layer_id,
            multipliers=args.multipliers,
            use_logits=args.use_logits,
            tokenizer=tokenizer if concept_evaluator else None,
            concept_evaluator=concept_evaluator,
            concept_eval_method="masked_token",  # Can be "masked_token", "token_probability", or "sequence_level"
        )
        
        # Add metadata to results
        results["metadata"] = {
            "model": args.model,
            "layer_id": args.layer_id,
            "mode": args.mode,
            "target_id": target_id,
            "target_seq_length": len(target_seq),
            "steering_vector_norm": torch.norm(steering_vec.vector).item(),
            "multipliers": args.multipliers,
            "experiment_name": args.experiment_name,
        }
        
        if args.mode == "motif":
            results["metadata"]["motif_pattern"] = args.motif_pattern
        elif args.mode == "go_cluster":
            results["metadata"]["prism_results"] = str(args.prism_results)
            results["metadata"]["cluster_id"] = args.cluster_id
        
        # Generate report
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 5: GENERATING REPORT")
        logger.info("=" * 80)
        
        report_path = generate_steering_report(
            results=results,
            output_dir=str(args.output_dir),
            experiment_name=args.experiment_name,
        )
        
        # Print summary
        summary = summarize_steering_results(results)
        logger.info("\n" + summary)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {report_path}")
        logger.info(f"Visualization saved to: {args.output_dir / f'{args.experiment_name}.png'}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error during experiment: {e}")
        return 1
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    sys.exit(main())

