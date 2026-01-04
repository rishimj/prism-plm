"""Analysis utilities for comparing baseline and steered model outputs.

This module provides functions for computing metrics that measure
the effect of activation steering on model representations.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
import json
import re

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logging_utils import get_logger

logger = get_logger("steering.analysis")


def compare_hidden_states(
    baseline_states: List[torch.Tensor],
    steered_states: List[torch.Tensor],
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute per-layer metrics comparing baseline and steered hidden states.
    
    Args:
        baseline_states: List of tensors, one per layer, shape [batch, seq_len, hidden_dim]
        steered_states: List of tensors, one per layer, shape [batch, seq_len, hidden_dim]
        layer_names: Optional names for each layer
        
    Returns:
        Dictionary containing per-layer metrics:
        - cosine_similarity: Mean cosine similarity between baseline and steered
        - euclidean_distance: Mean euclidean distance
        - relative_change: Norm of difference / norm of baseline
    """
    if len(baseline_states) != len(steered_states):
        raise ValueError(
            f"Mismatched number of layers: {len(baseline_states)} vs {len(steered_states)}"
        )
    
    n_layers = len(baseline_states)
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(n_layers)]
    
    results = {
        "n_layers": n_layers,
        "per_layer": {},
        "summary": {},
    }
    
    all_cosine_sims = []
    all_euclidean_dists = []
    all_relative_changes = []
    
    for i, (baseline, steered) in enumerate(zip(baseline_states, steered_states)):
        layer_name = layer_names[i]
        
        # Mean pool over sequence for comparison
        # Shape: [batch, hidden_dim]
        baseline_pooled = baseline.mean(dim=1)
        steered_pooled = steered.mean(dim=1)
        
        # Cosine similarity (per sample, then average)
        cos_sim = F.cosine_similarity(baseline_pooled, steered_pooled, dim=-1)
        mean_cos_sim = cos_sim.mean().item()
        
        # Euclidean distance
        diff = steered_pooled - baseline_pooled
        euclidean_dist = torch.norm(diff, dim=-1).mean().item()
        
        # Relative change (how much did it change relative to baseline magnitude)
        baseline_norm = torch.norm(baseline_pooled, dim=-1).mean().item()
        relative_change = euclidean_dist / (baseline_norm + 1e-9)
        
        results["per_layer"][layer_name] = {
            "cosine_similarity": mean_cos_sim,
            "euclidean_distance": euclidean_dist,
            "relative_change": relative_change,
            "layer_idx": i,
        }
        
        all_cosine_sims.append(mean_cos_sim)
        all_euclidean_dists.append(euclidean_dist)
        all_relative_changes.append(relative_change)
    
    # Summary statistics
    results["summary"] = {
        "mean_cosine_similarity": np.mean(all_cosine_sims),
        "min_cosine_similarity": np.min(all_cosine_sims),
        "mean_euclidean_distance": np.mean(all_euclidean_dists),
        "max_euclidean_distance": np.max(all_euclidean_dists),
        "mean_relative_change": np.mean(all_relative_changes),
        "max_relative_change": np.max(all_relative_changes),
    }
    
    return results


def compare_logits(
    baseline_logits: torch.Tensor,
    steered_logits: torch.Tensor,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Compute metrics comparing baseline and steered logits.
    
    Args:
        baseline_logits: Logits from baseline run, shape [batch, seq_len, vocab_size]
        steered_logits: Logits from steered run, shape [batch, seq_len, vocab_size]
        top_k: Number of top predictions to compare
        
    Returns:
        Dictionary containing:
        - kl_divergence: Mean KL divergence between probability distributions
        - top_k_overlap: Fraction of top-k predictions that match
        - prediction_changes: Number of positions where argmax changed
    """
    results = {}
    
    # Convert to probabilities
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    steered_probs = F.softmax(steered_logits, dim=-1)
    
    # KL divergence: D_KL(baseline || steered)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl_div = F.kl_div(
        (steered_probs + eps).log(),
        baseline_probs,
        reduction='batchmean',
    )
    results["kl_divergence"] = kl_div.item()
    
    # Top-k overlap
    baseline_topk = torch.topk(baseline_logits, top_k, dim=-1).indices
    steered_topk = torch.topk(steered_logits, top_k, dim=-1).indices
    
    # Count overlapping predictions
    overlaps = 0
    total = baseline_topk.shape[0] * baseline_topk.shape[1] * top_k
    
    for b in range(baseline_topk.shape[0]):
        for s in range(baseline_topk.shape[1]):
            baseline_set = set(baseline_topk[b, s].tolist())
            steered_set = set(steered_topk[b, s].tolist())
            overlaps += len(baseline_set & steered_set)
    
    results["top_k_overlap"] = overlaps / total if total > 0 else 0.0
    
    # Prediction changes (argmax)
    baseline_preds = baseline_logits.argmax(dim=-1)
    steered_preds = steered_logits.argmax(dim=-1)
    
    n_changed = (baseline_preds != steered_preds).sum().item()
    total_positions = baseline_preds.numel()
    
    results["prediction_changes"] = n_changed
    results["prediction_change_rate"] = n_changed / total_positions if total_positions > 0 else 0.0
    results["total_positions"] = total_positions
    
    return results


def compute_concept_probability_shift(
    baseline_logits: torch.Tensor,
    steered_logits: torch.Tensor,
    tokenizer,
    concept_evaluator: callable,
    input_ids: Optional[torch.Tensor] = None,
    method: str = "masked_token",
) -> Dict[str, Any]:
    """Compute probability shift for target concept after steering.
    
    Implements: Success = P(target_concept | Steered) - P(target_concept | Baseline)
    
    Args:
        baseline_logits: Logits from baseline, shape [batch, seq_len, vocab_size]
        steered_logits: Logits from steered, shape [batch, seq_len, vocab_size]
        tokenizer: Tokenizer for decoding
        concept_evaluator: Function that takes (sequence, position, predicted_token) 
                          and returns probability/score of concept match
        input_ids: Original input token IDs, shape [batch, seq_len]
        method: Evaluation method - "masked_token", "sequence_level", or "token_probability"
        
    Returns:
        Dictionary with:
        - baseline_concept_prob: P(target_concept | Baseline)
        - steered_concept_prob: P(target_concept | Steered)
        - probability_shift: Difference (Success metric)
        - relative_improvement: Percentage improvement
    """
    results = {}
    
    # Convert to probabilities
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    steered_probs = F.softmax(steered_logits, dim=-1)
    
    batch_size, seq_len, vocab_size = baseline_logits.shape
    
    if method == "masked_token":
        # Method 1: Masked token probability
        # For each position, compute probability that predicted token creates concept match
        baseline_concept_scores = []
        steered_concept_scores = []
        
        if input_ids is None:
            raise ValueError("input_ids required for masked_token method")
        
        # Decode input sequence
        input_sequence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        for pos_idx in range(seq_len):
            # Skip special tokens
            if input_ids[0, pos_idx].item() in tokenizer.all_special_ids:
                continue
            
            # Get probability distribution over vocabulary for this position
            baseline_pos_probs = baseline_probs[0, pos_idx, :]  # [vocab_size]
            steered_pos_probs = steered_probs[0, pos_idx, :]
            
            # For each possible token, check if it creates concept match
            baseline_concept_prob = 0.0
            steered_concept_prob = 0.0
            
            for token_id in range(vocab_size):
                # Decode token
                try:
                    token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                    if not token_str:  # Skip empty tokens
                        continue
                    
                    # Create test sequence with this token at position
                    test_seq = input_sequence[:pos_idx] + token_str + input_sequence[pos_idx+1:]
                    
                    # Evaluate concept match
                    concept_score = concept_evaluator(test_seq, pos_idx, token_str)
                    
                    # Weight by probability of this token
                    baseline_concept_prob += baseline_pos_probs[token_id].item() * concept_score
                    steered_concept_prob += steered_pos_probs[token_id].item() * concept_score
                except:
                    continue
            
            baseline_concept_scores.append(baseline_concept_prob)
            steered_concept_scores.append(steered_concept_prob)
        
        baseline_concept_prob = np.mean(baseline_concept_scores) if baseline_concept_scores else 0.0
        steered_concept_prob = np.mean(steered_concept_scores) if steered_concept_scores else 0.0
        
    elif method == "token_probability":
        # Method 2: Probability of key concept tokens
        # Get top-k predictions and check if they match concept
        top_k = 10
        baseline_topk = torch.topk(baseline_logits, top_k, dim=-1).indices
        steered_topk = torch.topk(steered_logits, top_k, dim=-1).indices
        
        baseline_matches = 0
        steered_matches = 0
        total = 0
        
        if input_ids is None:
            raise ValueError("input_ids required for token_probability method")
        
        input_sequence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        for pos_idx in range(seq_len):
            if input_ids[0, pos_idx].item() in tokenizer.all_special_ids:
                continue
            
            # Check top-k predictions
            for k in range(top_k):
                baseline_token = baseline_topk[0, pos_idx, k].item()
                steered_token = steered_topk[0, pos_idx, k].item()
                
                baseline_token_str = tokenizer.decode([baseline_token], skip_special_tokens=True)
                steered_token_str = tokenizer.decode([steered_token], skip_special_tokens=True)
                
                test_seq_baseline = input_sequence[:pos_idx] + baseline_token_str + input_sequence[pos_idx+1:]
                test_seq_steered = input_sequence[:pos_idx] + steered_token_str + input_sequence[pos_idx+1:]
                
                if concept_evaluator(test_seq_baseline, pos_idx, baseline_token_str):
                    baseline_matches += 1
                if concept_evaluator(test_seq_steered, pos_idx, steered_token_str):
                    steered_matches += 1
                
                total += 1
        
        baseline_concept_prob = baseline_matches / total if total > 0 else 0.0
        steered_concept_prob = steered_matches / total if total > 0 else 0.0
        
    elif method == "sequence_level":
        # Method 3: Sequence-level probability using argmax predictions
        baseline_preds = baseline_logits.argmax(dim=-1)
        steered_preds = steered_logits.argmax(dim=-1)
        
        if input_ids is None:
            raise ValueError("input_ids required for sequence_level method")
        
        # Decode full sequences
        baseline_seq = tokenizer.decode(baseline_preds[0], skip_special_tokens=True)
        steered_seq = tokenizer.decode(steered_preds[0], skip_special_tokens=True)
        
        # Evaluate concept match (binary)
        baseline_concept_prob = 1.0 if concept_evaluator(baseline_seq, None, None) else 0.0
        steered_concept_prob = 1.0 if concept_evaluator(steered_seq, None, None) else 0.0
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute success metric
    probability_shift = steered_concept_prob - baseline_concept_prob
    
    # Relative improvement
    if baseline_concept_prob > 0:
        relative_improvement = (probability_shift / baseline_concept_prob) * 100
    else:
        relative_improvement = float('inf') if probability_shift > 0 else 0.0
    
    results = {
        "baseline_concept_prob": baseline_concept_prob,
        "steered_concept_prob": steered_concept_prob,
        "probability_shift": probability_shift,  # This is the Success metric
        "relative_improvement_percent": relative_improvement,
        "method": method,
    }
    
    return results


def create_motif_evaluator(motif_pattern: str) -> callable:
    """Create a concept evaluator function for motif matching.
    
    Args:
        motif_pattern: Regex pattern for the motif
        
    Returns:
        Function that takes (sequence, position, token) and returns concept match score
    """
    import re
    pattern = re.compile(motif_pattern)
    
    def evaluator(sequence: str, position: Optional[int], token: Optional[str]) -> float:
        """Evaluate if sequence matches motif pattern.
        
        Returns:
            1.0 if matches, 0.0 otherwise (binary)
            Or probability score if using fuzzy matching
        """
        if sequence is None:
            return 0.0
        return 1.0 if pattern.search(sequence) else 0.0
    
    return evaluator


def compute_steering_effect_by_layer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    steering_vector: torch.Tensor,
    target_layer: int,
    multipliers: List[float],
    use_logits: bool = False,
    tokenizer: Optional[Any] = None,
    concept_evaluator: Optional[Callable] = None,
    concept_eval_method: str = "masked_token",
) -> Dict[str, Any]:
    """Run steering experiment with multiple multipliers.
    
    Args:
        model: ESM-2 model
        input_ids: Input token IDs
        attention_mask: Attention mask
        steering_vector: The steering vector to inject
        target_layer: Which layer to steer at
        multipliers: List of multiplier values to test
        use_logits: Whether to use logits for comparison
        tokenizer: Tokenizer for concept probability evaluation
        concept_evaluator: Function to evaluate concept match (e.g., motif pattern)
        concept_eval_method: Method for concept evaluation ("masked_token", "token_probability", "sequence_level")
        
    Returns:
        Dictionary with results for each multiplier, including concept probability shift
    """
    from .hooks import SteeringHook
    
    results = {
        "target_layer": target_layer,
        "multipliers": multipliers,
        "experiments": {},
    }
    
    # Baseline run (no steering)
    model.eval()
    with torch.no_grad():
        baseline_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        baseline_hidden = baseline_outputs.hidden_states
        baseline_logits = baseline_outputs.logits if hasattr(baseline_outputs, 'logits') else None
    
    # Run with each multiplier
    for mult in multipliers:
        logger.info(f"Running with multiplier {mult}")
        
        hook = SteeringHook(model, target_layer, steering_vector, multiplier=mult)
        
        with hook:
            with torch.no_grad():
                steered_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                steered_hidden = steered_outputs.hidden_states
                steered_logits = steered_outputs.logits if hasattr(steered_outputs, 'logits') else None
        
        # Compare hidden states
        hidden_comparison = compare_hidden_states(
            list(baseline_hidden),
            list(steered_hidden),
        )
        
        # Compare logits if available
        logit_comparison = None
        if baseline_logits is not None and steered_logits is not None:
            logit_comparison = compare_logits(baseline_logits, steered_logits)
        
        # Compute concept probability shift if evaluator provided
        concept_shift = None
        if concept_evaluator is not None and tokenizer is not None:
            if baseline_logits is not None and steered_logits is not None:
                try:
                    concept_shift = compute_concept_probability_shift(
                        baseline_logits=baseline_logits,
                        steered_logits=steered_logits,
                        tokenizer=tokenizer,
                        concept_evaluator=concept_evaluator,
                        input_ids=input_ids,
                        method=concept_eval_method,
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute concept probability shift: {e}")
        
        results["experiments"][str(mult)] = {
            "hidden_state_comparison": hidden_comparison,
            "logit_comparison": logit_comparison,
            "concept_probability_shift": concept_shift,
        }
    
    return results


def generate_steering_report(
    results: Dict[str, Any],
    output_dir: str,
    experiment_name: str = "steering_experiment",
) -> Path:
    """Generate a report with visualizations of steering effects.
    
    Args:
        results: Results from compute_steering_effect_by_layer
        output_dir: Directory to save report
        experiment_name: Name for the experiment
        
    Returns:
        Path to the generated report
    """
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f"{experiment_name}_{timestamp}"
    
    # Save JSON results
    json_path = output_path / f"{report_name}.json"
    
    # Convert tensors to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    
    logger.info(f"Saved JSON results to {json_path}")
    
    # Create visualization
    if "experiments" in results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        multipliers = results["multipliers"]
        
        # Extract metrics for plotting
        cosine_sims_by_mult = {}
        euclidean_dists_by_mult = {}
        
        for mult in multipliers:
            exp = results["experiments"].get(mult, {})
            hidden_comp = exp.get("hidden_state_comparison", {})
            per_layer = hidden_comp.get("per_layer", {})
            
            cosine_sims = []
            euclidean_dists = []
            
            for layer_name in sorted(per_layer.keys(), key=lambda x: per_layer[x]["layer_idx"]):
                cosine_sims.append(per_layer[layer_name]["cosine_similarity"])
                euclidean_dists.append(per_layer[layer_name]["euclidean_distance"])
            
            cosine_sims_by_mult[mult] = cosine_sims
            euclidean_dists_by_mult[mult] = euclidean_dists
        
        # Plot 1: Cosine similarity by layer for each multiplier
        ax1 = axes[0, 0]
        for mult, sims in cosine_sims_by_mult.items():
            if sims:
                ax1.plot(range(len(sims)), sims, label=f"mult={mult}", marker='o', markersize=3)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Cosine Similarity")
        ax1.set_title("Cosine Similarity (Baseline vs Steered)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Euclidean distance by layer for each multiplier
        ax2 = axes[0, 1]
        for mult, dists in euclidean_dists_by_mult.items():
            if dists:
                ax2.plot(range(len(dists)), dists, label=f"mult={mult}", marker='o', markersize=3)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Euclidean Distance")
        ax2.set_title("Euclidean Distance (Baseline vs Steered)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Effect at target layer vs multiplier
        ax3 = axes[1, 0]
        target_layer = results.get("target_layer", 0)
        target_cosines = []
        target_dists = []
        
        for mult in multipliers:
            exp = results["experiments"].get(mult, {})
            summary = exp.get("hidden_state_comparison", {}).get("summary", {})
            # Use mean across all layers as proxy
            target_cosines.append(summary.get("mean_cosine_similarity", 1.0))
            target_dists.append(summary.get("mean_euclidean_distance", 0.0))
        
        ax3.plot(multipliers, target_cosines, 'b-o', label="Cosine Similarity")
        ax3.set_xlabel("Multiplier")
        ax3.set_ylabel("Mean Cosine Similarity")
        ax3.set_title("Effect Magnitude vs Multiplier")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Concept probability shift or logit changes
        ax4 = axes[1, 1]
        concept_shifts = []
        kl_divs = []
        
        # Check for concept probability shift first (more informative)
        for mult in multipliers:
            exp = results["experiments"].get(str(mult), {})
            concept_shift = exp.get("concept_probability_shift", {})
            if concept_shift:
                concept_shifts.append(concept_shift.get("probability_shift", 0.0))
            else:
                # Fall back to logit comparison
                logit_comp = exp.get("logit_comparison", {})
                if logit_comp:
                    kl_divs.append(logit_comp.get("kl_divergence", 0.0))
        
        if concept_shifts:
            ax4.plot(multipliers, concept_shifts, 'g-o', label="Concept Prob Shift", linewidth=2, markersize=6)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax4.set_xlabel("Multiplier")
            ax4.set_ylabel("Probability Shift (Success)")
            ax4.set_title("Concept Probability Shift\nP(concept|Steered) - P(concept|Baseline)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        elif kl_divs:
            ax4.plot(multipliers, kl_divs, 'r-o', label="KL Divergence")
            ax4.set_xlabel("Multiplier")
            ax4.set_ylabel("KL Divergence")
            ax4.set_title("Logit Distribution Change")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No evaluation data available", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Evaluation (N/A)")
        
        plt.tight_layout()
        
        fig_path = output_path / f"{report_name}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved visualization to {fig_path}")
    
    return json_path


def summarize_steering_results(results: Dict[str, Any]) -> str:
    """Generate a text summary of steering experiment results.
    
    Args:
        results: Results from compute_steering_effect_by_layer
        
    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ACTIVATION STEERING EXPERIMENT SUMMARY")
    lines.append("=" * 60)
    
    lines.append(f"\nTarget Layer: {results.get('target_layer', 'N/A')}")
    lines.append(f"Multipliers Tested: {results.get('multipliers', [])}")
    
    lines.append("\n" + "-" * 40)
    lines.append("RESULTS BY MULTIPLIER:")
    lines.append("-" * 40)
    
    for mult, exp in results.get("experiments", {}).items():
        lines.append(f"\nMultiplier = {mult}:")
        
        hidden_comp = exp.get("hidden_state_comparison", {})
        summary = hidden_comp.get("summary", {})
        
        lines.append(f"  Hidden States:")
        lines.append(f"    Mean Cosine Similarity: {summary.get('mean_cosine_similarity', 'N/A'):.4f}")
        lines.append(f"    Min Cosine Similarity:  {summary.get('min_cosine_similarity', 'N/A'):.4f}")
        lines.append(f"    Mean Euclidean Dist:    {summary.get('mean_euclidean_distance', 'N/A'):.4f}")
        lines.append(f"    Max Relative Change:    {summary.get('max_relative_change', 'N/A'):.4f}")
        
        # Concept probability shift (most important metric)
        concept_shift = exp.get("concept_probability_shift")
        if concept_shift:
            lines.append(f"  Concept Probability Shift (Success Metric):")
            lines.append(f"    Baseline P(concept):   {concept_shift.get('baseline_concept_prob', 'N/A'):.4f}")
            lines.append(f"    Steered P(concept):     {concept_shift.get('steered_concept_prob', 'N/A'):.4f}")
            lines.append(f"    Probability Shift:      {concept_shift.get('probability_shift', 'N/A'):.4f}")
            lines.append(f"    Relative Improvement:   {concept_shift.get('relative_improvement_percent', 'N/A'):.2f}%")
            lines.append(f"    Method:                 {concept_shift.get('method', 'N/A')}")
        
        logit_comp = exp.get("logit_comparison")
        if logit_comp:
            lines.append(f"  Logits/Probabilities:")
            lines.append(f"    KL Divergence:          {logit_comp.get('kl_divergence', 'N/A'):.4f}")
            lines.append(f"    Top-K Overlap:          {logit_comp.get('top_k_overlap', 'N/A'):.4f}")
            lines.append(f"    Prediction Changes:     {logit_comp.get('prediction_changes', 'N/A')}/{logit_comp.get('total_positions', 'N/A')}")
            lines.append(f"    Prediction Change Rate: {logit_comp.get('prediction_change_rate', 'N/A'):.4f}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)

