"""Steering Vector derivation for ESM-2 models.

This module provides utilities for computing steering vectors from
positive and negative protein sequence sets.
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.logging_utils import get_logger

logger = get_logger("steering.vector")


class SteeringVector:
    """Derives steering vectors from positive/negative protein sequence sets.
    
    A steering vector is the mean difference in activations between proteins
    with a desired property (positive set) and proteins without it (negative set).
    This vector can then be added to the residual stream to "steer" the model
    toward that property.
    
    Attributes:
        model: The ESM-2 model
        tokenizer: The tokenizer for the model
        layer_id: Which layer to extract activations from
        device: Device to run computations on
        aggregation: How to aggregate over sequence positions ("mean", "first", "max")
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        layer_id: int,
        device: torch.device,
        aggregation: str = "mean",
        max_length: int = 1024,
    ):
        """Initialize the SteeringVector extractor.
        
        Args:
            model: ESM-2 model (from HuggingFace transformers)
            tokenizer: Tokenizer for the model
            layer_id: Layer index to extract activations from
            device: Device for computations
            aggregation: How to aggregate activations over sequence ("mean", "first", "max")
            max_length: Maximum sequence length for tokenization
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_id = layer_id
        self.device = device
        self.aggregation = aggregation
        self.max_length = max_length
        self._vector = None
        
    @property
    def vector(self) -> Optional[torch.Tensor]:
        """The computed steering vector, or None if not yet computed."""
        return self._vector
    
    @property
    def hidden_dim(self) -> int:
        """Hidden dimension of the model."""
        if self._vector is not None:
            return self._vector.shape[0]
        # Try to get from model config
        if hasattr(self.model, 'config'):
            return self.model.config.hidden_size
        raise ValueError("Vector not computed and cannot determine hidden dim from model")
    
    def from_motif(
        self,
        sequences: List[Dict[str, Any]],
        motif_pattern: str,
        n_positive: int = 100,
        n_negative: int = 100,
        random_seed: int = 42,
    ) -> torch.Tensor:
        """Derive steering vector from sequences matching a motif pattern.
        
        Args:
            sequences: List of sequence dicts with 'id' and 'sequence' keys
            motif_pattern: Regex pattern for the motif (e.g., "C.{2,4}C" for zinc finger)
            n_positive: Number of positive examples to use
            n_negative: Number of negative examples to use
            random_seed: Random seed for sampling
            
        Returns:
            Steering vector tensor of shape [hidden_dim]
        """
        logger.info(f"Deriving steering vector from motif pattern: {motif_pattern}")
        
        pattern = re.compile(motif_pattern)
        
        positive_seqs = []
        negative_seqs = []
        
        for seq_dict in sequences:
            seq = seq_dict.get("sequence", "")
            if pattern.search(seq):
                positive_seqs.append(seq_dict)
            else:
                negative_seqs.append(seq_dict)
        
        logger.info(f"Found {len(positive_seqs)} sequences matching motif, {len(negative_seqs)} not matching")
        
        # Sample if we have more than needed
        np.random.seed(random_seed)
        
        if len(positive_seqs) > n_positive:
            indices = np.random.choice(len(positive_seqs), n_positive, replace=False)
            positive_seqs = [positive_seqs[i] for i in indices]
        
        if len(negative_seqs) > n_negative:
            indices = np.random.choice(len(negative_seqs), n_negative, replace=False)
            negative_seqs = [negative_seqs[i] for i in indices]
        
        if len(positive_seqs) < 5:
            raise ValueError(f"Not enough positive examples: {len(positive_seqs)} < 5")
        if len(negative_seqs) < 5:
            raise ValueError(f"Not enough negative examples: {len(negative_seqs)} < 5")
        
        logger.info(f"Using {len(positive_seqs)} positive and {len(negative_seqs)} negative examples")
        
        return self.compute(positive_seqs, negative_seqs)
    
    def from_go_clusters(
        self,
        prism_results_path: str,
        target_cluster_id: int,
        sequences: List[Dict[str, Any]],
        n_positive: int = 100,
        n_negative: int = 100,
        random_seed: int = 42,
    ) -> torch.Tensor:
        """Derive steering vector from PRISM cluster assignments.
        
        Uses the cluster assignments from a PRISM analysis run to define
        positive (in-cluster) and negative (out-of-cluster) sets.
        
        Args:
            prism_results_path: Path to PRISM results JSON file
            target_cluster_id: Cluster ID to use as positive set
            sequences: List of sequence dicts with 'id' and 'sequence' keys
            n_positive: Max number of positive examples
            n_negative: Max number of negative examples
            random_seed: Random seed for sampling
            
        Returns:
            Steering vector tensor of shape [hidden_dim]
        """
        logger.info(f"Deriving steering vector from PRISM cluster {target_cluster_id}")
        
        with open(prism_results_path, 'r') as f:
            prism_data = json.load(f)
        
        # Find the cluster in the results
        # PRISM results have structure: {"neurons": [{"clusters": {cluster_id: {"sequence_ids": [...]}}}]}
        cluster_seq_ids = set()
        all_seq_ids = set()
        
        for neuron in prism_data.get("neurons", []):
            clusters = neuron.get("clusters", {})
            for cid, cluster_data in clusters.items():
                seq_ids = cluster_data.get("sequence_ids", [])
                all_seq_ids.update(seq_ids)
                if int(cid) == target_cluster_id:
                    cluster_seq_ids.update(seq_ids)
        
        logger.info(f"Found {len(cluster_seq_ids)} sequences in cluster {target_cluster_id}")
        
        # Build positive and negative sets
        seq_lookup = {s.get("id"): s for s in sequences}
        
        positive_seqs = [seq_lookup[sid] for sid in cluster_seq_ids if sid in seq_lookup]
        negative_ids = all_seq_ids - cluster_seq_ids
        negative_seqs = [seq_lookup[sid] for sid in negative_ids if sid in seq_lookup]
        
        # Sample
        np.random.seed(random_seed)
        
        if len(positive_seqs) > n_positive:
            indices = np.random.choice(len(positive_seqs), n_positive, replace=False)
            positive_seqs = [positive_seqs[i] for i in indices]
        
        if len(negative_seqs) > n_negative:
            indices = np.random.choice(len(negative_seqs), n_negative, replace=False)
            negative_seqs = [negative_seqs[i] for i in indices]
        
        if len(positive_seqs) < 3:
            raise ValueError(f"Not enough positive examples in cluster: {len(positive_seqs)} < 3")
        
        logger.info(f"Using {len(positive_seqs)} positive and {len(negative_seqs)} negative examples")
        
        return self.compute(positive_seqs, negative_seqs)
    
    def from_id_lists(
        self,
        positive_ids_path: str,
        negative_ids_path: str,
        sequences: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Derive steering vector from explicit ID lists.
        
        Args:
            positive_ids_path: Path to text file with positive protein IDs (one per line)
            negative_ids_path: Path to text file with negative protein IDs (one per line)
            sequences: List of sequence dicts with 'id' and 'sequence' keys
            
        Returns:
            Steering vector tensor of shape [hidden_dim]
        """
        logger.info("Deriving steering vector from ID lists")
        
        with open(positive_ids_path, 'r') as f:
            positive_ids = set(line.strip() for line in f if line.strip())
        
        with open(negative_ids_path, 'r') as f:
            negative_ids = set(line.strip() for line in f if line.strip())
        
        seq_lookup = {s.get("id"): s for s in sequences}
        
        positive_seqs = [seq_lookup[pid] for pid in positive_ids if pid in seq_lookup]
        negative_seqs = [seq_lookup[nid] for nid in negative_ids if nid in seq_lookup]
        
        logger.info(f"Found {len(positive_seqs)}/{len(positive_ids)} positive and "
                   f"{len(negative_seqs)}/{len(negative_ids)} negative sequences")
        
        if len(positive_seqs) < 3:
            raise ValueError(f"Not enough positive sequences found: {len(positive_seqs)}")
        if len(negative_seqs) < 3:
            raise ValueError(f"Not enough negative sequences found: {len(negative_seqs)}")
        
        return self.compute(positive_seqs, negative_seqs)
    
    def compute(
        self,
        positive_seqs: List[Dict[str, Any]],
        negative_seqs: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Compute the steering vector from positive and negative sequence sets.
        
        The steering vector is: mean(positive_activations) - mean(negative_activations)
        
        Args:
            positive_seqs: List of positive sequence dicts
            negative_seqs: List of negative sequence dicts
            
        Returns:
            Steering vector tensor of shape [hidden_dim]
        """
        logger.info(f"Computing steering vector at layer {self.layer_id}")
        
        # Extract activations for both sets
        pos_activations = self._extract_activations(positive_seqs)
        neg_activations = self._extract_activations(negative_seqs)
        
        # Compute means
        pos_mean = pos_activations.mean(dim=0)
        neg_mean = neg_activations.mean(dim=0)
        
        # Steering vector is the difference
        self._vector = pos_mean - neg_mean
        
        # Log some stats
        vector_norm = torch.norm(self._vector).item()
        logger.info(f"Steering vector computed: dim={self._vector.shape[0]}, norm={vector_norm:.4f}")
        
        return self._vector
    
    def _extract_activations(
        self,
        sequences: List[Dict[str, Any]],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Extract mean-pooled activations for a list of sequences.
        
        Args:
            sequences: List of sequence dicts with 'sequence' key
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape [n_sequences, hidden_dim]
        """
        all_activations = []
        
        self.model.eval()
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_texts = [s.get("sequence", "") for s in batch_seqs]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                
                # Get hidden states at target layer
                hidden_states = outputs.hidden_states[self.layer_id]
                
                # Aggregate over sequence positions
                if self.aggregation == "mean":
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                    batch_activations = sum_embeddings / sum_mask
                elif self.aggregation == "first":
                    batch_activations = hidden_states[:, 0]
                elif self.aggregation == "max":
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    hidden_masked = hidden_states.clone()
                    hidden_masked[~mask_expanded.bool()] = float('-inf')
                    batch_activations = hidden_masked.max(dim=1)[0]
                else:
                    batch_activations = hidden_states.mean(dim=1)
                
                all_activations.append(batch_activations)
        
        return torch.cat(all_activations, dim=0)
    
    def save(self, path: str) -> None:
        """Save the steering vector to a file.
        
        Args:
            path: Path to save the vector (.pt file)
        """
        if self._vector is None:
            raise ValueError("No steering vector computed yet")
        
        torch.save({
            "vector": self._vector.cpu(),
            "layer_id": self.layer_id,
            "aggregation": self.aggregation,
            "hidden_dim": self._vector.shape[0],
        }, path)
        logger.info(f"Saved steering vector to {path}")
    
    @classmethod
    def load(cls, path: str, model: torch.nn.Module, tokenizer, device: torch.device) -> "SteeringVector":
        """Load a steering vector from a file.
        
        Args:
            path: Path to the saved vector
            model: ESM-2 model
            tokenizer: Tokenizer
            device: Device to load to
            
        Returns:
            SteeringVector instance with loaded vector
        """
        data = torch.load(path, map_location=device)
        
        instance = cls(
            model=model,
            tokenizer=tokenizer,
            layer_id=data["layer_id"],
            device=device,
            aggregation=data.get("aggregation", "mean"),
        )
        instance._vector = data["vector"].to(device)
        
        logger.info(f"Loaded steering vector from {path}: dim={instance._vector.shape[0]}")
        
        return instance

