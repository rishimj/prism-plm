"""Forward hook management for activation steering.

This module provides utilities for attaching and managing forward hooks
that inject steering vectors into the ESM-2 residual stream.
"""

from typing import Optional, Callable, Tuple, Any
from contextlib import contextmanager

import torch
import torch.nn as nn

from src.utils.logging_utils import get_logger

logger = get_logger("steering.hooks")


class SteeringHook:
    """Manages forward hooks for injecting steering vectors into ESM-2.
    
    The hook intercepts the output of a transformer layer and adds
    the steering vector scaled by a multiplier:
    
        output_steered = output + multiplier * steering_vector
    
    The steering vector is broadcast across batch and sequence dimensions.
    
    Attributes:
        model: The ESM-2 model
        layer_id: Which layer to hook into
        steering_vector: The vector to inject
        multiplier: Scaling factor for the steering vector
        handle: The hook handle (for removal)
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_id: int,
        steering_vector: torch.Tensor,
        multiplier: float = 1.0,
    ):
        """Initialize the steering hook.
        
        Args:
            model: ESM-2 model (from HuggingFace transformers)
            layer_id: Layer index to hook into
            steering_vector: Vector to add, shape [hidden_dim]
            multiplier: Scaling factor for the vector
        """
        self.model = model
        self.layer_id = layer_id
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._is_attached = False
        
    def _get_target_layer(self) -> nn.Module:
        """Get the target layer module from the model.
        
        ESM-2 models from HuggingFace have structure:
            model.esm.encoder.layer[i]  or
            model.encoder.layer[i]
        
        Returns:
            The target transformer layer module
        """
        # Try different possible structures
        if hasattr(self.model, 'esm'):
            # HuggingFace ESM2 structure: model.esm.encoder.layer[i]
            if hasattr(self.model.esm, 'encoder'):
                return self.model.esm.encoder.layer[self.layer_id]
        
        if hasattr(self.model, 'encoder'):
            # Alternative structure: model.encoder.layer[i]
            if hasattr(self.model.encoder, 'layer'):
                return self.model.encoder.layer[self.layer_id]
        
        if hasattr(self.model, 'layers'):
            # Some models use model.layers[i]
            return self.model.layers[self.layer_id]
        
        raise ValueError(
            f"Cannot find layer {self.layer_id} in model. "
            "Supported structures: model.esm.encoder.layer[i], "
            "model.encoder.layer[i], model.layers[i]"
        )
    
    def _hook_fn(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Any,
    ) -> Any:
        """The forward hook function that performs steering.
        
        Args:
            module: The hooked module
            input: Input to the module
            output: Output from the module (to be modified)
            
        Returns:
            Modified output with steering vector added
        """
        # Handle different output types
        # ESM layers can return (hidden_states,) or (hidden_states, attn_weights, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # hidden_states shape: [batch, seq_len, hidden_dim]
        # steering_vector shape: [hidden_dim]
        
        # Broadcast steering vector: [1, 1, hidden_dim]
        steering = self.steering_vector.unsqueeze(0).unsqueeze(0)
        
        # Ensure same device and dtype
        steering = steering.to(hidden_states.device, dtype=hidden_states.dtype)
        
        # Add the scaled steering vector
        steered_hidden = hidden_states + self.multiplier * steering
        
        if rest is not None:
            return (steered_hidden,) + rest
        return steered_hidden
    
    def attach(self) -> "SteeringHook":
        """Attach the hook to the model.
        
        Returns:
            self for chaining
        """
        if self._is_attached:
            logger.warning("Hook already attached, removing old hook first")
            self.remove()
        
        target_layer = self._get_target_layer()
        self.handle = target_layer.register_forward_hook(self._hook_fn)
        self._is_attached = True
        
        logger.info(f"Attached steering hook to layer {self.layer_id} with multiplier {self.multiplier}")
        
        return self
    
    def remove(self) -> None:
        """Remove the hook from the model."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self._is_attached = False
            logger.info(f"Removed steering hook from layer {self.layer_id}")
    
    def set_multiplier(self, multiplier: float) -> None:
        """Update the steering multiplier.
        
        Args:
            multiplier: New multiplier value
        """
        self.multiplier = multiplier
        logger.debug(f"Updated steering multiplier to {multiplier}")
    
    def __enter__(self) -> "SteeringHook":
        """Context manager entry - attach the hook."""
        return self.attach()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - remove the hook."""
        self.remove()


class MultiLayerSteeringHook:
    """Manages steering hooks across multiple layers.
    
    This allows steering at multiple layers simultaneously with
    potentially different vectors and multipliers per layer.
    """
    
    def __init__(self, model: nn.Module):
        """Initialize the multi-layer hook manager.
        
        Args:
            model: ESM-2 model
        """
        self.model = model
        self.hooks: dict[int, SteeringHook] = {}
    
    def add_hook(
        self,
        layer_id: int,
        steering_vector: torch.Tensor,
        multiplier: float = 1.0,
    ) -> None:
        """Add a steering hook at a specific layer.
        
        Args:
            layer_id: Layer index
            steering_vector: Vector to inject
            multiplier: Scaling factor
        """
        if layer_id in self.hooks:
            self.hooks[layer_id].remove()
        
        hook = SteeringHook(
            model=self.model,
            layer_id=layer_id,
            steering_vector=steering_vector,
            multiplier=multiplier,
        )
        self.hooks[layer_id] = hook
    
    def attach_all(self) -> "MultiLayerSteeringHook":
        """Attach all hooks.
        
        Returns:
            self for chaining
        """
        for hook in self.hooks.values():
            hook.attach()
        return self
    
    def remove_all(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks.values():
            hook.remove()
    
    def set_all_multipliers(self, multiplier: float) -> None:
        """Set the same multiplier for all hooks.
        
        Args:
            multiplier: New multiplier value
        """
        for hook in self.hooks.values():
            hook.set_multiplier(multiplier)
    
    def __enter__(self) -> "MultiLayerSteeringHook":
        """Context manager entry."""
        return self.attach_all()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.remove_all()


@contextmanager
def steering_context(
    model: nn.Module,
    layer_id: int,
    steering_vector: torch.Tensor,
    multiplier: float = 1.0,
):
    """Context manager for temporary steering.
    
    Usage:
        with steering_context(model, layer_id, vector, multiplier=2.0):
            output = model(input_ids)  # Steered output
        # Hook automatically removed
    
    Args:
        model: ESM-2 model
        layer_id: Layer to hook
        steering_vector: Vector to inject
        multiplier: Scaling factor
        
    Yields:
        The SteeringHook instance
    """
    hook = SteeringHook(model, layer_id, steering_vector, multiplier)
    try:
        hook.attach()
        yield hook
    finally:
        hook.remove()

