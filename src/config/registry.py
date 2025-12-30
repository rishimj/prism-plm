"""Component registry for dynamic instantiation of pluggable components."""
from typing import Any, Callable, Dict, List, Type, TypeVar

from src.utils.logging_utils import get_logger

logger = get_logger("config.registry")

T = TypeVar("T")


class Registry:
    """Generic registry for pluggable components.

    This allows registering classes by name and instantiating them dynamically
    based on configuration. Supports datasets, models, clustering algorithms,
    dimensionality reducers, etc.

    Usage:
        # Create a registry
        MODEL_REGISTRY = Registry("model")

        # Register a class
        @MODEL_REGISTRY.register("esm2")
        class ESM2Wrapper:
            def __init__(self, model_name: str):
                ...

        # Get and instantiate
        model_cls = MODEL_REGISTRY.get("esm2")
        model = MODEL_REGISTRY.create("esm2", model_name="facebook/esm2_t36_3B_UR50D")
    """

    def __init__(self, name: str):
        """Initialize the registry.

        Args:
            name: Name of this registry (for logging)
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
        logger.debug(f"Created registry: {name}")

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a component class.

        Args:
            name: Name to register the class under

        Returns:
            Decorator function
        """

        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                logger.warning(
                    f"Overwriting existing registration for '{name}' in {self.name} registry"
                )
            self._registry[name] = cls
            logger.debug(f"Registered '{name}' -> {cls.__name__} in {self.name} registry")
            return cls

        return decorator

    def register_class(self, name: str, cls: Type) -> None:
        """Register a component class directly (non-decorator).

        Args:
            name: Name to register the class under
            cls: The class to register
        """
        if name in self._registry:
            logger.warning(
                f"Overwriting existing registration for '{name}' in {self.name} registry"
            )
        self._registry[name] = cls
        logger.debug(f"Registered '{name}' -> {cls.__name__} in {self.name} registry")

    def get(self, name: str) -> Type:
        """Get component class by name.

        Args:
            name: Registered name of the component

        Returns:
            The registered class

        Raises:
            KeyError: If name is not registered
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(
                f"Unknown {self.name}: '{name}'. Available: [{available}]"
            )
        logger.debug(f"Retrieved '{name}' from {self.name} registry")
        return self._registry[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a component by name.

        Args:
            name: Registered name of the component
            **kwargs: Arguments to pass to the constructor

        Returns:
            Instance of the registered class
        """
        cls = self.get(name)
        logger.debug(f"Creating {self.name} '{name}' with kwargs: {list(kwargs.keys())}")
        return cls(**kwargs)

    def list_available(self) -> List[str]:
        """List all registered component names.

        Returns:
            List of registered names
        """
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered.

        Args:
            name: Name to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._registry

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return self.is_registered(name)

    def __len__(self) -> int:
        """Return number of registered components."""
        return len(self._registry)


# Global registries for each component type
DATASET_REGISTRY = Registry("dataset")
MODEL_REGISTRY = Registry("model")
CLUSTERING_REGISTRY = Registry("clustering")
REDUCER_REGISTRY = Registry("reducer")
TOKENIZER_REGISTRY = Registry("tokenizer")


def list_all_registries() -> Dict[str, List[str]]:
    """List all available components in all registries.

    Returns:
        Dictionary mapping registry name to list of registered components
    """
    return {
        "dataset": DATASET_REGISTRY.list_available(),
        "model": MODEL_REGISTRY.list_available(),
        "clustering": CLUSTERING_REGISTRY.list_available(),
        "reducer": REDUCER_REGISTRY.list_available(),
        "tokenizer": TOKENIZER_REGISTRY.list_available(),
    }






