"""Plugin system for modular extensibility of 4D Neural Cognition.

This module provides a flexible plugin architecture that allows users to
extend the framework with custom neuron models, plasticity rules, sensory
modules, tasks, and visualizations without modifying core code.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable
import importlib
import inspect
import os
from pathlib import Path


class PluginBase(ABC):
    """Base class for all plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name (must be unique within plugin type)."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version string."""
        pass
    
    @property
    def description(self) -> str:
        """Plugin description (optional)."""
        return ""
    
    @property
    def author(self) -> str:
        """Plugin author (optional)."""
        return ""
    
    def validate(self) -> bool:
        """Validate plugin compatibility and requirements.
        
        Returns:
            True if plugin is valid and can be loaded
        """
        return True


class NeuronModelBase(PluginBase):
    """Base class for custom neuron models."""
    
    @abstractmethod
    def update(
        self,
        v_membrane: float,
        u_recovery: float,
        external_input: float,
        params: Dict[str, Any],
        dt: float = 1.0
    ) -> tuple[float, float, bool]:
        """Update neuron state for one time step.
        
        Args:
            v_membrane: Current membrane potential
            u_recovery: Current recovery variable
            external_input: External input current
            params: Model parameters
            dt: Time step
            
        Returns:
            Tuple of (new_v_membrane, new_u_recovery, did_spike)
        """
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this neuron model."""
        pass


class PlasticityRuleBase(PluginBase):
    """Base class for custom plasticity rules."""
    
    @abstractmethod
    def update_weights(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
        current_weight: float,
        params: Dict[str, Any],
        current_time: int
    ) -> float:
        """Update synaptic weight based on spike timing.
        
        Args:
            pre_spike_times: List of presynaptic spike times
            post_spike_times: List of postsynaptic spike times
            current_weight: Current synaptic weight
            params: Learning rule parameters
            current_time: Current simulation time
            
        Returns:
            Updated synaptic weight
        """
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this plasticity rule."""
        pass


class SenseModuleBase(PluginBase):
    """Base class for custom sensory processing modules."""
    
    @abstractmethod
    def process_input(
        self,
        raw_input: Any,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process raw sensory input.
        
        Args:
            raw_input: Raw sensory data (format depends on sense type)
            params: Processing parameters
            
        Returns:
            Dictionary with processed data and metadata
        """
        pass
    
    @abstractmethod
    def get_input_shape(self) -> tuple:
        """Get expected input shape."""
        pass
    
    @abstractmethod
    def get_output_shape(self) -> tuple:
        """Get output shape after processing."""
        pass


class TaskBase(PluginBase):
    """Base class for custom tasks/benchmarks."""
    
    @abstractmethod
    def generate_trial(self, trial_num: int) -> Dict[str, Any]:
        """Generate a single trial.
        
        Args:
            trial_num: Trial number
            
        Returns:
            Dictionary with trial data (inputs, targets, metadata)
        """
        pass
    
    @abstractmethod
    def evaluate_response(
        self,
        response: Any,
        trial_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate network response for a trial.
        
        Args:
            response: Network output/response
            trial_data: Original trial data
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    @abstractmethod
    def get_num_trials(self) -> int:
        """Get total number of trials."""
        pass


class VisualizationBase(PluginBase):
    """Base class for custom visualizations."""
    
    @abstractmethod
    def plot(
        self,
        data: Any,
        params: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Any:
        """Create visualization.
        
        Args:
            data: Data to visualize
            params: Visualization parameters
            save_path: Optional path to save figure
            
        Returns:
            Figure or plot object (format depends on backend)
        """
        pass
    
    @abstractmethod
    def get_required_data_format(self) -> str:
        """Get description of required data format."""
        pass


class PluginRegistry:
    """Registry for managing plugins."""
    
    PLUGIN_TYPES = {
        'neuron_model': NeuronModelBase,
        'plasticity_rule': PlasticityRuleBase,
        'sense_module': SenseModuleBase,
        'task': TaskBase,
        'visualization': VisualizationBase
    }
    
    def __init__(self):
        """Initialize plugin registry."""
        self._plugins: Dict[str, Dict[str, PluginBase]] = {
            plugin_type: {} for plugin_type in self.PLUGIN_TYPES
        }
        self._load_callbacks: Dict[str, List[Callable]] = {
            plugin_type: [] for plugin_type in self.PLUGIN_TYPES
        }
    
    def register_plugin(
        self,
        plugin_type: str,
        plugin_instance: PluginBase
    ) -> None:
        """Register a plugin instance.
        
        Args:
            plugin_type: Type of plugin (e.g., 'neuron_model', 'plasticity_rule')
            plugin_instance: Plugin instance to register
            
        Raises:
            ValueError: If plugin type is invalid or plugin name already exists
            TypeError: If plugin instance doesn't inherit from correct base class
        """
        # Validate plugin type
        if plugin_type not in self.PLUGIN_TYPES:
            raise ValueError(
                f"Invalid plugin type: {plugin_type}. "
                f"Must be one of: {list(self.PLUGIN_TYPES.keys())}"
            )
        
        # Validate plugin instance type
        expected_base = self.PLUGIN_TYPES[plugin_type]
        if not isinstance(plugin_instance, expected_base):
            raise TypeError(
                f"Plugin must inherit from {expected_base.__name__}, "
                f"got {type(plugin_instance).__name__}"
            )
        
        # Validate plugin
        if not plugin_instance.validate():
            raise ValueError(f"Plugin validation failed: {plugin_instance.name}")
        
        # Check for name conflicts
        plugin_name = plugin_instance.name
        if plugin_name in self._plugins[plugin_type]:
            raise ValueError(
                f"Plugin '{plugin_name}' already registered for type '{plugin_type}'"
            )
        
        # Register plugin
        self._plugins[plugin_type][plugin_name] = plugin_instance
        
        # Call load callbacks
        for callback in self._load_callbacks[plugin_type]:
            callback(plugin_instance)
    
    def get_plugin(self, plugin_type: str, plugin_name: str) -> Optional[PluginBase]:
        """Get a registered plugin by type and name.
        
        Args:
            plugin_type: Type of plugin
            plugin_name: Name of plugin
            
        Returns:
            Plugin instance or None if not found
        """
        if plugin_type not in self._plugins:
            return None
        return self._plugins[plugin_type].get(plugin_name)
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all registered plugins.
        
        Args:
            plugin_type: If specified, only list plugins of this type
            
        Returns:
            Dictionary mapping plugin types to lists of plugin names
        """
        if plugin_type:
            if plugin_type not in self._plugins:
                return {}
            return {plugin_type: list(self._plugins[plugin_type].keys())}
        
        return {
            ptype: list(plugins.keys())
            for ptype, plugins in self._plugins.items()
        }
    
    def get_plugin_info(
        self,
        plugin_type: str,
        plugin_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get information about a plugin.
        
        Args:
            plugin_type: Type of plugin
            plugin_name: Name of plugin
            
        Returns:
            Dictionary with plugin information or None if not found
        """
        plugin = self.get_plugin(plugin_type, plugin_name)
        if plugin is None:
            return None
        
        return {
            'name': plugin.name,
            'version': plugin.version,
            'description': plugin.description,
            'author': plugin.author,
            'type': plugin_type
        }
    
    def unregister_plugin(self, plugin_type: str, plugin_name: str) -> bool:
        """Unregister a plugin.
        
        Args:
            plugin_type: Type of plugin
            plugin_name: Name of plugin
            
        Returns:
            True if plugin was unregistered, False if not found
        """
        if plugin_type not in self._plugins:
            return False
        
        if plugin_name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][plugin_name]
            return True
        
        return False
    
    def add_load_callback(
        self,
        plugin_type: str,
        callback: Callable[[PluginBase], None]
    ) -> None:
        """Add a callback to be called when plugins are loaded.
        
        Args:
            plugin_type: Type of plugin to watch
            callback: Function to call with plugin instance when loaded
        """
        if plugin_type in self._load_callbacks:
            self._load_callbacks[plugin_type].append(callback)
    
    def discover_plugins(self, plugin_dir: str) -> Dict[str, int]:
        """Discover and load plugins from a directory.
        
        Args:
            plugin_dir: Directory containing plugin modules
            
        Returns:
            Dictionary mapping plugin types to number of plugins loaded
        """
        loaded_counts = {ptype: 0 for ptype in self.PLUGIN_TYPES}
        
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return loaded_counts
        
        # Find all Python files in plugin directory
        for py_file in plugin_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            try:
                # Import module
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find plugin classes in module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it's a plugin class (not base class)
                        for plugin_type, base_class in self.PLUGIN_TYPES.items():
                            if (issubclass(obj, base_class) and 
                                obj is not base_class and
                                not inspect.isabstract(obj)):
                                
                                # Try to instantiate and register
                                try:
                                    plugin_instance = obj()
                                    self.register_plugin(plugin_type, plugin_instance)
                                    loaded_counts[plugin_type] += 1
                                except Exception as e:
                                    print(f"Warning: Failed to load plugin {name}: {e}")
            
            except Exception as e:
                print(f"Warning: Failed to import plugin module {py_file.name}: {e}")
        
        return loaded_counts


# Global plugin registry instance
_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance.
    
    Returns:
        Global PluginRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def register_plugin(plugin_type: str, plugin_instance: PluginBase) -> None:
    """Register a plugin with the global registry.
    
    Args:
        plugin_type: Type of plugin
        plugin_instance: Plugin instance to register
    """
    registry = get_plugin_registry()
    registry.register_plugin(plugin_type, plugin_instance)


def get_plugin(plugin_type: str, plugin_name: str) -> Optional[PluginBase]:
    """Get a plugin from the global registry.
    
    Args:
        plugin_type: Type of plugin
        plugin_name: Name of plugin
        
    Returns:
        Plugin instance or None if not found
    """
    registry = get_plugin_registry()
    return registry.get_plugin(plugin_type, plugin_name)


def list_plugins(plugin_type: Optional[str] = None) -> Dict[str, List[str]]:
    """List all registered plugins.
    
    Args:
        plugin_type: If specified, only list plugins of this type
        
    Returns:
        Dictionary mapping plugin types to lists of plugin names
    """
    registry = get_plugin_registry()
    return registry.list_plugins(plugin_type)
