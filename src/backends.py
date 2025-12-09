"""Accelerated backend system for 4D Neural Cognition.

This module implements a hybrid architecture that automatically selects
the optimal backend based on network size and available hardware.

Backends:
- NumPyBackend: For small networks (<10K neurons)
- JAXBackend: For large networks (>10K neurons) with GPU/TPU support
- GraphBackend: For sparse connectivity patterns
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Try importing JAX (optional dependency)
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class BackendBase(ABC):
    """Abstract base class for computation backends."""
    
    @abstractmethod
    def compute_neuron_updates(
        self,
        v_membrane: np.ndarray,
        u_recovery: np.ndarray,
        external_input: np.ndarray,
        neuron_params: Dict[str, Any],
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute neuron state updates.
        
        Args:
            v_membrane: Current membrane potentials
            u_recovery: Current recovery variables (for Izhikevich)
            external_input: External input currents
            neuron_params: Neuron model parameters
            dt: Time step
            
        Returns:
            Tuple of (new_v_membrane, new_u_recovery, spikes)
        """
        pass
    
    @abstractmethod
    def compute_synaptic_input(
        self,
        spikes: np.ndarray,
        weights: np.ndarray,
        pre_ids: np.ndarray,
        post_ids: np.ndarray,
        num_neurons: int
    ) -> np.ndarray:
        """Compute synaptic input for all neurons.
        
        Args:
            spikes: Boolean array of which neurons spiked
            weights: Synaptic weights
            pre_ids: Presynaptic neuron IDs
            post_ids: Postsynaptic neuron IDs
            num_neurons: Total number of neurons
            
        Returns:
            Array of synaptic inputs for each neuron
        """
        pass
    
    @abstractmethod
    def supports_gpu(self) -> bool:
        """Check if backend supports GPU acceleration."""
        pass


class NumPyBackend(BackendBase):
    """NumPy-based backend for small to medium networks.
    
    This backend uses standard NumPy operations and is suitable for
    networks with fewer than 10,000 neurons.
    """
    
    def __init__(self):
        """Initialize NumPy backend."""
        self.name = "NumPy"
    
    def compute_neuron_updates(
        self,
        v_membrane: np.ndarray,
        u_recovery: np.ndarray,
        external_input: np.ndarray,
        neuron_params: Dict[str, Any],
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute neuron updates using NumPy."""
        model_type = neuron_params.get('model_type', 'lif')
        
        if model_type == 'lif':
            return self._compute_lif_updates(v_membrane, external_input, neuron_params, dt)
        elif model_type == 'izhikevich':
            return self._compute_izhikevich_updates(
                v_membrane, u_recovery, external_input, neuron_params, dt
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _compute_lif_updates(
        self,
        v_membrane: np.ndarray,
        external_input: np.ndarray,
        params: Dict[str, Any],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute LIF neuron updates."""
        tau_m = params.get('tau_m', 10.0)
        v_rest = params.get('v_rest', -65.0)
        v_threshold = params.get('v_threshold', -50.0)
        v_reset = params.get('v_reset', -65.0)
        
        # Leak current
        dv = -(v_membrane - v_rest) / tau_m + external_input
        new_v = v_membrane + dv * dt
        
        # Check for spikes
        spikes = new_v >= v_threshold
        
        # Reset spiking neurons
        new_v[spikes] = v_reset
        
        # u_recovery not used in LIF
        new_u = np.zeros_like(v_membrane)
        
        return new_v, new_u, spikes
    
    def _compute_izhikevich_updates(
        self,
        v_membrane: np.ndarray,
        u_recovery: np.ndarray,
        external_input: np.ndarray,
        params: Dict[str, Any],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Izhikevich neuron updates."""
        a = params.get('a', 0.02)
        b = params.get('b', 0.2)
        c = params.get('c', -65.0)
        d = params.get('d', 8.0)
        v_threshold = params.get('v_threshold', 30.0)
        
        # Izhikevich dynamics
        dv = (0.04 * v_membrane**2 + 5 * v_membrane + 140 - u_recovery + external_input) * dt
        du = (a * (b * v_membrane - u_recovery)) * dt
        
        new_v = v_membrane + dv
        new_u = u_recovery + du
        
        # Check for spikes
        spikes = new_v >= v_threshold
        
        # Reset spiking neurons
        new_v[spikes] = c
        new_u[spikes] += d
        
        return new_v, new_u, spikes
    
    def compute_synaptic_input(
        self,
        spikes: np.ndarray,
        weights: np.ndarray,
        pre_ids: np.ndarray,
        post_ids: np.ndarray,
        num_neurons: int
    ) -> np.ndarray:
        """Compute synaptic input using NumPy."""
        synaptic_input = np.zeros(num_neurons)
        
        # Find which presynaptic neurons spiked
        spiking_mask = spikes[pre_ids]
        
        # Accumulate weighted inputs
        if np.any(spiking_mask):
            active_weights = weights[spiking_mask]
            active_post = post_ids[spiking_mask]
            np.add.at(synaptic_input, active_post, active_weights)
        
        return synaptic_input
    
    def supports_gpu(self) -> bool:
        """NumPy backend does not support GPU."""
        return False


class JAXBackend(BackendBase):
    """JAX-based backend for large networks with GPU/TPU support.
    
    This backend uses JAX for automatic differentiation and JIT compilation,
    making it suitable for networks with more than 10,000 neurons and
    providing automatic GPU/TPU support when available.
    """
    
    def __init__(self):
        """Initialize JAX backend."""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not installed. Install with: pip install jax jaxlib")
        
        self.name = "JAX"
        self._device = jax.devices()[0]
    
    def compute_neuron_updates(
        self,
        v_membrane: np.ndarray,
        u_recovery: np.ndarray,
        external_input: np.ndarray,
        neuron_params: Dict[str, Any],
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute neuron updates using JAX with JIT compilation."""
        # Convert to JAX arrays
        v_jax = jnp.array(v_membrane)
        u_jax = jnp.array(u_recovery)
        input_jax = jnp.array(external_input)
        
        model_type = neuron_params.get('model_type', 'lif')
        
        if model_type == 'lif':
            new_v, new_u, spikes = self._compute_lif_jax(v_jax, input_jax, neuron_params, dt)
        elif model_type == 'izhikevich':
            new_v, new_u, spikes = self._compute_izhikevich_jax(
                v_jax, u_jax, input_jax, neuron_params, dt
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Convert back to numpy
        return np.array(new_v), np.array(new_u), np.array(spikes)
    
    @staticmethod
    @jax.jit
    def _compute_lif_jax(v, external_input, params, dt):
        """JIT-compiled LIF neuron updates."""
        tau_m = params.get('tau_m', 10.0)
        v_rest = params.get('v_rest', -65.0)
        v_threshold = params.get('v_threshold', -50.0)
        v_reset = params.get('v_reset', -65.0)
        
        dv = -(v - v_rest) / tau_m + external_input
        new_v = v + dv * dt
        
        spikes = new_v >= v_threshold
        new_v = jnp.where(spikes, v_reset, new_v)
        
        new_u = jnp.zeros_like(v)
        
        return new_v, new_u, spikes
    
    @staticmethod
    @jax.jit
    def _compute_izhikevich_jax(v, u, external_input, params, dt):
        """JIT-compiled Izhikevich neuron updates."""
        a = params.get('a', 0.02)
        b = params.get('b', 0.2)
        c = params.get('c', -65.0)
        d = params.get('d', 8.0)
        v_threshold = params.get('v_threshold', 30.0)
        
        dv = (0.04 * v**2 + 5 * v + 140 - u + external_input) * dt
        du = (a * (b * v - u)) * dt
        
        new_v = v + dv
        new_u = u + du
        
        spikes = new_v >= v_threshold
        new_v = jnp.where(spikes, c, new_v)
        new_u = jnp.where(spikes, new_u + d, new_u)
        
        return new_v, new_u, spikes
    
    def compute_synaptic_input(
        self,
        spikes: np.ndarray,
        weights: np.ndarray,
        pre_ids: np.ndarray,
        post_ids: np.ndarray,
        num_neurons: int
    ) -> np.ndarray:
        """Compute synaptic input using JAX."""
        # Convert to JAX arrays
        spikes_jax = jnp.array(spikes)
        weights_jax = jnp.array(weights)
        pre_ids_jax = jnp.array(pre_ids)
        post_ids_jax = jnp.array(post_ids)
        
        result = self._compute_synaptic_jax(
            spikes_jax, weights_jax, pre_ids_jax, post_ids_jax, num_neurons
        )
        
        return np.array(result)
    
    @staticmethod
    @jax.jit
    def _compute_synaptic_jax(spikes, weights, pre_ids, post_ids, num_neurons):
        """JIT-compiled synaptic input computation."""
        spiking_mask = spikes[pre_ids]
        active_weights = weights * spiking_mask
        
        # Use segment_sum for efficient accumulation
        synaptic_input = jax.ops.segment_sum(
            active_weights,
            post_ids,
            num_segments=num_neurons
        )
        
        return synaptic_input
    
    def supports_gpu(self) -> bool:
        """Check if GPU is available."""
        return self._device.platform == 'gpu'


class GraphBackend(BackendBase):
    """Graph-based backend for sparse connectivity patterns.
    
    This backend is optimized for networks with sparse connectivity
    using graph-based representations and operations.
    """
    
    def __init__(self):
        """Initialize graph backend."""
        self.name = "Graph"
    
    def compute_neuron_updates(
        self,
        v_membrane: np.ndarray,
        u_recovery: np.ndarray,
        external_input: np.ndarray,
        neuron_params: Dict[str, Any],
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute neuron updates (uses NumPy implementation)."""
        # Delegate to NumPy backend for neuron updates
        numpy_backend = NumPyBackend()
        return numpy_backend.compute_neuron_updates(
            v_membrane, u_recovery, external_input, neuron_params, dt
        )
    
    def compute_synaptic_input(
        self,
        spikes: np.ndarray,
        weights: np.ndarray,
        pre_ids: np.ndarray,
        post_ids: np.ndarray,
        num_neurons: int
    ) -> np.ndarray:
        """Compute synaptic input using graph operations."""
        # Use sparse operations for efficiency
        synaptic_input = np.zeros(num_neurons)
        
        # Only process spiking neurons
        spiking_indices = np.where(spikes)[0]
        
        for spike_id in spiking_indices:
            # Find all connections from this spiking neuron
            connection_mask = pre_ids == spike_id
            if np.any(connection_mask):
                target_neurons = post_ids[connection_mask]
                connection_weights = weights[connection_mask]
                np.add.at(synaptic_input, target_neurons, connection_weights)
        
        return synaptic_input
    
    def supports_gpu(self) -> bool:
        """Graph backend does not support GPU directly."""
        return False


class AcceleratedEngine:
    """Hybrid engine that automatically selects the optimal backend.
    
    This engine manages multiple backends and automatically selects
    the most appropriate one based on network size and hardware availability.
    """
    
    def __init__(self, prefer_backend: Optional[str] = None):
        """Initialize the accelerated engine.
        
        Args:
            prefer_backend: Preferred backend ('numpy', 'jax', 'graph', or None for auto)
        """
        self.cpu_backend = NumPyBackend()
        self.sparse_backend = GraphBackend()
        
        # Try to initialize JAX backend
        self.gpu_backend: Optional[JAXBackend] = None
        if JAX_AVAILABLE:
            try:
                self.gpu_backend = JAXBackend()
            except Exception as e:
                print(f"Warning: JAX backend initialization failed: {e}")
        
        self.prefer_backend = prefer_backend
        self.current_backend: Optional[BackendBase] = None
    
    def select_backend(
        self,
        num_neurons: int,
        num_synapses: int,
        force_backend: Optional[str] = None
    ) -> BackendBase:
        """Automatically select the optimal backend.
        
        Args:
            num_neurons: Number of neurons in the network
            num_synapses: Number of synapses in the network
            force_backend: Force a specific backend (overrides auto-selection)
            
        Returns:
            Selected backend instance
        """
        # Use forced backend if specified
        if force_backend:
            return self._get_backend_by_name(force_backend)
        
        # Use preferred backend if specified
        if self.prefer_backend:
            return self._get_backend_by_name(self.prefer_backend)
        
        # Auto-select based on network characteristics
        connectivity_ratio = num_synapses / (num_neurons ** 2) if num_neurons > 0 else 0
        
        # For sparse networks, use graph backend
        if connectivity_ratio < 0.1 and num_neurons > 1000:
            self.current_backend = self.sparse_backend
            return self.sparse_backend
        
        # For large networks, use GPU backend if available
        if num_neurons > 10000 and self.gpu_backend and self.gpu_backend.supports_gpu():
            self.current_backend = self.gpu_backend
            return self.gpu_backend
        
        # Default to CPU backend for small to medium networks
        self.current_backend = self.cpu_backend
        return self.cpu_backend
    
    def _get_backend_by_name(self, name: str) -> BackendBase:
        """Get backend by name."""
        name_lower = name.lower()
        
        if name_lower == 'numpy' or name_lower == 'cpu':
            return self.cpu_backend
        elif name_lower == 'jax' or name_lower == 'gpu':
            if self.gpu_backend is None:
                raise ValueError("JAX backend not available. Install JAX to use GPU acceleration.")
            return self.gpu_backend
        elif name_lower == 'graph' or name_lower == 'sparse':
            return self.sparse_backend
        else:
            raise ValueError(f"Unknown backend: {name}")
    
    def get_available_backends(self) -> Dict[str, bool]:
        """Get list of available backends.
        
        Returns:
            Dictionary mapping backend names to availability
        """
        return {
            'numpy': True,
            'jax': self.gpu_backend is not None,
            'graph': True,
            'gpu_acceleration': self.gpu_backend is not None and self.gpu_backend.supports_gpu()
        }
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend and available options.
        
        Returns:
            Dictionary with backend information
        """
        info = {
            'available_backends': self.get_available_backends(),
            'current_backend': self.current_backend.name if self.current_backend else None,
            'jax_available': JAX_AVAILABLE,
        }
        
        if self.gpu_backend and self.gpu_backend.supports_gpu():
            info['gpu_device'] = str(self.gpu_backend._device)
        
        return info
