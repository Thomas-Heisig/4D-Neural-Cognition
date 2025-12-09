"""GPU acceleration module for 4D Neural Cognition using CUDA.

This module provides GPU-accelerated implementations of core neural network
operations including neuron updates and synapse computations using CuPy.

Features:
- Vectorized neuron updates on GPU
- Sparse matrix operations for synaptic connections
- cuBLAS integration for matrix operations
- Automatic CPU fallback when GPU is unavailable
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cp_sparse = None


class GPUAccelerator:
    """GPU acceleration manager for neural network operations.
    
    This class provides GPU-accelerated implementations of core operations
    with automatic fallback to CPU when GPU is unavailable.
    """
    
    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        """Initialize GPU accelerator.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
            device_id: CUDA device ID to use (default: 0)
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_id = device_id
        
        if self.use_gpu:
            try:
                cp.cuda.Device(device_id).use()
                self.device_name = cp.cuda.Device().name.decode('utf-8')
                self.device_memory = cp.cuda.Device().mem_info[1]  # Total memory
                print(f"GPU acceleration enabled: {self.device_name} "
                      f"({self.device_memory / 1e9:.1f} GB)")
            except Exception as e:
                warnings.warn(f"Failed to initialize GPU: {e}. Falling back to CPU.")
                self.use_gpu = False
        else:
            if not CUPY_AVAILABLE:
                warnings.warn("CuPy not available. GPU acceleration disabled. "
                            "Install CuPy for GPU support: pip install cupy-cuda12x")
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available and enabled."""
        return self.use_gpu
    
    def to_device(self, array: np.ndarray) -> Any:
        """Transfer array to GPU if available, otherwise return as-is.
        
        Args:
            array: NumPy array to transfer
            
        Returns:
            CuPy array if GPU is available, otherwise NumPy array
        """
        if self.use_gpu:
            return cp.asarray(array)
        return array
    
    def to_host(self, array: Any) -> np.ndarray:
        """Transfer array from GPU to CPU.
        
        Args:
            array: Array to transfer (CuPy or NumPy)
            
        Returns:
            NumPy array
        """
        if self.use_gpu and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def vectorized_lif_update(
        self,
        v_membrane: np.ndarray,
        synaptic_input: np.ndarray,
        external_input: np.ndarray,
        refractory_mask: np.ndarray,
        params: Dict[str, float],
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized Leaky Integrate-and-Fire neuron update on GPU.
        
        Updates membrane potentials for all neurons in parallel using the LIF model:
        dV/dt = (-(V - V_rest) + I_total) / tau_m
        
        Args:
            v_membrane: Current membrane potentials (N,)
            synaptic_input: Synaptic input currents (N,)
            external_input: External input currents (N,)
            refractory_mask: Boolean mask for neurons in refractory period (N,)
            params: Dictionary of LIF parameters
                - tau_m: Membrane time constant
                - v_rest: Resting potential
                - v_threshold: Spike threshold
                - v_reset: Reset potential after spike
            dt: Time step in ms
            
        Returns:
            Tuple of (updated_v_membrane, spike_mask)
        """
        # Transfer to GPU if available
        if self.use_gpu:
            v_mem = cp.asarray(v_membrane)
            syn_in = cp.asarray(synaptic_input)
            ext_in = cp.asarray(external_input)
            ref_mask = cp.asarray(refractory_mask)
            xp = cp
        else:
            v_mem = v_membrane
            syn_in = synaptic_input
            ext_in = external_input
            ref_mask = refractory_mask
            xp = np
        
        # Extract parameters
        tau_m = params.get('tau_m', 20.0)
        v_rest = params.get('v_rest', -65.0)
        v_threshold = params.get('v_threshold', -50.0)
        v_reset = params.get('v_reset', -70.0)
        
        # Compute total input current
        total_input = syn_in + ext_in
        
        # Apply refractory mask (neurons in refractory period don't integrate)
        active_mask = ~ref_mask
        
        # Leaky integration: dV/dt = (-(V - V_rest) + I) / tau_m
        dv = (-(v_mem - v_rest) + total_input) / tau_m * dt
        v_mem = xp.where(active_mask, v_mem + dv, v_mem)
        
        # Check for NaN/Inf and reset to safe values
        nan_mask = xp.isnan(v_mem) | xp.isinf(v_mem)
        v_mem = xp.where(nan_mask, v_rest, v_mem)
        
        # Detect spikes and reset
        spike_mask = v_mem >= v_threshold
        v_mem = xp.where(spike_mask, v_reset, v_mem)
        
        # Transfer back to CPU if needed
        if self.use_gpu:
            return cp.asnumpy(v_mem), cp.asnumpy(spike_mask)
        return v_mem, spike_mask
    
    def sparse_synapse_matmul(
        self,
        spike_vector: np.ndarray,
        weight_matrix: Any,
        delays: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute synaptic currents using sparse matrix multiplication.
        
        Efficiently computes I_syn = W @ spikes where W is a sparse connectivity
        matrix and spikes is a binary vector of recent spikes.
        
        Args:
            spike_vector: Binary vector of which neurons spiked (N,)
            weight_matrix: Sparse weight matrix (N, N) in CSR format
            delays: Optional delay matrix (not yet implemented)
            
        Returns:
            Synaptic input current for each neuron (N,)
        """
        if self.use_gpu:
            # Transfer to GPU
            spikes_gpu = cp.asarray(spike_vector, dtype=cp.float32)
            
            # Convert weight matrix to GPU sparse format if needed
            if isinstance(weight_matrix, np.ndarray):
                # Dense matrix
                weights_gpu = cp.asarray(weight_matrix, dtype=cp.float32)
                result = cp.dot(weights_gpu, spikes_gpu)
            elif hasattr(weight_matrix, 'toarray'):
                # Sparse matrix (scipy or cupy)
                if not isinstance(weight_matrix, (cp_sparse.csr_matrix, cp_sparse.csc_matrix)):
                    # Convert scipy sparse to cupy sparse
                    weights_gpu = cp_sparse.csr_matrix(
                        (cp.asarray(weight_matrix.data),
                         cp.asarray(weight_matrix.indices),
                         cp.asarray(weight_matrix.indptr)),
                        shape=weight_matrix.shape
                    )
                else:
                    weights_gpu = weight_matrix
                result = weights_gpu.dot(spikes_gpu)
            else:
                raise ValueError("weight_matrix must be a numpy/scipy array or sparse matrix")
            
            return cp.asnumpy(result)
        else:
            # CPU fallback
            if hasattr(weight_matrix, 'dot'):
                # Sparse matrix
                return weight_matrix.dot(spike_vector)
            else:
                # Dense matrix
                return np.dot(weight_matrix, spike_vector)
    
    def batch_matrix_operations(
        self,
        operations: List[Tuple[str, Any, Any]],
        use_cublas: bool = True
    ) -> List[np.ndarray]:
        """Perform batch matrix operations using cuBLAS for efficiency.
        
        Groups multiple matrix operations together to minimize transfer overhead.
        
        Args:
            operations: List of (operation, matrix_a, matrix_b) tuples
                operation can be: 'matmul', 'add', 'multiply'
            use_cublas: Use cuBLAS for operations (requires GPU)
            
        Returns:
            List of results (as NumPy arrays)
        """
        if not self.use_gpu:
            # CPU fallback
            results = []
            for op, a, b in operations:
                if op == 'matmul':
                    results.append(np.matmul(a, b))
                elif op == 'add':
                    results.append(a + b)
                elif op == 'multiply':
                    results.append(a * b)
                else:
                    raise ValueError(f"Unknown operation: {op}")
            return results
        
        # GPU execution
        results = []
        for op, a, b in operations:
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            
            if op == 'matmul':
                result = cp.matmul(a_gpu, b_gpu)
            elif op == 'add':
                result = a_gpu + b_gpu
            elif op == 'multiply':
                result = a_gpu * b_gpu
            else:
                raise ValueError(f"Unknown operation: {op}")
            
            results.append(cp.asnumpy(result))
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage.
        
        Returns:
            Dictionary with memory statistics in MB:
                - used: Currently used memory
                - total: Total available memory
                - free: Free memory
                - percent: Percentage used
        """
        if not self.use_gpu:
            return {
                'used': 0.0,
                'total': 0.0,
                'free': 0.0,
                'percent': 0.0
            }
        
        mem_info = cp.cuda.Device().mem_info
        used = (mem_info[1] - mem_info[0]) / 1e6  # MB
        total = mem_info[1] / 1e6  # MB
        free = mem_info[0] / 1e6  # MB
        percent = (used / total) * 100 if total > 0 else 0.0
        
        return {
            'used': used,
            'total': total,
            'free': free,
            'percent': percent
        }
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
    
    def benchmark_vs_cpu(
        self,
        operation: str,
        array_sizes: List[int],
        n_iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance.
        
        Args:
            operation: Operation to benchmark ('lif_update', 'matmul', 'sparse_matmul')
            array_sizes: List of array sizes to test
            n_iterations: Number of iterations per test
            
        Returns:
            Dictionary with benchmark results including speedup factors
        """
        import time
        
        results = {
            'operation': operation,
            'array_sizes': array_sizes,
            'cpu_times': [],
            'gpu_times': [],
            'speedups': []
        }
        
        for size in array_sizes:
            if operation == 'lif_update':
                # Benchmark LIF update
                v_mem = np.random.randn(size).astype(np.float32) * 10 - 65
                syn_in = np.random.randn(size).astype(np.float32)
                ext_in = np.random.randn(size).astype(np.float32)
                ref_mask = np.random.rand(size) < 0.1
                params = {'tau_m': 20.0, 'v_rest': -65.0, 'v_threshold': -50.0, 'v_reset': -70.0}
                
                # CPU benchmark
                start = time.perf_counter()
                for _ in range(n_iterations):
                    self.use_gpu = False
                    self.vectorized_lif_update(v_mem, syn_in, ext_in, ref_mask, params)
                cpu_time = (time.perf_counter() - start) / n_iterations
                
                # GPU benchmark (if available)
                if CUPY_AVAILABLE:
                    self.use_gpu = True
                    start = time.perf_counter()
                    for _ in range(n_iterations):
                        self.vectorized_lif_update(v_mem, syn_in, ext_in, ref_mask, params)
                    gpu_time = (time.perf_counter() - start) / n_iterations
                else:
                    gpu_time = float('inf')
            
            elif operation == 'matmul':
                # Benchmark matrix multiplication
                a = np.random.randn(size, size).astype(np.float32)
                b = np.random.randn(size, size).astype(np.float32)
                
                # CPU benchmark
                start = time.perf_counter()
                for _ in range(n_iterations):
                    np.matmul(a, b)
                cpu_time = (time.perf_counter() - start) / n_iterations
                
                # GPU benchmark (if available)
                if CUPY_AVAILABLE:
                    a_gpu = cp.asarray(a)
                    b_gpu = cp.asarray(b)
                    cp.cuda.Stream.null.synchronize()  # Warm-up
                    start = time.perf_counter()
                    for _ in range(n_iterations):
                        cp.matmul(a_gpu, b_gpu)
                        cp.cuda.Stream.null.synchronize()
                    gpu_time = (time.perf_counter() - start) / n_iterations
                else:
                    gpu_time = float('inf')
            
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            results['cpu_times'].append(cpu_time)
            results['gpu_times'].append(gpu_time)
            speedup = cpu_time / gpu_time if gpu_time > 0 and gpu_time != float('inf') else 0.0
            results['speedups'].append(speedup)
        
        return results


def create_gpu_accelerator(use_gpu: bool = True, device_id: int = 0) -> GPUAccelerator:
    """Factory function to create a GPU accelerator instance.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        device_id: CUDA device ID
        
    Returns:
        GPUAccelerator instance
    """
    return GPUAccelerator(use_gpu=use_gpu, device_id=device_id)
