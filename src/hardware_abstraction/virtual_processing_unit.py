"""Virtual Processing Unit (VPU) for parallel neuronal computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..brain_model import BrainModel


class VirtualProcessingUnit:
    """Virtual Processing Unit that processes a 4D slice of neurons in parallel.
    
    Each VPU is responsible for processing a batch of neurons in a specific
    4D slice of the lattice. It operates in sync with a global virtual clock,
    processing all neurons in its slice during each clock cycle.
    
    Attributes:
        vpu_id: Unique identifier for this VPU
        clock_speed: Virtual clock speed in Hz (e.g., 20e6 for 20 MHz)
        assigned_slices: List of 4D coordinate slices assigned to this VPU
        neuron_batch: Batch of neuron IDs in this slice
        synapse_batch: Batch of synapse data for this slice
        statistics: Performance statistics for this VPU
    """
    
    def __init__(self, vpu_id: int, clock_speed_hz: float = 20e6):
        """Initialize a Virtual Processing Unit.
        
        Args:
            vpu_id: Unique identifier for this VPU
            clock_speed_hz: Virtual clock speed in Hz (default: 20 MHz)
        """
        self.vpu_id = vpu_id
        self.clock_speed = clock_speed_hz
        self.assigned_slices: List[Tuple[int, int, int, int, int, int, int, int]] = []
        self.neuron_batch: List[int] = []
        self.synapse_batch: List[Tuple[int, int, float]] = []
        
        # Statistics
        self.statistics = {
            "neurons_processed": 0,
            "spikes_generated": 0,
            "cycles_executed": 0,
            "processing_time_ms": 0.0,
        }
        
        # Internal state
        self._model: Optional[BrainModel] = None
        self._simulation = None
        self._inputs_cache = {}
        self._outputs_cache = {}
        
    def assign_slice(self, slice_bounds: Tuple[int, int, int, int, int, int, int, int]) -> None:
        """Assign a 4D slice to this VPU.
        
        Args:
            slice_bounds: 8-tuple defining the slice bounds:
                (x_min, x_max, y_min, y_max, z_min, z_max, w_min, w_max)
        """
        self.assigned_slices.append(slice_bounds)
        
    def initialize_batch(self, model: BrainModel, simulation) -> None:
        """Initialize neuron and synapse batches from the brain model.
        
        Args:
            model: The brain model containing neurons and synapses
            simulation: The simulation instance for processing
        """
        self._model = model
        self._simulation = simulation
        
        # Collect all neurons in assigned slices
        self.neuron_batch = []
        for slice_bounds in self.assigned_slices:
            x_min, x_max, y_min, y_max, z_min, z_max, w_min, w_max = slice_bounds
            
            # Find neurons in this slice
            for neuron_id, neuron in model.neurons.items():
                x, y, z, w = neuron.x, neuron.y, neuron.z, neuron.w
                if (x_min <= x <= x_max and 
                    y_min <= y <= y_max and 
                    z_min <= z <= z_max and 
                    w_min <= w <= w_max):
                    self.neuron_batch.append(neuron_id)
        
        # Collect synapses related to neurons in this batch
        self.synapse_batch = []
        for neuron_id in self.neuron_batch:
            for synapse in model.get_synapses_for_neuron(neuron_id, direction="post"):
                self.synapse_batch.append((synapse.pre_id, synapse.post_id, synapse.weight))
    
    def gather_inputs(self, global_clock_cycle: int) -> dict:
        """Gather all inputs for neurons in this slice.
        
        Args:
            global_clock_cycle: Current global clock cycle
            
        Returns:
            Dictionary mapping neuron IDs to their input currents
        """
        inputs = {}
        
        for neuron_id in self.neuron_batch:
            # Start with external input
            neuron = self._model.neurons.get(neuron_id)
            if neuron is not None:
                inputs[neuron_id] = neuron.external_input
        
        return inputs
    
    def process_cycle(self, global_clock_cycle: int) -> dict:
        """Process one complete clock cycle for all neurons in this VPU.
        
        This is the main processing function that:
        1. Gathers inputs from connected neurons
        2. Updates all neurons in parallel (vectorized)
        3. Collects outputs for distribution
        4. Updates statistics
        
        Args:
            global_clock_cycle: Current global clock cycle number
            
        Returns:
            Dictionary with processing results including spikes and statistics
        """
        import time
        start_time = time.time()
        
        # Step 1: Gather inputs
        inputs = self.gather_inputs(global_clock_cycle)
        
        # Step 2: Process all neurons in batch (vectorized)
        spikes = []
        for neuron_id in self.neuron_batch:
            if self._simulation is not None:
                # Use simulation's lif_step for each neuron
                did_spike = self._simulation.lif_step(neuron_id)
                if did_spike:
                    spikes.append(neuron_id)
        
        # Step 3: Scatter outputs (spikes are recorded in simulation)
        self._outputs_cache[global_clock_cycle] = spikes
        
        # Step 4: Update statistics
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        self.statistics["neurons_processed"] += len(self.neuron_batch)
        self.statistics["spikes_generated"] += len(spikes)
        self.statistics["cycles_executed"] += 1
        self.statistics["processing_time_ms"] += processing_time
        
        return {
            "vpu_id": self.vpu_id,
            "cycle": global_clock_cycle,
            "neurons_processed": len(self.neuron_batch),
            "spikes": len(spikes),
            "processing_time_ms": processing_time,
        }
    
    def scatter_outputs(self, global_clock_cycle: int) -> None:
        """Distribute outputs to target slices.
        
        This is called after processing to ensure outputs are available
        for the next cycle's inputs.
        
        Args:
            global_clock_cycle: Current global clock cycle
        """
        # Outputs are already recorded in spike history by simulation
        # This method is here for future extensions where we might
        # need explicit inter-VPU communication
        pass
    
    def wait_for_clock_barrier(self) -> None:
        """Wait for global clock synchronization barrier.
        
        This ensures all VPUs complete their cycle before proceeding.
        The actual barrier wait is handled by GlobalVirtualClock.
        """
        pass
    
    def get_statistics(self) -> dict:
        """Get performance statistics for this VPU.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = self.statistics.copy()
        if stats["cycles_executed"] > 0:
            stats["avg_processing_time_ms"] = (
                stats["processing_time_ms"] / stats["cycles_executed"]
            )
            stats["neurons_per_second"] = (
                stats["neurons_processed"] / (stats["processing_time_ms"] / 1000.0)
                if stats["processing_time_ms"] > 0 else 0
            )
        else:
            stats["avg_processing_time_ms"] = 0.0
            stats["neurons_per_second"] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.statistics = {
            "neurons_processed": 0,
            "spikes_generated": 0,
            "cycles_executed": 0,
            "processing_time_ms": 0.0,
        }
